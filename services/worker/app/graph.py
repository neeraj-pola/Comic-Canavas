"""Assembles the daily pipeline (beats -> script -> prompts -> generate -> critic -> compose ->
memory) into one LangGraph `StateGraph` over `contracts.DayState`, with Postgres-backed
checkpointing so a job can resume after a crash or be cleanly redone."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from storage import Storage

from app.compose import compose_day
from app.config import get_settings
from app.db import validate_schema_identifier
from app.images.base import IdentityRef
from app.memory import MemoryStore, get_memory_store
from app.nodes.beats import extract_beats
from app.nodes.critic import choose_best_candidates
from app.nodes.generate import generate_one_panel, generate_panels
from app.nodes.prompts import write_prompts, write_single_prompt
from app.nodes.score import judge_candidates, score_candidates, score_one_candidate
from app.nodes.script import DEFAULT_HUMOR_LEVEL, write_script
from app.preference.state import PreferenceState
from contracts import Candidate, DayState, ImagePrompt, LookCard

NODE_ORDER = ("input", "beats", "script", "prompts", "generate", "critic", "compose", "memory")


class AsrNotImplementedError(NotImplementedError):
    """Raised for `source="audio"` days — speech-to-text isn't implemented."""

    def __init__(self) -> None:
        super().__init__("DayGraph: source='audio' needs ASR, not built yet")


@dataclass
class GraphDeps:
    """Everything a day's pipeline nodes need beyond the evolving `DayState` itself."""

    storage: Storage
    identity: IdentityRef
    look_card: LookCard
    memory: MemoryStore | None = None
    reference_embedding: np.ndarray | None = None
    reference_bank: np.ndarray | None = None
    humor_level: int = DEFAULT_HUMOR_LEVEL
    taste_notes: list[str] | None = None
    # The person's learned taste model; `None` for a brand-new user. Inactive models still
    # steer which axis panels explore.
    preference: PreferenceState | None = None


def _timed(node_name: str, state: DayState, start: float) -> dict[str, str]:
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return {**state.versions, f"{node_name}_ms": str(elapsed_ms)}


def _feature_flags() -> dict[str, str]:
    """The champion/challenger env flags active for this job, recorded into `versions` so a
    day/panel can be traced back to what produced it."""
    return {
        "writer_flag": os.environ.get("WRITER", "base"),
        "prompter_flag": os.environ.get("PROMPTER", "base"),
        "reward_head_flag": os.environ.get("REWARD_HEAD", "hand"),
        "generator_flag": os.environ.get("IMAGE_PROVIDER", "leonardo"),
    }


def build_day_graph(deps: GraphDeps) -> StateGraph[DayState, None, DayState, DayState]:
    """Builds the uncompiled graph; `DayGraph` (below) is the real entry point."""

    async def _input(state: DayState) -> DayState:
        start = time.monotonic()
        if state.transcript is not None:
            pass
        elif state.source == "text":
            if state.text is None:
                raise ValueError("DayGraph: source='text' but DayState.text is None")
            state = state.model_copy(update={"transcript": state.text})
        else:
            raise AsrNotImplementedError
        versions = {**_timed("input", state, start), **_feature_flags()}
        return state.model_copy(update={"versions": versions})

    async def _beats(state: DayState) -> DayState:
        start = time.monotonic()
        result = await extract_beats(state, memory=deps.memory)
        return result.model_copy(update={"versions": _timed("beats", result, start)})

    async def _script(state: DayState) -> DayState:
        start = time.monotonic()
        result = await write_script(
            state,
            memory=deps.memory,
            humor_level=deps.humor_level,
            taste_notes=deps.taste_notes,
        )
        return result.model_copy(update={"versions": _timed("script", result, start)})

    async def _prompts(state: DayState) -> DayState:
        start = time.monotonic()
        result = await write_prompts(
            state,
            memory=deps.memory,
            look_card=deps.look_card,
            taste_notes=deps.taste_notes,
        )
        return result.model_copy(update={"versions": _timed("prompts", result, start)})

    async def _generate(state: DayState) -> DayState:
        start = time.monotonic()
        result = await generate_panels(
            state, storage=deps.storage, identity=deps.identity, preference=deps.preference
        )
        return result.model_copy(update={"versions": _timed("generate", result, start)})

    async def _critic(state: DayState) -> DayState:
        start = time.monotonic()
        scored_state = await score_candidates(
            state,
            storage=deps.storage,
            look_card=deps.look_card,
            reference_embedding=deps.reference_embedding,
            reference_bank=deps.reference_bank,
            reference_url=deps.identity.master_url,
        )

        assert state.script is not None  # critic only ever runs after script/prompts
        script = state.script  # narrowed once here — a nested closure can't see the assert above

        # A retry's fresh prompt is recorded too, so a retried panel's `image_prompts` row
        # describes the candidates that actually survived, not the discarded first attempt.
        retried_prompts: list[ImagePrompt] = []

        async def regenerate(panel_id: int) -> list[Candidate]:
            """Rewrites one panel's prompt and generates + scores a fresh batch for it."""
            prompt, _usd = await write_single_prompt(
                state,
                panel_id,
                memory=deps.memory,
                look_card=deps.look_card,
                taste_notes=deps.taste_notes,
            )
            retried_prompts.append(prompt)
            fresh = await generate_one_panel(prompt, storage=deps.storage, identity=deps.identity)
            panel = next(p for p in script.panels if p.id == panel_id)
            rescored = list(
                await asyncio.gather(
                    *(
                        score_one_candidate(
                            c,
                            panel,
                            storage=deps.storage,
                            look_card=deps.look_card,
                            reference_embedding=deps.reference_embedding,
                            reference_bank=deps.reference_bank,
                        )
                        for c in fresh
                    )
                )
            )
            rated, _usd = await judge_candidates(
                panel,
                rescored,
                storage=deps.storage,
                look_card=deps.look_card,
                reference_url=deps.identity.master_url,
            )
            return rated

        result = await choose_best_candidates(
            scored_state, regenerate=regenerate, preference=deps.preference
        )
        if retried_prompts:
            result = result.model_copy(update={"prompts": [*result.prompts, *retried_prompts]})
        return result.model_copy(update={"versions": _timed("critic", result, start)})

    async def _compose(state: DayState) -> DayState:
        start = time.monotonic()
        result = compose_day(state, storage=deps.storage)
        return result.model_copy(update={"versions": _timed("compose", result, start)})

    async def _memory(state: DayState) -> DayState:
        """Re-writes the day's beats with their final panel image URLs, now that compose has
        produced them."""
        start = time.monotonic()
        if state.beats is None or state.script is None:
            return state
        memory_store = deps.memory or get_memory_store()
        panel_by_id = {p.id: p for p in state.script.panels}
        chosen_panel_ids = {c.panel_id for c in state.candidates if c.chosen}
        prefix = f"strips/{state.user_id}/{state.date.isoformat()}"
        panel_urls: dict[str, list[str]] = {}
        for panel_id in chosen_panel_ids:
            panel = panel_by_id.get(panel_id)
            if panel is None:
                continue
            url = deps.storage.get_url(f"{prefix}/panel-{panel_id}.png")
            panel_urls.setdefault(panel.beat_id, []).append(url)
        memory_store.write_beats(state.user_id, state.beats, panel_urls=panel_urls)
        return state.model_copy(update={"versions": _timed("memory", state, start)})

    builder = StateGraph(DayState)
    builder.add_node("input", _input)
    builder.add_node("beats", _beats)
    builder.add_node("script", _script)
    builder.add_node("prompts", _prompts)
    builder.add_node("generate", _generate)
    builder.add_node("critic", _critic)
    builder.add_node("compose", _compose)
    builder.add_node("memory", _memory)

    builder.add_edge(START, "input")
    builder.add_edge("input", "beats")
    builder.add_edge("beats", "script")
    builder.add_edge("script", "prompts")
    builder.add_edge("prompts", "generate")
    builder.add_edge("generate", "critic")
    builder.add_edge("critic", "compose")
    builder.add_edge("compose", "memory")
    builder.add_edge("memory", END)

    return builder


class DayGraph:
    """A compiled daily pipeline, optionally checkpointed to Postgres for resumability."""

    def __init__(
        self,
        *,
        storage: Storage,
        identity: IdentityRef,
        look_card: LookCard,
        memory: MemoryStore | None = None,
        reference_embedding: np.ndarray | None = None,
        reference_bank: np.ndarray | None = None,
        humor_level: int = DEFAULT_HUMOR_LEVEL,
        taste_notes: list[str] | None = None,
        preference: PreferenceState | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        deps = GraphDeps(
            storage=storage,
            identity=identity,
            look_card=look_card,
            memory=memory,
            reference_embedding=reference_embedding,
            reference_bank=reference_bank,
            humor_level=humor_level,
            taste_notes=taste_notes,
            preference=preference,
        )
        self._checkpointer = checkpointer
        self._compiled = build_day_graph(deps).compile(checkpointer=checkpointer)

    async def run(self, state: DayState) -> DayState:
        """Runs the pipeline. A `job_id` with an existing checkpoint resumes from its last
        completed node instead of starting over."""
        if self._checkpointer is None:
            result = await self._compiled.ainvoke(state)
            return DayState.model_validate(result)

        config: RunnableConfig = {"configurable": {"thread_id": state.job_id}}
        snapshot = await self._compiled.aget_state(config)
        graph_input = None if snapshot.values else state
        result = await self._compiled.ainvoke(graph_input, config=config)
        return DayState.model_validate(result)

    async def reset(self, job_id: str) -> None:
        """Deletes a job's checkpoint so its next run starts fresh instead of resuming to the
        end (a genuine redo, not a resume-after-crash — the caller decides which)."""
        if self._checkpointer is not None:
            await self._checkpointer.adelete_thread(job_id)

    async def get_checkpointed_state(self, job_id: str) -> DayState | None:
        """The finished `DayState` for an already-run job, read from its checkpoint."""
        if self._checkpointer is None:
            return None
        config: RunnableConfig = {"configurable": {"thread_id": job_id}}
        snapshot = await self._compiled.aget_state(config)
        if not snapshot.values:
            return None
        return DayState.model_validate(snapshot.values)

    async def run_streaming(self, state: DayState) -> AsyncIterator[tuple[str, DayState]]:
        """Like `run`, but yields `(node_name, state)` as each node completes, for progress
        streaming."""
        graph_input: DayState | None
        if self._checkpointer is None:
            config: RunnableConfig = {}
            graph_input = state
        else:
            config = {"configurable": {"thread_id": state.job_id}}
            snapshot = await self._compiled.aget_state(config)
            graph_input = None if snapshot.values else state

        current = state
        stream = self._compiled.astream(graph_input, config=config, stream_mode="updates")
        async for update in stream:
            for node_name, partial in update.items():
                current = current.model_copy(update=partial)
                yield node_name, current


@asynccontextmanager
async def open_day_graph(
    *,
    storage: Storage,
    identity: IdentityRef,
    look_card: LookCard,
    memory: MemoryStore | None = None,
    reference_embedding: np.ndarray | None = None,
    reference_bank: np.ndarray | None = None,
    humor_level: int = DEFAULT_HUMOR_LEVEL,
    taste_notes: list[str] | None = None,
    preference: PreferenceState | None = None,
    dsn: str | None = None,
    schema: str = "langgraph",
) -> AsyncIterator[DayGraph]:
    """Opens a Postgres-backed checkpointer for the duration of the `async with` block and
    yields a ready `DayGraph`. `schema` keeps LangGraph's own bookkeeping tables separate from
    the app's `checkpoints` table (a different, unrelated table of the same name)."""
    resolved_dsn = dsn or get_settings().database_url
    validate_schema_identifier(schema)
    with psycopg.connect(resolved_dsn) as conn:
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.commit()
    separator = "&" if "?" in resolved_dsn else "?"
    resolved_dsn = f"{resolved_dsn}{separator}options=-csearch_path%3D{schema}"

    async with AsyncPostgresSaver.from_conn_string(resolved_dsn) as checkpointer:
        await checkpointer.setup()
        yield DayGraph(
            storage=storage,
            identity=identity,
            look_card=look_card,
            memory=memory,
            reference_embedding=reference_embedding,
            reference_bank=reference_bank,
            humor_level=humor_level,
            taste_notes=taste_notes,
            preference=preference,
            checkpointer=checkpointer,
        )
