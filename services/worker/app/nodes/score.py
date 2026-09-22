"""Candidate scorer — computes the four critic signals for every candidate
before `nodes/critic.py` selects among them. Kept as its own module so
`choose_best_candidates` only ever needs to read `candidate.scores`, unaware
of how they got there.
"""

from __future__ import annotations

import numpy as np
from storage import Storage

from app.critic.alignment import score_alignment
from app.critic.detail import score_detail
from app.critic.guards import has_detected_text
from app.critic.identity import score_identity
from app.critic.judge import judge_panel
from app.critic.style import score_style_from_embedding
from app.preference.features import pixel_features
from contracts import Candidate, DayState, LookCard, Panel
from ml.identity.faces import decode_jpeg_bgr
from ml.identity.style_score import embed_image


async def score_one_candidate(
    candidate: Candidate,
    panel: Panel,
    *,
    storage: Storage,
    look_card: LookCard,
    reference_embedding: np.ndarray | None = None,
    reference_bank: np.ndarray | None = None,
) -> Candidate:
    """Fetches `candidate`'s image bytes back from storage and runs all four
    critic signals plus the text-detection guard, returning a new
    `Candidate` with `scores`/`face_box` populated. Alignment is scored
    against `panel.action` + `panel.caption_a`."""
    image_bytes = storage.get_object_by_url(candidate.url)
    identity_score, face_box = await score_identity(
        image_bytes, look_card, reference_embedding=reference_embedding
    )

    image_bgr = decode_jpeg_bgr(image_bytes)
    # One DINOv2 pass per image: its vector feeds the style signal and is kept, for the
    # personal ranking model, as the candidate's embedding.
    embedding = embed_image(image_bgr)
    style = score_style_from_embedding(embedding, reference_bank=reference_bank)
    alignment = score_alignment(image_bgr, f"{panel.action}. {panel.caption_a}")
    detail = score_detail(image_bgr, face_box)
    text_detected = has_detected_text(image_bgr)

    # Keep what generation already recorded (which axis this candidate varied
    # along, and the expression level asked for) — see `preference/axes.py`.
    scores = {
        **candidate.scores,
        **pixel_features(image_bgr, face_box),
        "identity": identity_score,
        "style": style,
        "alignment": alignment,
        "detail": detail,
        "text_detected": 1.0 if text_detected else 0.0,
    }
    return candidate.model_copy(
        update={
            "scores": scores,
            "face_box": face_box,
            "embedding": [round(float(x), 5) for x in embedding],
        }
    )


async def judge_candidates(
    panel: Panel,
    candidates: list[Candidate],
    *,
    storage: Storage,
    look_card: LookCard | None,
    reference_url: str | None,
) -> tuple[list[Candidate], float]:
    """Adds the vision judge's ratings (`critic/judge.py`) to one panel's already-scored
    candidates. Returns them unchanged, and $0, if the judge is unavailable."""
    ratings, usd = await judge_panel(
        panel, candidates, storage=storage, look_card=look_card, reference_url=reference_url
    )
    if not ratings:
        return candidates, usd
    return [
        c.model_copy(update={"scores": {**c.scores, **ratings[c.id]}}) if c.id in ratings else c
        for c in candidates
    ], usd


async def score_candidates(
    state: DayState,
    *,
    storage: Storage,
    look_card: LookCard,
    reference_embedding: np.ndarray | None = None,
    reference_bank: np.ndarray | None = None,
    reference_url: str | None = None,
) -> DayState:
    """Scores every candidate in `state.candidates` currently missing scores;
    one already scored (e.g. carried through a partial resume) is left
    untouched rather than re-scored at real cost for no reason."""
    if state.script is None:
        return state
    panel_by_id = {p.id: p for p in state.script.panels}

    scored: list[Candidate] = []
    for candidate in state.candidates:
        # "Already scored" means the critic signals exist — generation now
        # pre-fills `scores` with the axis it varied, which must not count.
        if "identity" in candidate.scores:
            scored.append(candidate)
            continue
        panel = panel_by_id.get(candidate.panel_id)
        if panel is None:
            scored.append(candidate)
            continue
        scored.append(
            await score_one_candidate(
                candidate,
                panel,
                storage=storage,
                look_card=look_card,
                reference_embedding=reference_embedding,
                reference_bank=reference_bank,
            )
        )

    # The vision judge, once per panel, over that panel's candidates together.
    judged_usd = 0.0
    by_panel: dict[int, list[Candidate]] = {}
    for c in scored:
        by_panel.setdefault(c.panel_id, []).append(c)
    rated: dict[str, Candidate] = {}
    for panel_id, group in by_panel.items():
        panel = panel_by_id.get(panel_id)
        if panel is None or all(k in group[0].scores for k in ("judge_overall",)):
            continue  # unknown panel, or a resumed job whose candidates were already judged
        group, usd = await judge_candidates(
            panel, group, storage=storage, look_card=look_card, reference_url=reference_url
        )
        judged_usd += usd
        rated.update({c.id: c for c in group})
    scored = [rated.get(c.id, c) for c in scored]
    return state.model_copy(update={"candidates": scored, "cost_usd": state.cost_usd + judged_usd})
