"""Regularization / style reference set: ~150 generic "comic character"
renders in the style card's visual language, no identity conditioning at
all — this is what `style_score.py`'s DINOv2 bank is built from, and a
possible future LoRA regularization set.

File layout: `NNN.jpg` + `NNN.txt` (caption always `comic character`),
zero-padded to 3 digits so 150 files sort lexically in numeric order, plus
a `manifest.json` recording every seed/prompt/subject — an
ai-toolkit-compatible dataset pair layout, on the chance a future
regularization use wants the captions; DINOv2 itself never reads them.

`ml/reg/` is gitignored (no LFS configured); only `manifest.json` and the
derived DINOv2 embedding bank are committed — the images themselves are
regenerable from the manifest's recorded seeds/prompts.

`n` defaults to a small test batch (5), not the full 150 — worth looking
at a handful of real renders and the real billed price before committing
to the full set. `confirm=True` is required before any request is sent.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.llm.cost import CostEvent, get_cost_sink  # noqa: E402
from app.prompts.style_card import StyleCard, load_style_card  # noqa: E402
from ml.identity.fal_jobs import download, submit_and_wait  # noqa: E402

REG_ROOT = _REPO_ROOT / "ml" / "reg" / "comic_character"
CAPTION = "comic character"
MODEL_ID = "fal-ai/flux/dev"
IMAGE_SIZE = "square"  # 512x512 — DINOv2-small ingests 224px; bigger is wasted money
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
# Inferred from fal-ai/flux-lora's own $0.035/megapixel (images/fal_flux.py)
# since flux/dev is the same base model without a LoRA attached — a
# reasonable estimate, not a confirmed price; the real billed amount from
# the small test batch decides whether the full-150 estimate is trustworthy.
USD_PER_MEGAPIXEL = 0.035
SQUARE_MEGAPIXELS = (512 * 512) / 1_000_000
DEFAULT_N = 5
CONCURRENCY = 4

# ~25 generic, identity-free subjects, cycled with a per-image seed so
# ~150 renders are varied without 150 hand-written prompts. Deliberately
# ordinary scenes/poses — this set defines "what does our comic style
# look like on *any* character," not any particular person.
SUBJECTS: list[str] = [
    "a person reading a book on a couch",
    "two friends talking at a cafe table",
    "a person walking a dog in a park",
    "a person cooking at a stove",
    "a person typing at a laptop on a desk",
    "a person waiting at a bus stop",
    "a person watering plants on a balcony",
    "a person riding a bicycle down a street",
    "a person browsing shelves in a bookstore",
    "a person stretching before a morning jog",
    "two coworkers reviewing papers at a table",
    "a person washing dishes at a kitchen sink",
    "a person sitting on a park bench eating lunch",
    "a person unpacking groceries in a kitchen",
    "a person playing guitar on a bedroom floor",
    "a person standing at a bus window looking out",
    "a person folding laundry in a bedroom",
    "a person painting at an easel",
    "a person carrying an umbrella in the rain",
    "a person sitting cross-legged writing in a notebook",
    "a person waiting in line at a counter",
    "a person hiking on a forest trail",
    "a person setting a table for dinner",
    "a person napping in a hammock",
    "a person sweeping a porch",
]


class SpendNotConfirmedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "generate_reg_set: pass confirm=True to spend real money generating images "
            "(see estimate_usd() first)"
        )


class RegSetReport(BaseModel):
    written: list[str]
    skipped: list[str]
    estimated_usd: float
    actual_images: int
    model_id: str
    style_card_version: str


@dataclass(frozen=True)
class _PlannedImage:
    index: int
    subject: str
    seed: int


def estimate_usd(n: int) -> float:
    return SQUARE_MEGAPIXELS * USD_PER_MEGAPIXEL * n


def build_prompt(subject: str, style: StyleCard) -> tuple[str, str]:
    """(positive, negative) — same style_phrase/negative_standard every
    render draws from, matching how `ml/identity/master.py` builds
    prompts."""
    positive = f"{subject}. {style.style_phrase}."
    negative = ", ".join([*style.negative_standard, *style.character.avoid])
    return positive, negative


def _existing_indices(out_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in out_dir.glob("*.jpg") if p.stem.isdigit())


def _plan_images(n: int, *, start_index: int, seed0: int) -> list[_PlannedImage]:
    planned = []
    for offset in range(n):
        index = start_index + offset
        subject = SUBJECTS[index % len(SUBJECTS)]
        seed = seed0 + index
        planned.append(_PlannedImage(index=index, subject=subject, seed=seed))
    return planned


async def _generate_one(
    planned: _PlannedImage,
    *,
    client: httpx.AsyncClient,
    style: StyleCard,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, object]:
    positive, negative = build_prompt(planned.subject, style)
    body = {
        "prompt": positive,
        "negative_prompt": negative,
        "image_size": IMAGE_SIZE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_images": 1,
        "seed": planned.seed,
    }
    async with semaphore:
        result = await submit_and_wait(MODEL_ID, body, client=client)
        images = result.get("images", [])
        if not images:
            raise RuntimeError(f"reg_set image {planned.index}: no images returned")
        data = await download(images[0]["url"], client=client)

    stem = f"{planned.index:03d}"
    (out_dir / f"{stem}.jpg").write_bytes(data)
    (out_dir / f"{stem}.txt").write_text(CAPTION)

    get_cost_sink().record(
        CostEvent(provider="fal", model=MODEL_ID, role="image", units=1, usd=estimate_usd(1))
    )
    return {
        "index": planned.index,
        "subject": planned.subject,
        "seed": planned.seed,
        "prompt": positive,
    }


async def generate_reg_set(
    n: int = DEFAULT_N,
    *,
    out_dir: Path = REG_ROOT,
    start_index: int | None = None,
    seed0: int = 20260914,
    confirm: bool = False,
    style: StyleCard | None = None,
) -> RegSetReport:
    """Generates `n` images starting after the highest existing `NNN.jpg`
    in `out_dir` (so a 5-image test batch and a later 145-image top-up
    together produce one 150-image set), writing each as `NNN.jpg` +
    `NNN.txt` plus a `manifest.json` entry. Raises `SpendNotConfirmedError`
    unless `confirm=True` — no request is sent before that check."""
    if not confirm:
        raise SpendNotConfirmedError
    out_dir.mkdir(parents=True, exist_ok=True)
    style = style or load_style_card()

    existing = _existing_indices(out_dir)
    resolved_start = start_index if start_index is not None else (max(existing, default=-1) + 1)
    planned = _plan_images(n, start_index=resolved_start, seed0=seed0)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    manifest_path = out_dir / "manifest.json"
    manifest: list[dict[str, object]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    async with httpx.AsyncClient(timeout=120.0) as client:
        entries = await asyncio.gather(
            *(
                _generate_one(p, client=client, style=style, out_dir=out_dir, semaphore=semaphore)
                for p in planned
            )
        )

    manifest.extend(entries)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    written = [f"{p.index:03d}.jpg" for p in planned]
    return RegSetReport(
        written=written,
        skipped=[],
        estimated_usd=estimate_usd(len(planned)),
        actual_images=len(planned),
        model_id=MODEL_ID,
        style_card_version=style.version,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", type=Path, default=REG_ROOT)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="confirm real spend, required to run")
    args = parser.parse_args()

    print(f"estimated cost for {args.n} images: ${estimate_usd(args.n):.4f}")
    if not args.yes:
        print("pass --yes to actually generate (no request sent)")
        return

    report = asyncio.run(
        generate_reg_set(args.n, out_dir=args.out, start_index=args.start_index, confirm=True)
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
