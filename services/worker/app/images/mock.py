"""Deterministic mock image generator.

Never makes a network call — used by every test and whenever
`IMAGE_PROVIDER=mock` (the default without real generator API keys).
Renders a small colored placeholder PNG per candidate (background color from
the seed, `panel/seed/generator` text overlaid so candidates are visually
distinguishable during manual testing) and writes it via `Storage.put_object`.
"""

from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image, ImageDraw
from storage import Storage

from app.identity_token import apply_identity
from app.images.base import IdentityRef, variant_tag
from contracts import Candidate, ImagePrompt

CANDIDATE_SIZE = (512, 512)
USD_PER_IMAGE = 0.0  # mock generation is free


def _color_for_seed(seed: int | str) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(seed).encode()).digest()
    return digest[0], digest[1], digest[2]


def _render_placeholder(prompt: ImagePrompt, *, seed: int, resolved_positive: str) -> bytes:
    # Options that share a seed still differ in their prompt, so the placeholder does too.
    image = Image.new("RGB", CANDIDATE_SIZE, _color_for_seed(f"{seed}-{variant_tag(prompt)}"))
    draw = ImageDraw.Draw(image)
    lines = [
        f"panel {prompt.panel_id}",
        f"seed {seed}",
        f"generator {prompt.generator}",
        resolved_positive[:60],
    ]
    draw.multiline_text((16, 16), "\n".join(lines), fill=(255, 255, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class MockImageProvider:
    name = "mock"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]:
        resolved_positive = apply_identity(prompt.positive, identity.trigger_token)
        base_seed = prompt.seed if prompt.seed is not None else 0

        candidates = []
        for i in range(n):
            seed = base_seed + i
            data = _render_placeholder(prompt, seed=seed, resolved_positive=resolved_positive)
            key = f"candidates/panel-{prompt.panel_id}/{seed}-{variant_tag(prompt)}.png"
            url = self.storage.put_object(key, data, content_type="image/png")
            candidates.append(
                Candidate(
                    id=f"panel{prompt.panel_id}-{seed}-{variant_tag(prompt)}",
                    panel_id=prompt.panel_id,
                    url=url,
                    seed=seed,
                )
            )
        return candidates

    def estimate_usd(self, *, n: int) -> float:
        return USD_PER_IMAGE * n
