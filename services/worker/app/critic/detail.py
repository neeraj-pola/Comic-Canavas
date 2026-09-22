"""Detail critic — pipeline node 7's background-detail signal.

Uses Canny edge density outside the face box as a proxy for background
detail: a flat, empty background has few edges; a detailed one (furniture,
textures, props) has many. Excludes the face region (when known) so this
measures environment detail specifically, not facial linework already
scored by the identity checklist.
"""

from __future__ import annotations

import cv2
import numpy as np


def score_detail(image_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> float:
    """Returns the fraction of non-face pixels that are edge pixels (Canny),
    in `[0, 1]` — higher means a more detailed background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        mask = np.ones(edges.shape, dtype=bool)
        mask[y1:y2, x1:x2] = False
        edges = edges[mask]

    if edges.size == 0:
        return 0.0
    return float(np.mean(edges > 0))
