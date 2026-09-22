"""Face detection, alignment, cropping, and quality filtering.

Uses InsightFace's `buffalo_l` model pack, which bundles both a
RetinaFace-based detector (with 5-point landmarks) and the ArcFace
recognition model in one download, so one dependency and one model load
covers both. Runs on CPU via onnxruntime; the first call downloads ~280MB
to `~/.insightface/models/buffalo_l/` (real network, so keep this out of
anything tests run against a bare cache).

`detect_and_embed`/`decode_jpeg_bgr` are the detect+align+crop+embed core,
extracted out of `process_batch` and exposed without its blur/duplicate
gating: scoring generated candidate renders through `process_batch`'s
blur check misclassifies sharp, in-focus AI images as "blur", since smooth
synthetic skin genuinely produces lower Laplacian variance than real skin
texture. That threshold is calibrated for real camera photos; generated
renders need face detection and an embedding, not a quality gate meant for
someone's camera roll. `process_batch`'s own behavior is unchanged — it
still requires exactly one face and still applies blur/duplicate
filtering, via the same extracted helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import cv2
import imagehash
import numpy as np
import pillow_heif
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()  # type: ignore[attr-defined]  # lets PIL.Image.open decode .heic/.heif

PhotoStatus = Literal["usable", "blur", "duplicate", "no_face", "multi_face"]

CROP_SIZE = 512

# Empirical thresholds: reasonable starting points from common
# blur/duplicate-detection practice, not derived from any formula specific
# to this project; expect to retune them once more onboarding batches exist.
BLUR_VARIANCE_THRESHOLD = 80.0
DUPLICATE_HAMMING_THRESHOLD = 6


@dataclass
class FacePhoto:
    """One intake photo's classification result. `crop_bgr`/`embedding`
    are only meaningful once a face was found — `blur`/`duplicate` still
    carry a crop (so a caller can show *why* it was rejected), but
    `no_face`/`multi_face` have nothing to align in the first place."""

    path: Path
    status: PhotoStatus
    blur_score: float | None = None
    crop_bgr: np.ndarray | None = None  # (512, 512, 3) uint8, aligned
    embedding: np.ndarray | None = None  # (512,) float32, ArcFace


class DetectedFace(Protocol):
    """The subset of insightface's `Face` object this module actually
    uses — narrowed to a Protocol so tests can inject a fake detector
    without loading the real ~280MB buffalo_l model (same "mock the
    dependency, not the transport" approach as the LLM/image provider
    tests)."""

    kps: np.ndarray  # (5, 2) landmark points
    embedding: np.ndarray  # (512,) float32


class FaceDetector(Protocol):
    def get(self, img: np.ndarray) -> list[DetectedFace]: ...


_app: FaceAnalysis | None = None


def _get_app() -> FaceAnalysis:
    """Lazy singleton — loading buffalo_l is expensive enough (model
    load + first download) that a batch of photos should share one
    instance rather than reloading per photo."""
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def load_bgr(path: Path) -> np.ndarray:
    """PIL (with the HEIC/HEIF plugin registered above) handles every
    format onboarding accepts (jpg/png/heic — most phone photos land here
    as HEIC); cv2's own functions expect BGR channel order. Public since
    `ml/evals/critic_style.py` also needs to load raw onboarding photos
    the same way `process_batch` does.

    `exif_transpose` matters generally: phones commonly store pixels in
    the sensor's native orientation with an EXIF tag saying how to
    *display* it upright, rather than physically rotating the data —
    skip this and the detector sees a face lying on its side and fails.

    It won't save every sideways photo, though: one real onboarding
    photo (a graduation shot) has orientation tag `1` ("no rotation
    needed") despite the pixel data itself genuinely being sideways —
    likely re-saved/re-exported by something that reset the tag without
    re-encoding the pixels. No metadata exists to auto-correct that case;
    it correctly comes back `no_face`, not a bug in this function.
    """
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _blur_variance(crop_bgr: np.ndarray) -> float:
    """Laplacian variance of the aligned crop — a standard, cheap blur
    proxy: a sharp image has strong edges (high-variance second
    derivative), a blurry one doesn't."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _align_crop(image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
    return cast(
        np.ndarray, face_align.norm_crop(image_bgr, landmark=face.kps, image_size=CROP_SIZE)
    )


def _face_size(face: DetectedFace) -> float:
    """A bbox-area proxy from landmark spread. `DetectedFace` is
    deliberately narrowed to `kps`/`embedding` (so tests can fake it
    without insightface's real `Face` object, which also carries a
    `bbox`) — this reconstructs "how big is this face" from the one
    spatial field the Protocol does carry, rather than widening it."""
    xs, ys = face.kps[:, 0], face.kps[:, 1]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _largest_face(faces: list[DetectedFace]) -> DetectedFace:
    return max(faces, key=_face_size)


@dataclass
class DetectedFaceResult:
    """`detect_and_embed`'s result — the largest face found, aligned and
    embedded, plus how many faces were actually in the image."""

    crop_bgr: np.ndarray  # (512, 512, 3) uint8, aligned
    embedding: np.ndarray  # (512,) float32, ArcFace
    face_count: int  # > 1 is an artifact signal for callers scoring generated images


def detect_and_embed(
    image_bgr: np.ndarray, *, detector: FaceDetector | None = None
) -> DetectedFaceResult | None:
    """The core of `process_batch`, without blur/duplicate gating — see
    this module's docstring for why generated-image scoring needs this
    instead. Returns `None` only when no face is found at all; a
    multi-face image still returns a result (the largest face, plus
    `face_count`) so the caller can decide whether that's disqualifying for
    its own purpose, rather than this function deciding for every caller
    the way `process_batch`'s stricter intake gate does."""
    app = detector or _get_app()
    faces = app.get(image_bgr)
    if not faces:
        return None
    face = _largest_face(faces)
    crop = _align_crop(image_bgr, face)
    return DetectedFaceResult(crop_bgr=crop, embedding=face.embedding, face_count=len(faces))


def largest_face_kps(
    image_bgr: np.ndarray, *, detector: FaceDetector | None = None
) -> np.ndarray | None:
    """The largest detected face's 5-point landmarks, in the original
    image's coordinate space — unlike `detect_and_embed`, which only
    returns the aligned, cropped face. A face-mask weighted loss needs to
    know where the face sits in the full, uncropped training image to draw
    a loss-weight mask over it. Returns `None` if no face is found, same
    convention as `detect_and_embed`."""
    app = detector or _get_app()
    faces = app.get(image_bgr)
    if not faces:
        return None
    return _largest_face(faces).kps


def face_bbox_from_kps(shape: tuple[int, int], kps: np.ndarray) -> tuple[int, int, int, int]:
    """A clamped rectangular face region from 5-point landmarks — `kps`
    spans only the central face (eyes to mouth), so padded generously for
    forehead/cheeks/chin/ears. Shared by `images/refine_face.py` and
    `critic/identity.py` (which needs `Candidate.face_box` for the
    composer's bubble-avoidance) rather than duplicated in each."""
    height, width = shape[:2]
    xs, ys = kps[:, 0], kps[:, 1]
    center_x, center_y = float(xs.mean()), float(ys.mean())
    spread_x = float(xs.max() - xs.min())
    spread_y = float(ys.max() - ys.min())
    half_w = max(spread_x * 1.8, spread_y * 1.4, 1.0)
    half_h = max(spread_y * 2.2, spread_x * 1.6, 1.0)
    center_y -= half_h * 0.15
    x1 = max(0, int(center_x - half_w))
    y1 = max(0, int(center_y - half_h))
    x2 = min(width, int(center_x + half_w))
    y2 = min(height, int(center_y + half_h))
    return x1, y1, x2, y2


def detect_face_box(
    image_bgr: np.ndarray, *, detector: FaceDetector | None = None
) -> tuple[int, int, int, int] | None:
    """`largest_face_kps` + `face_bbox_from_kps` in one call — `None` if
    no face is detected (the same InsightFace/RetinaFace limitation on
    stylized art documented elsewhere in this codebase)."""
    kps = largest_face_kps(image_bgr, detector=detector)
    if kps is None:
        return None
    return face_bbox_from_kps(image_bgr.shape, kps)


def decode_jpeg_bgr(data: bytes) -> np.ndarray:
    """bytes -> BGR array, the inverse of `encode_crop_jpeg`. Downloaded
    candidate renders arrive as bytes from an HTTP response, never as a
    `Path` on disk — `load_bgr`'s file/HEIC/EXIF handling doesn't apply to
    them."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image bytes")
    return image


def process_batch(paths: list[Path], *, detector: FaceDetector | None = None) -> list[FacePhoto]:
    """Classifies every photo as usable/blur/duplicate/no_face/multi_face.

    Order matters for duplicates: each photo is only compared against
    photos *already accepted earlier in `paths`* — the first of a
    near-identical pair stays usable, later ones get marked duplicate
    (mirrors reviewing a camera roll and keeping one shot from a burst,
    not an arbitrary pick).

    `detector` defaults to the real (lazily-loaded) buffalo_l app; tests
    inject a fake `FaceDetector` instead.
    """
    app = detector or _get_app()
    results: list[FacePhoto] = []
    accepted_hashes: list[imagehash.ImageHash] = []

    for path in paths:
        bgr = load_bgr(path)
        faces = app.get(bgr)

        if len(faces) == 0:
            results.append(FacePhoto(path=path, status="no_face"))
            continue
        if len(faces) > 1:
            results.append(FacePhoto(path=path, status="multi_face"))
            continue

        face = faces[0]
        crop = _align_crop(bgr, face)
        blur_score = _blur_variance(crop)

        if blur_score < BLUR_VARIANCE_THRESHOLD:
            results.append(
                FacePhoto(path=path, status="blur", blur_score=blur_score, crop_bgr=crop)
            )
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        photo_hash = imagehash.phash(Image.fromarray(crop_rgb))
        if any(photo_hash - seen <= DUPLICATE_HAMMING_THRESHOLD for seen in accepted_hashes):
            results.append(
                FacePhoto(path=path, status="duplicate", blur_score=blur_score, crop_bgr=crop)
            )
            continue

        accepted_hashes.append(photo_hash)
        results.append(
            FacePhoto(
                path=path,
                status="usable",
                blur_score=blur_score,
                crop_bgr=crop,
                embedding=face.embedding,
            )
        )

    return results


def encode_crop_jpeg(crop_bgr: np.ndarray, *, quality: int = 92) -> bytes:
    """JPEG-encode a crop for `Storage.put_object` — crops live under
    `people/{id}/crops/`, mirroring `raw/`'s convention."""
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("failed to encode crop as JPEG")
    return buf.tobytes()
