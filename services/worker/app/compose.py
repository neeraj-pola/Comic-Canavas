"""Composer — turns a day's chosen panel candidates into shareable artifacts:
a flattened `strip.png` grid with date header, captions and speech bubbles
baked in, a vertical `story.png`, one annotated PNG per panel, and a
`layout.json` recording where everything landed in pixel space.

`caption_a` is the displayed caption; `caption_b` is reserved for future A/B
caption experiments and isn't rendered here.

Bubble placement (`_bubble_box`) tries the 4 cell corners in a fixed order and
returns the first with zero overlap against the panel's face box. Given the
bubble size fractions and `MAX_FACE_FRAC` (the largest face box this
guarantee assumes), at least one corner is always fully clear; if a real face
box exceeds that bound, it falls back to the least-overlapping corner instead
of crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
from storage import Storage

from contracts import Candidate, DayState, Panel, Script

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
INTER_PATH = FONTS_DIR / "Inter-Variable.ttf"
MONO_PATH = FONTS_DIR / "JetBrainsMono-Variable.ttf"
# Comic-lettering font (SIL OFL) for in-panel captions/bubbles; Inter/JetBrains
# Mono are used for the surrounding UI chrome (date header, mood pill).
# Bangers is an all-caps display face, so text through `_comic()` is uppercased.
COMIC_PATH = FONTS_DIR / "Bangers-Regular.ttf"

# Brand tokens shared with the frontend's design system.
INK = (21, 21, 21)
INK_TEXT_ON_MUSTARD = (30, 30, 28)  # "charcoal" — used on the mustard pill
FIELD = (247, 246, 241)
MUSTARD = (233, 210, 74)
RULE = (221, 219, 211)
WHITE = (255, 255, 255)

STRIP_SIZE = 2160
STORY_SIZE = (1080, 1920)
GUTTER = 16
DATE_HEADER_HEIGHT = 180
CAPTION_BAR_FRAC = 0.14  # of cell height
CAPTION_PADDING = 24

BUBBLE_WIDTH_FRAC = 0.20
BUBBLE_HEIGHT_FRAC = 0.12
BUBBLE_MARGIN_FRAC = 0.03
MAX_FACE_FRAC = 0.55  # the largest face box the bubble-placement guarantee assumes

Box = tuple[int, int, int, int]


class ComposeError(ValueError):
    pass


class MissingPanelImageError(ComposeError):
    def __init__(self, panel_id: int) -> None:
        super().__init__(f"no image bytes provided for panel {panel_id}")


@dataclass
class ComposedStrip:
    strip_png: bytes
    story_png: bytes
    panel_pngs: dict[int, bytes]
    layout: dict[str, Any]


def _font(
    path: Path, size: int, *, weight: int = 400, opsz: int | None = None
) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), size)
    try:
        axes = font.get_variation_axes()
    except OSError:
        return font
    if len(axes) >= 2:
        resolved_opsz = opsz if opsz is not None else max(14, min(32, size // 3))
        font.set_variation_by_axes([resolved_opsz, weight])
    return font


def _inter(size: int, *, weight: int = 400) -> ImageFont.FreeTypeFont:
    return _font(INTER_PATH, size, weight=weight)


def _mono(size: int, *, weight: int = 400) -> ImageFont.FreeTypeFont:
    return _font(MONO_PATH, size, weight=weight)


def _comic(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(COMIC_PATH), size)


def _fit_square(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def _contain_on_background(
    image: Image.Image, size: tuple[int, int], background: tuple[int, int, int]
) -> Image.Image:
    """Scales `image` down to fit within `size` preserving aspect ratio (never
    cropping), centered on a `background`-filled canvas of exactly `size`.
    Used for the story render, where a hard center-crop would cut off faces
    in short, wide cells; letterboxing keeps the whole panel visible."""
    fitted = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _scale_face_box_contain(
    face_box: Box, *, from_size: tuple[int, int], to_size: tuple[int, int]
) -> Box:
    """The `_contain_on_background` counterpart to `_scale_face_box`
    (which assumes a center-crop transform instead)."""
    src_w, src_h = from_size
    dst_w, dst_h = to_size
    scale = min(dst_w / src_w, dst_h / src_h)
    offset_x = (dst_w - src_w * scale) / 2
    offset_y = (dst_h - src_h * scale) / 2
    x1, y1, x2, y2 = face_box
    return (
        round(x1 * scale + offset_x),
        round(y1 * scale + offset_y),
        round(x2 * scale + offset_x),
        round(y2 * scale + offset_y),
    )


def _scale_face_box(face_box: Box, *, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    """`ImageOps.fit` center-crops then scales; a `face_box` measured on
    the original candidate image must go through the same transform to
    stay aligned with the resized cell it's used against."""
    src_w, src_h = from_size
    dst_w, dst_h = to_size
    src_aspect = src_w / src_h
    dst_aspect = dst_w / dst_h

    if src_aspect > dst_aspect:
        crop_h = src_h
        crop_w = round(dst_aspect * crop_h)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        crop_w = src_w
        crop_h = round(crop_w / dst_aspect)
        crop_x = 0
        crop_y = (src_h - crop_h) // 2

    scale = dst_w / crop_w
    x1, y1, x2, y2 = face_box
    scaled = (
        round((x1 - crop_x) * scale),
        round((y1 - crop_y) * scale),
        round((x2 - crop_x) * scale),
        round((y2 - crop_y) * scale),
    )
    return (
        max(0, min(dst_w, scaled[0])),
        max(0, min(dst_h, scaled[1])),
        max(0, min(dst_w, scaled[2])),
        max(0, min(dst_h, scaled[3])),
    )


def _overlap_area(a: Box, b: Box) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    return ox * oy


def _bubble_box(face_box: Box | None, cell_size: tuple[int, int]) -> Box:
    """Picks one of the 4 cell corners for the speech bubble, preferring (in
    order) top-right, top-left, bottom-right, bottom-left — the corner with
    zero overlap against `face_box`, or the least-overlapping one if every
    corner overlaps."""
    w, h = cell_size
    bw = int(w * BUBBLE_WIDTH_FRAC)
    bh = int(h * BUBBLE_HEIGHT_FRAC)
    margin = int(min(w, h) * BUBBLE_MARGIN_FRAC)

    corners: list[Box] = [
        (w - margin - bw, margin, w - margin, margin + bh),  # top-right
        (margin, margin, margin + bw, margin + bh),  # top-left
        (w - margin - bw, h - margin - bh, w - margin, h - margin),  # bottom-right
        (margin, h - margin - bh, margin + bw, h - margin),  # bottom-left
    ]

    if face_box is None:
        return corners[0]

    scored = [(corner, _overlap_area(corner, face_box)) for corner in corners]
    scored.sort(key=lambda pair: pair[1])
    return scored[0][0]


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_size: int,
    min_size: int,
    font_fn: Any = _comic,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Shrinks font size until `text` fits one line within `max_width`;
    if even `min_size` doesn't fit, wraps into two lines at `min_size`."""
    for size in range(max_size, min_size - 1, -4):
        font = font_fn(size)
        if draw.textlength(text, font=font) <= max_width:
            return font, [text]

    font = font_fn(min_size)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return font, lines[:2]


def _draw_caption_bar(image: Image.Image, caption: str) -> Box:
    w, h = image.size
    bar_h = int(h * CAPTION_BAR_FRAC)
    bar_box: Box = (0, h - bar_h, w, h)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bar_box, fill=(*INK, 235))

    if caption:
        font, lines = _fit_text_to_width(
            draw,
            caption.upper(),
            max_width=w - 2 * CAPTION_PADDING,
            max_size=max(20, int(bar_h * 0.7)),
            min_size=16,
        )
        line_height = font.size + 6
        total_h = line_height * len(lines)
        y = bar_box[1] + (bar_h - total_h) // 2
        for line in lines:
            draw.text((CAPTION_PADDING, y), line, font=font, fill=(*WHITE, 255))
            y += line_height

    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    return bar_box


def _draw_bubble(image: Image.Image, bubble_box: Box, text: str) -> None:
    if not text:
        return
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bubble_box
    radius = min(16, (y2 - y1) // 2)
    draw.rounded_rectangle(bubble_box, radius=radius, fill=WHITE, outline=INK, width=3)

    font, lines = _fit_text_to_width(
        draw,
        text.upper(),
        max_width=(x2 - x1) - 16,
        max_size=max(14, (y2 - y1) // 2),
        min_size=12,
    )
    line_height = font.size + 2
    total_h = line_height * len(lines)
    y = y1 + ((y2 - y1) - total_h) // 2
    for line in lines:
        line_w = draw.textlength(line, font=font)
        x = x1 + ((x2 - x1) - line_w) // 2
        draw.text((x, y), line, font=font, fill=INK)
        y += line_height


def render_panel_cell(
    image_bytes: bytes,
    panel: Panel,
    face_box: Box | None,
    *,
    size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Renders one panel (source image + caption bar + optional speech
    bubble) into a `size` x `size` square. Returns the image and a
    layout dict with the caption/bubble boxes in this cell's own local
    pixel space (the caller offsets them when placing the cell)."""
    source = Image.open(BytesIO(image_bytes))
    scaled_face_box = (
        _scale_face_box(face_box, from_size=source.size, to_size=(size, size))
        if face_box is not None
        else None
    )
    cell = _fit_square(source, size)

    caption_box = _draw_caption_bar(cell, panel.caption_a)

    bubble_box: Box | None = None
    if panel.bubble:
        available_h = caption_box[1]  # bubble must stay above the caption bar
        bubble_box = _bubble_box(scaled_face_box, (size, available_h))
        _draw_bubble(cell, bubble_box, panel.bubble)

    layout = {
        "caption_box": list(caption_box),
        "bubble_box": list(bubble_box) if bubble_box else None,
        "face_box": list(scaled_face_box) if scaled_face_box else None,
    }
    return cell, layout


def render_panel_png(
    image_bytes: bytes, panel: Panel, face_box: Box | None, *, size: int = 1024
) -> bytes:
    cell, _layout = render_panel_cell(image_bytes, panel, face_box, size=size)
    buf = BytesIO()
    cell.save(buf, format="PNG")
    return buf.getvalue()


def _grid_shape(n_panels: int) -> tuple[int, int]:
    """Rows/cols per panel count. 6 (weekly recap) is 3 columns x 2 rows,
    matching the frontend's own grid; 3 panels use a 2x2 grid whose 4th cell
    is the day card (see `_strip_geometry`)."""
    return {2: (1, 2), 3: (2, 2), 4: (2, 2), 6: (2, 3)}.get(n_panels, (1, n_panels))


def _strip_geometry(n_panels: int) -> tuple[list[Box], Box | None, int]:
    """Where everything goes on the strip canvas: `(panel boxes in reading
    order, the day card's box or None, canvas height)`.

    Panels are always square cells, and the grid is centered horizontally.
    3 panels use a 2x2 grid with a square day card in the 4th cell; 2 panels
    are one row with the day card as a full-width band below; 6 panels (weekly
    recap) are 3x2 on a canvas sized to fit; 4 panels are 2x2, centered.
    """
    width, top, gutter = STRIP_SIZE, DATE_HEADER_HEIGHT, GUTTER
    rows, cols = _grid_shape(n_panels)
    cell = min(
        (width - gutter * (cols + 1)) // cols,
        (STRIP_SIZE - top - gutter * (rows + 1)) // rows,
    )
    grid_w = cols * cell + (cols - 1) * gutter
    grid_h = rows * cell + (rows - 1) * gutter
    x0 = (width - grid_w) // 2

    canvas_h = STRIP_SIZE
    if n_panels == 6:
        canvas_h = top + gutter + grid_h + gutter
    y0 = top + gutter if n_panels in (2, 6) else top + (canvas_h - top - grid_h) // 2

    boxes: list[Box] = []
    for i in range(n_panels):
        row, col = divmod(i, cols)
        x = x0 + col * (cell + gutter)
        y = y0 + row * (cell + gutter)
        boxes.append((x, y, x + cell, y + cell))

    card: Box | None = None
    if n_panels == 3:
        x = x0 + (cell + gutter)
        y = y0 + (cell + gutter)
        card = (x, y, x + cell, y + cell)
    elif n_panels == 2:
        y = y0 + cell + gutter
        card = (x0, y, x0 + grid_w, canvas_h - gutter)
    return boxes, card, canvas_h


def _render_day_card(width: int, height: int, *, mood: str, day: date) -> Image.Image:
    """Filler cell for 2- and 3-panel days: the day's mood in big comic
    lettering on mustard, so a shorter strip fills its canvas without
    stretching or cropping a panel."""
    card = Image.new("RGB", (width, height), MUSTARD)
    draw = ImageDraw.Draw(card)
    dot = (222, 196, 56)  # a slightly darker mustard: halftone dots, like the panel art
    pitch = max(24, min(width, height) // 34)
    for y in range(pitch // 2, height, pitch):
        for x in range(pitch // 2 + (pitch // 2 if (y // pitch) % 2 else 0), width, pitch):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=dot)
    draw.rectangle((0, 0, width - 1, height - 1), outline=INK, width=8)

    margin = int(min(width, height) * 0.09)
    max_w = width - 2 * margin
    max_h = int(height * 0.62)
    text = mood.strip().upper() or "A DAY"

    def wrapped(font: ImageFont.FreeTypeFont) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= max_w or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        return [*lines, current] if current else lines

    font = _comic(40)
    lines = wrapped(font)
    for size in range(int(min(width, height) * 0.28), 39, -6):
        font = _comic(size)
        lines = wrapped(font)
        fits_w = all(draw.textlength(line, font=font) <= max_w for line in lines)
        if fits_w and len(lines) * int(size * 1.08) <= max_h:
            break

    line_h = int(font.size * 1.08)
    y = (height - len(lines) * line_h) // 2 - margin // 3
    for line in lines:
        draw.text((width // 2, y + line_h // 2), line, font=font, fill=INK, anchor="mm")
        y += line_h
    label_font = _mono(max(20, int(min(width, height) * 0.045)), weight=500)
    draw.text(
        (width // 2, height - margin),
        day.strftime("%A · %b %-d").upper(),
        font=label_font,
        fill=INK,
        anchor="mm",
    )
    return card


def _shrink_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_size: int,
    min_size: int,
    font_fn: Any = None,
    weight: int = 600,
) -> tuple[ImageFont.FreeTypeFont, str]:
    """Single-line variant of `_fit_text_to_width`: shrinks font size to fit
    `text` on one line, truncating with an ellipsis if even `min_size`
    overflows rather than wrapping."""
    load = font_fn or _inter
    for size in range(max_size, min_size - 1, -2):
        font = load(size, weight=weight)
        if draw.textlength(text, font=font) <= max_width:
            return font, text

    font = load(min_size, weight=weight)
    truncated = text
    while truncated and draw.textlength(truncated + "…", font=font) > max_width:
        truncated = truncated[:-1]
    return font, (truncated + "…") if truncated != text else text


def _draw_date_header(
    canvas: Image.Image,
    *,
    day: date,
    mood: str,
    width: int,
    height: int,
    title: str | None = None,
) -> Box:
    draw = ImageDraw.Draw(canvas)
    header_box: Box = (0, 0, width, height)
    draw.rectangle(header_box, fill=FIELD)
    draw.line([(0, height - 1), (width, height - 1)], fill=RULE, width=2)

    margin = CAPTION_PADDING * 2
    max_date_size = max(24, min(56, height * 3 // 8))
    pill_h = min(56, height * 2 // 3)
    pill_pad = 20

    pill_box: Box | None = None
    pill_font: ImageFont.FreeTypeFont | None = None
    pill_text = ""
    if mood:
        max_pill_w = int(width * 0.45)
        pill_font, pill_text = _shrink_to_width(
            draw,
            f"mood: {mood}",
            max_width=max_pill_w - 2 * pill_pad,
            max_size=max(16, max_date_size // 2),
            min_size=14,
            font_fn=_mono,
            weight=400,
        )
        pill_w = int(draw.textlength(pill_text, font=pill_font)) + 2 * pill_pad
        pill_x2 = width - margin
        pill_box = (pill_x2 - pill_w, (height - pill_h) // 2, pill_x2, (height + pill_h) // 2)

    available_date_w = (pill_box[0] - margin - 16) if pill_box else (width - 2 * margin)
    date_font, date_text = _shrink_to_width(
        draw,
        title or day.strftime("%A, %B %-d, %Y"),
        max_width=available_date_w,
        max_size=max_date_size,
        min_size=18,
    )
    draw.text((margin, height // 2), date_text, font=date_font, fill=INK, anchor="lm")

    if pill_box is not None and pill_font is not None:
        draw.rounded_rectangle(pill_box, radius=pill_h // 2, fill=MUSTARD)
        draw.text(
            ((pill_box[0] + pill_box[2]) // 2, (pill_box[1] + pill_box[3]) // 2),
            pill_text,
            font=pill_font,
            fill=INK_TEXT_ON_MUSTARD,
            anchor="mm",
        )
    return header_box


def compose_strip(
    images_by_panel: dict[int, bytes],
    script: Script,
    *,
    day: date,
    face_boxes: dict[int, Box | None] | None = None,
    header_title: str | None = None,
) -> ComposedStrip:
    """`header_title` replaces the date in the header, e.g. the weekly recap's
    "The week of September 14 to 20, 2026" instead of a single day's date."""
    face_boxes = face_boxes or {}
    panels = sorted(script.panels, key=lambda p: p.id)
    for panel in panels:
        if panel.id not in images_by_panel:
            raise MissingPanelImageError(panel.id)

    boxes, card_box, canvas_h = _strip_geometry(len(panels))
    cell_size = boxes[0][2] - boxes[0][0]

    strip = Image.new("RGB", (STRIP_SIZE, canvas_h), FIELD)
    header_box = _draw_date_header(
        strip,
        day=day,
        mood=script.mood,
        width=STRIP_SIZE,
        height=DATE_HEADER_HEIGHT,
        title=header_title,
    )

    panel_layouts: list[dict[str, Any]] = []
    panel_pngs: dict[int, bytes] = {}
    for panel, panel_box in zip(panels, boxes, strict=True):
        x, y = panel_box[0], panel_box[1]
        cell, cell_layout = render_panel_cell(
            images_by_panel[panel.id], panel, face_boxes.get(panel.id), size=cell_size
        )
        strip.paste(cell, (x, y))
        panel_layouts.append(
            {
                "id": panel.id,
                "box": list(panel_box),
                "caption_box": _offset_box(cell_layout["caption_box"], x, y),
                "bubble_box": _offset_box(cell_layout["bubble_box"], x, y),
                "face_box": _offset_box(cell_layout["face_box"], x, y),
            }
        )
        panel_pngs[panel.id] = render_panel_png(
            images_by_panel[panel.id], panel, face_boxes.get(panel.id)
        )

    if card_box is not None:
        card = _render_day_card(
            card_box[2] - card_box[0], card_box[3] - card_box[1], mood=script.mood, day=day
        )
        strip.paste(card, (card_box[0], card_box[1]))

    strip_buf = BytesIO()
    strip.save(strip_buf, format="PNG")

    story = _render_story(images_by_panel, panels, face_boxes, day=day, mood=script.mood)
    story_buf = BytesIO()
    story.save(story_buf, format="PNG")

    layout = {
        "strip": {
            "width": STRIP_SIZE,
            "height": canvas_h,
            "date_header_box": list(header_box),
            "panels": panel_layouts,
            "card_box": list(card_box) if card_box is not None else None,
        },
        "story": {"width": STORY_SIZE[0], "height": STORY_SIZE[1]},
    }

    return ComposedStrip(
        strip_png=strip_buf.getvalue(),
        story_png=story_buf.getvalue(),
        panel_pngs=panel_pngs,
        layout=layout,
    )


def _offset_box(box: list[int] | None, dx: int, dy: int) -> list[int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def _render_story(
    images_by_panel: dict[int, bytes],
    panels: list[Panel],
    face_boxes: dict[int, Box | None],
    *,
    day: date,
    mood: str,
) -> Image.Image:
    width, height = STORY_SIZE
    story = Image.new("RGB", (width, height), FIELD)
    header_h = 140
    _draw_date_header(story, day=day, mood=mood, width=width, height=header_h)
    remaining_h = height - header_h
    cell_h = remaining_h // len(panels)

    for i, panel in enumerate(panels):
        source = Image.open(BytesIO(images_by_panel[panel.id]))
        cell = _contain_on_background(source, (width, cell_h), FIELD)
        _draw_caption_bar(cell, panel.caption_a)
        if panel.bubble:
            face_box = face_boxes.get(panel.id)
            scaled_face_box = (
                _scale_face_box_contain(face_box, from_size=source.size, to_size=(width, cell_h))
                if face_box is not None
                else None
            )
            bubble_box = _bubble_box(scaled_face_box, (width, cell_h))
            _draw_bubble(cell, bubble_box, panel.bubble)
        story.paste(cell, (0, header_h + i * cell_h))

    return story


def chosen_candidates_by_panel(state: DayState) -> dict[int, Candidate]:
    chosen: dict[int, Candidate] = {}
    for candidate in state.candidates:
        if candidate.chosen:
            chosen[candidate.panel_id] = candidate
    return chosen


def compose_day(state: DayState, *, storage: Storage) -> DayState:
    """Loads each panel's chosen candidate image back from `storage`,
    composes the strip/story/per-panel PNGs, writes them all back to
    `storage`, and returns `state` with `strip_url`/`layout` populated."""
    if state.script is None:
        raise ComposeError("compose_day: DayState.script is None")

    chosen = chosen_candidates_by_panel(state)
    images_by_panel = {
        panel_id: storage.get_object_by_url(candidate.url) for panel_id, candidate in chosen.items()
    }
    face_boxes = {panel_id: candidate.face_box for panel_id, candidate in chosen.items()}

    composed = compose_strip(images_by_panel, state.script, day=state.date, face_boxes=face_boxes)

    prefix = f"strips/{state.user_id}/{state.date.isoformat()}"
    strip_url = storage.put_object(
        f"{prefix}/strip.png", composed.strip_png, content_type="image/png"
    )
    storage.put_object(f"{prefix}/story.png", composed.story_png, content_type="image/png")
    for panel_id, png in composed.panel_pngs.items():
        storage.put_object(f"{prefix}/panel-{panel_id}.png", png, content_type="image/png")
    storage.put_object(
        f"{prefix}/layout.json",
        json.dumps(composed.layout, indent=2).encode(),
        content_type="application/json",
    )

    return state.model_copy(update={"strip_url": strip_url, "layout": composed.layout})
