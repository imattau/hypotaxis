from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw

from .fonts import load_font, wrap_to_width
from .layouts import boxes_for
from .schema import DialogueLine, Page

_GAP = 4
_MIN_FONT_SIZE = 8
_TAIL_LENGTH = 36
_TAIL_BASE_HALF_WIDTH = 10
_THOUGHT_TRAIL_STEPS = 3
_SHOUT_SPIKES = 14
_SHOUT_JITTER = 0.22
_BUBBLE_WIDTH_FRACTION = 0.62
_MIN_BUBBLE_WIDTH = 140


def _bubble_text(line: DialogueLine) -> str:
    return f"{line.speaker}: {line.text}" if line.kind == "narration" else line.text


def _is_shouted(text: str) -> bool:
    """Cheap heuristic for whether a speech line reads as shouted/exclaimed
    - nothing upstream currently tags dialogue with an intensity signal
    (Stage A's dialogue extraction doesn't distinguish shouted from calm
    speech), so this looks at the line's own punctuation/casing instead: a
    trailing "!" (the standard prose convention for shouted/exclaimed
    dialogue), or the line being substantially uppercase (some prose writes
    shouted dialogue in caps instead of, or alongside, the exclamation
    point). Only meaningful for speech lines - thought/narration don't get
    a shout-styled bubble regardless, see _draw_bubble.
    """
    stripped = text.strip()
    if stripped.endswith("!"):
        return True
    letters = [c for c in stripped if c.isalpha()]
    return len(letters) >= 4 and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8


def _wrapped_height(draw: ImageDraw.ImageDraw, wrapped: str, font, pad: int) -> int:
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    return (bbox[3] - bbox[1]) + 2 * pad


def _bubble_max_width(panel_w: float, kind: str) -> float:
    """Speech/thought bubbles wrap to a natural reading width rather than
    the full panel - a real page showed the bug this fixes: wrap_to_width
    used to be given almost the entire panel width as its limit, so a line
    only wrapped once it was nearly panel-wide, and most dialogue ended up
    as one long single-line bubble stretching across most of the panel
    instead of a compact, multi-line one. Narration keeps the full-width
    behavior deliberately - a caption box conventionally spans the panel,
    unlike a speech/thought bubble attached to one character."""
    if kind == "narration":
        return panel_w
    return max(_MIN_BUBBLE_WIDTH, min(panel_w, panel_w * _BUBBLE_WIDTH_FRACTION))


def _bubble_x0(x0: float, x1: float, bubble_w: float, anchor: tuple[float, float] | None) -> float:
    """Where a bubble's left edge lands, horizontally: hovering near its
    speaker (anchor) when one's known, clamped to stay inside the panel -
    or flush against the panel's left edge (with a small 4px inset) when
    nothing was detected, the original fallback behavior."""
    if anchor is None:
        return x0 + 4
    ax, _ = anchor
    return max(x0 + 4, min(x1 - bubble_w - 4, ax - bubble_w / 2))


def _tail_geometry(
    box: tuple[float, float, float, float], anchor: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """Where a speech/thought bubble's tail attaches to its own body and
    points toward `anchor` (the speaking character's detected face - see
    face_detect.py). Picks whichever edge (left/right/top/bottom) the
    anchor is most in the direction of, rather than exact rounded-rect/
    vector-intersection math - simpler, and visually indistinguishable for
    a small tail. Returns None if the anchor is inside the bubble itself
    (nothing sensible to point at)."""
    bx0, by0, bx1, by1 = box
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ax, ay = anchor
    if bx0 <= ax <= bx1 and by0 <= ay <= by1:
        return None
    dx, dy = ax - cx, ay - cy

    if abs(dx) >= abs(dy):
        edge_x = bx0 if dx < 0 else bx1
        t = max(0.15, min(0.85, (ay - by0) / max(1.0, by1 - by0)))
        attach_y = by0 + t * (by1 - by0)
        base1 = (edge_x, attach_y - _TAIL_BASE_HALF_WIDTH)
        base2 = (edge_x, attach_y + _TAIL_BASE_HALF_WIDTH)
        tip_x = edge_x + (_TAIL_LENGTH if dx > 0 else -_TAIL_LENGTH)
        tip = (tip_x, attach_y + max(-1.0, min(1.0, dy / max(1.0, abs(dx)))) * (_TAIL_LENGTH * 0.3))
    else:
        edge_y = by0 if dy < 0 else by1
        t = max(0.15, min(0.85, (ax - bx0) / max(1.0, bx1 - bx0)))
        attach_x = bx0 + t * (bx1 - bx0)
        base1 = (attach_x - _TAIL_BASE_HALF_WIDTH, edge_y)
        base2 = (attach_x + _TAIL_BASE_HALF_WIDTH, edge_y)
        tip_y = edge_y + (_TAIL_LENGTH if dy > 0 else -_TAIL_LENGTH)
        tip = (attach_x + max(-1.0, min(1.0, dx / max(1.0, abs(dy)))) * (_TAIL_LENGTH * 0.3), tip_y)
    return base1, base2, tip


def _thought_trail(box: tuple[float, float, float, float], anchor: tuple[float, float]) -> list[tuple[float, float, float]]:
    """A shrinking chain of circles from the thought bubble toward
    `anchor`, the classic manga "thought trail" - (x, y, radius) triples."""
    bx0, by0, bx1, by1 = box
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ax, ay = anchor
    dx, dy = ax - cx, ay - cy
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    # start just outside the bubble's edge in the anchor's direction
    edge_dist = min(bx1 - bx0, by1 - by0) / 2
    start_x, start_y = cx + ux * edge_dist, cy + uy * edge_dist
    trail = []
    for step in range(1, _THOUGHT_TRAIL_STEPS + 1):
        radius = max(2.0, 8.0 - step * 2.0)
        px = start_x + ux * step * (_TAIL_LENGTH / _THOUGHT_TRAIL_STEPS)
        py = start_y + uy * step * (_TAIL_LENGTH / _THOUGHT_TRAIL_STEPS)
        trail.append((px, py, radius))
    return trail


def _render_svg(width: int, height: int, body: str) -> Image.Image:
    import cairosvg

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{body}</svg>'
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def _speech_bubble_image(w: int, h: int, tail, margin: int) -> tuple[Image.Image, int, int]:
    """Renders a rounded-rect speech bubble (optionally with a tail
    triangle) to its own RGBA image, padded by `margin` on every side so a
    tail extending past the bubble box isn't clipped. Returns (image,
    origin_x, origin_y) - the offset to paste this image at, in the same
    coordinate space `box`/`tail` were given in."""
    canvas_w, canvas_h = w + 2 * margin, h + 2 * margin
    body = ""
    if tail is not None:
        # tail drawn first, rounded rect on top - the rect's own fill+stroke
        # naturally covers the part of the tail that overlaps it, leaving
        # just the exposed outline outside the rect, no separate seam
        # patch needed
        base1, base2, tip = tail
        pts = " ".join(f"{x + margin},{y + margin}" for x, y in (base1, tip, base2))
        body += f'<polygon points="{pts}" fill="white" stroke="black" stroke-width="3"/>'
    body += f'<rect x="{margin}" y="{margin}" width="{w}" height="{h}" rx="14" fill="white" stroke="black" stroke-width="3"/>'
    return _render_svg(canvas_w, canvas_h, body), -margin, -margin


def _shout_outline_points(w: float, h: float) -> list[tuple[float, float]]:
    """Deterministic jagged burst outline (alternating outer/inner radius
    around an ellipse) for a shouted/exclaimed speech line - see
    _is_shouted. Sized generously larger than the tight text box (1.3x,
    not just inscribed at w/2, h/2) so the inward-dipping valleys don't
    cut into the text even at the shape's tightest points."""
    cx, cy = w / 2, h / 2
    rx, ry = (w / 2) * 1.3, (h / 2) * 1.3
    points = []
    for i in range(_SHOUT_SPIKES * 2):
        angle = (i / (_SHOUT_SPIKES * 2)) * 2 * math.pi
        scale = 1.0 if i % 2 == 0 else (1.0 - _SHOUT_JITTER)
        points.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    return points


def _shout_bubble_image(w: int, h: int, tail, margin: int) -> tuple[Image.Image, int, int]:
    """Jagged burst-shaped speech bubble for shouted/exclaimed dialogue
    (see _is_shouted) - same tail mechanism as the normal speech bubble,
    just a spikier body instead of a smooth rounded rect. The burst
    deliberately extends past the tight text box (see
    _shout_outline_points), so the canvas margin here is widened to match
    whatever that actually needs, not just the fixed tail-only margin every
    other bubble style uses - a caller-supplied margin too small for the
    burst's own spikes would silently clip them.
    """
    burst_margin = max(margin, int(0.3 * max(w, h) / 2) + 8)
    canvas_w, canvas_h = w + 2 * burst_margin, h + 2 * burst_margin
    body = ""
    if tail is not None:
        base1, base2, tip = tail
        pts = " ".join(f"{x + burst_margin},{y + burst_margin}" for x, y in (base1, tip, base2))
        body += f'<polygon points="{pts}" fill="white" stroke="black" stroke-width="3"/>'
    outline = _shout_outline_points(w, h)
    pts_str = " ".join(f"{x + burst_margin},{y + burst_margin}" for x, y in outline)
    body += f'<polygon points="{pts_str}" fill="white" stroke="black" stroke-width="3"/>'
    return _render_svg(canvas_w, canvas_h, body), -burst_margin, -burst_margin


def _thought_bubble_image(w: int, h: int, trail, margin: int) -> tuple[Image.Image, int, int]:
    canvas_w, canvas_h = w + 2 * margin, h + 2 * margin
    body = f'<ellipse cx="{margin + w / 2}" cy="{margin + h / 2}" rx="{w / 2}" ry="{h / 2}" fill="white" stroke="black" stroke-width="3"/>'
    for x, y, r in trail or []:
        body += f'<circle cx="{x + margin}" cy="{y + margin}" r="{r}" fill="white" stroke="black" stroke-width="2"/>'
    return _render_svg(canvas_w, canvas_h, body), -margin, -margin


def _narration_box_image(w: int, h: int, margin: int) -> tuple[Image.Image, int, int]:
    canvas_w, canvas_h = w + 2 * margin, h + 2 * margin
    body = f'<rect x="{margin}" y="{margin}" width="{w}" height="{h}" fill="#fffcc8" stroke="black" stroke-width="3"/>'
    return _render_svg(canvas_w, canvas_h, body), -margin, -margin


def _draw_bubble(
    page_img: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], line: DialogueLine, font, anchor
) -> int:
    """Draws the bubble (SVG-rendered shape, composited onto page_img, plus
    PIL-drawn text on top - text stays PIL so wrap_to_width's sizing, which
    this function's own layout math depends on, stays exactly what actually
    gets rendered) and returns its rendered height, so the caller can
    advance its cursor by the real amount instead of a fixed guess.

    anchor: an (x, y) page-pixel point (the speaking character's detected
    face - see face_detect.py) to point the bubble's tail toward, or None
    if nothing was detected for this panel - a bubble with no anchor is
    drawn with no tail at all rather than guessing a direction, since a
    guessed tail pointing at nothing would be worse than no tail.
    """
    x0, y0, x1, y1 = box
    panel_w = x1 - x0
    is_shout = line.kind == "speech" and _is_shouted(line.text)
    # the shout burst's inward-dipping valleys (see _shout_outline_points)
    # need more clearance from the text than a smooth rounded rect does
    pad = 14 if is_shout else 8
    text = _bubble_text(line)

    max_bubble_w = _bubble_max_width(panel_w, line.kind)
    wrapped = wrap_to_width(draw, text, font, max_bubble_w - 2 * pad)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    bubble_w = min(max_bubble_w, text_w + 2 * pad)
    bubble_h = text_h + 2 * pad
    bx0 = _bubble_x0(x0, x1, bubble_w, anchor)
    by0 = y0 + 4
    bx1, by1 = bx0 + bubble_w, by0 + bubble_h

    margin = _TAIL_LENGTH + _TAIL_BASE_HALF_WIDTH + 4
    if line.kind == "narration":
        bubble_img, ox, oy = _narration_box_image(bubble_w, bubble_h, margin)
    elif line.kind == "thought":
        trail = _thought_trail((bx0, by0, bx1, by1), anchor) if anchor is not None else None
        # trail() computed in page-absolute coords, but the bubble image is
        # rendered in its own local (0,0)-origin space - shift back
        local_trail = [(x - bx0, y - by0, r) for x, y, r in trail] if trail else None
        bubble_img, ox, oy = _thought_bubble_image(bubble_w, bubble_h, local_trail, margin)
    elif is_shout:
        tail = _tail_geometry((bx0, by0, bx1, by1), anchor) if anchor is not None else None
        local_tail = [(x - bx0, y - by0) for x, y in tail] if tail else None
        bubble_img, ox, oy = _shout_bubble_image(bubble_w, bubble_h, local_tail, margin)
    else:
        tail = _tail_geometry((bx0, by0, bx1, by1), anchor) if anchor is not None else None
        local_tail = [(x - bx0, y - by0) for x, y in tail] if tail else None
        bubble_img, ox, oy = _speech_bubble_image(bubble_w, bubble_h, local_tail, margin)

    # render_bubbles() already ensures page_img is RGBA before any bubble is drawn
    page_img.alpha_composite(bubble_img, (int(bx0 + ox), int(by0 + oy)))
    draw.multiline_text((bx0 + pad, by0 + pad), wrapped, fill=(0, 0, 0), font=font)
    return bubble_h + 4


_MAX_MERGED_CHARS = 160


def _merge_consecutive(dialogue: list[DialogueLine]) -> list[DialogueLine]:
    """Collapses a run of consecutive same-speaker, same-kind lines into one
    bubble. Stage A's dialogue extraction (split_dialogue in story_adapt.py)
    emits one DialogueLine per quoted/italicized span in the source prose,
    so an uninterrupted turn split across several sentences/quotes in the
    manuscript - "She paused." "I wasn't ready for. After that." "And now?",
    all the same speaker with no other speaker between them - previously
    rendered as a tower of separate bubbles instead of the single turn it
    actually is.

    Capped at _MAX_MERGED_CHARS: Stage A's speaker attribution is a
    heuristic (see split_dialogue's docstring) and sometimes wrong for many
    lines in a row - unlike an uninterrupted real turn, that failure mode
    produces long runs of "same speaker" that are actually several
    back-and-forth exchanges mis-tagged as one speaker. Merging those
    wholesale turned into a single bubble spanning the entire dialogue
    stack, overflowing the panel - worse than the many-small-bubbles problem
    this function exists to fix. The cap keeps a real short turn intact
    while forcing a long mis-attributed run back into separate bubbles."""
    merged: list[DialogueLine] = []
    for line in dialogue:
        prev = merged[-1] if merged else None
        combined_len = len(prev.text) + 1 + len(line.text) if prev is not None else 0
        if prev is not None and prev.speaker == line.speaker and prev.kind == line.kind and combined_len <= _MAX_MERGED_CHARS:
            merged[-1] = DialogueLine(speaker=prev.speaker, text=f"{prev.text} {line.text}", kind=prev.kind)
        else:
            merged.append(line)
    return merged


def _font_size_for_panel(draw: ImageDraw.ImageDraw, dialogue: list[DialogueLine], panel_w: int, panel_h: int, base_size: int) -> int:
    """Shrinks the font (down to a readable floor) until this panel's whole
    dialogue stack actually fits in its height - rather than always using
    the page's base font size and letting overflow silently stack past the
    panel edge, which is what previously produced overlapping bubbles on
    any panel with more than 2-3 lines."""
    for size in range(base_size, _MIN_FONT_SIZE - 1, -1):
        font = load_font(size)
        total = sum(
            _wrapped_height(draw, wrap_to_width(draw, _bubble_text(line), font, _bubble_max_width(panel_w, line.kind) - 16), font, 8) + _GAP
            for line in dialogue
        )
        if total <= panel_h or size == _MIN_FONT_SIZE:
            return size
    return _MIN_FONT_SIZE


def render_bubbles(
    page_img: Image.Image,
    page: Page,
    page_size: tuple[int, int],
    panel_images: list[Image.Image] | None = None,
    caption_reserve_fraction: float = 0.0,
    on_warning=None,
) -> Image.Image:
    """panel_images: the same list generate_panel() produced for this page
    (already resized to each panel's own box - see pipeline.run()), used to
    find a face anchor per panel to point each panel's speech/thought
    bubble tails at (see face_detect.FaceAnchorDetector). Optional and
    best-effort: omit it (or a panel where detection finds nothing) and
    bubbles render exactly as before, tail-less, at the same position.
    """
    if not 0.0 <= caption_reserve_fraction < 1.0:
        raise ValueError("caption_reserve_fraction must be between 0 and 1")
    if page_img.mode != "RGBA":
        page_img = page_img.convert("RGBA")
    draw = ImageDraw.Draw(page_img)
    # //45 (the previous divisor) picks a font sized for the page as a
    # whole regardless of how many panels split it up, so a page with 2-3
    # columns of panels still started the shrink-to-fit search in
    # _font_size_for_panel at full letterer-for-one-big-panel size - a
    # narrow panel would only come down to something proportionate after
    # several lines' worth of bubbles forced the shrink loop to act. //60
    # keeps the same fit-driven shrink behavior but starts it from a size
    # that already reads as normal comic lettering on a typical panel.
    base_size = max(12, page_size[1] // 60)
    boxes = boxes_for(page.layout)

    detector = None
    if panel_images is not None:
        from .face_detect import FaceAnchorDetector

        detector = FaceAnchorDetector()

    for panel_index, ((x, y, w, h), panel) in enumerate(zip(boxes, page.panels)):
        box_px = (int(x * page_size[0]), int(y * page_size[1]), int((x + w) * page_size[0]), int((y + h) * page_size[1]))
        panel_w, panel_h = box_px[2] - box_px[0], box_px[3] - box_px[1]
        if not panel.dialogue:
            continue
        dialogue_bottom = box_px[3] - round(panel_h * caption_reserve_fraction)
        dialogue_h = max(1, dialogue_bottom - box_px[1])

        anchors: list[tuple[float, float]] = []
        if detector is not None and panel_index < len(panel_images):
            local_anchors = detector.find_anchors(panel_images[panel_index])
            anchors = [(box_px[0] + ax, box_px[1] + ay) for ax, ay in local_anchors]

        dialogue = _merge_consecutive(panel.dialogue)
        font_size = _font_size_for_panel(draw, dialogue, panel_w, dialogue_h, base_size)
        font = load_font(font_size)
        if font_size == _MIN_FONT_SIZE:
            if on_warning is not None:
                on_warning(
                    f"panel {panel_index + 1}: dialogue needs minimum lettering size; "
                    "some lines may be omitted before the caption area"
                )

        cursor_y = box_px[1]
        for line_index, line in enumerate(dialogue):
            if dialogue_bottom - cursor_y < _MIN_FONT_SIZE + 8:
                if on_warning is not None:
                    on_warning(f"panel {panel_index + 1}: omitted dialogue after the lettering area filled")
                break
            local_box = (box_px[0], cursor_y, box_px[2], dialogue_bottom)
            # no reliable way to match a dialogue line's speaker to a
            # specific detected face from pixels alone (same open problem
            # as pose-ControlNet's multi-figure identity assignment) - cycle
            # through whatever anchors this panel has, in left-to-right order
            anchor = anchors[line_index % len(anchors)] if anchors else None
            cursor_y += _draw_bubble(page_img, draw, local_box, line, font, anchor)

    # composited bubbles are fully opaque everywhere they cover, and the
    # page underneath started opaque too - safe (and needed for PDF export,
    # which doesn't handle RGBA cleanly) to flatten back to RGB
    return page_img.convert("RGB")
