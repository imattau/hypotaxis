from __future__ import annotations

from PIL import Image, ImageDraw

from .fonts import load_font, wrap_to_width
from .layouts import boxes_for
from .schema import DialogueLine, Page


_GAP = 4
_MIN_FONT_SIZE = 8


def _bubble_text(line: DialogueLine) -> str:
    return f"{line.speaker}: {line.text}" if line.kind == "narration" else line.text


def _wrapped_height(draw: ImageDraw.ImageDraw, wrapped: str, font, pad: int) -> int:
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    return (bbox[3] - bbox[1]) + 2 * pad


def _draw_bubble(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], line: DialogueLine, font) -> int:
    """Draws the bubble and returns its actual rendered height, so the
    caller can advance its cursor by the real amount instead of a fixed
    guess - a fixed guess is what let long-wrapped bubbles overlap the next
    one (or run past the panel entirely) on dialogue-dense panels."""
    x0, y0, x1, y1 = box
    pad = 8
    text = _bubble_text(line)
    wrapped = wrap_to_width(draw, text, font, (x1 - x0) - 2 * pad)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    bubble_w = min(x1 - x0, text_w + 2 * pad)
    bubble_h = text_h + 2 * pad
    bx0, by0 = x0 + 4, y0 + 4
    bx1, by1 = bx0 + bubble_w, by0 + bubble_h

    if line.kind == "narration":
        draw.rectangle((bx0, by0, bx1, by1), fill=(255, 255, 200), outline=(0, 0, 0), width=2)
    elif line.kind == "thought":
        draw.ellipse((bx0, by0, bx1, by1), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    else:
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=10, fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    draw.multiline_text((bx0 + pad, by0 + pad), wrapped, fill=(0, 0, 0), font=font)
    return bubble_h + 4


def _font_size_for_panel(draw: ImageDraw.ImageDraw, dialogue: list[DialogueLine], panel_w: int, panel_h: int, base_size: int) -> int:
    """Shrinks the font (down to a readable floor) until this panel's whole
    dialogue stack actually fits in its height - rather than always using
    the page's base font size and letting overflow silently stack past the
    panel edge, which is what previously produced overlapping bubbles on
    any panel with more than 2-3 lines."""
    for size in range(base_size, _MIN_FONT_SIZE - 1, -1):
        font = load_font(size)
        total = sum(_wrapped_height(draw, wrap_to_width(draw, _bubble_text(line), font, panel_w - 16), font, 8) + _GAP for line in dialogue)
        if total <= panel_h or size == _MIN_FONT_SIZE:
            return size
    return _MIN_FONT_SIZE


def render_bubbles(page_img: Image.Image, page: Page, page_size: tuple[int, int]) -> Image.Image:
    draw = ImageDraw.Draw(page_img)
    base_size = max(12, page_size[1] // 45)
    boxes = boxes_for(page.layout)

    for (x, y, w, h), panel in zip(boxes, page.panels):
        box_px = (int(x * page_size[0]), int(y * page_size[1]), int((x + w) * page_size[0]), int((y + h) * page_size[1]))
        panel_w, panel_h = box_px[2] - box_px[0], box_px[3] - box_px[1]
        if not panel.dialogue:
            continue

        font = load_font(_font_size_for_panel(draw, panel.dialogue, panel_w, panel_h, base_size))

        cursor_y = box_px[1]
        for line in panel.dialogue:
            if box_px[3] - cursor_y < _MIN_FONT_SIZE + 8:
                break  # no usable room left in this panel even at the minimum font size
            local_box = (box_px[0], cursor_y, box_px[2], box_px[3])
            cursor_y += _draw_bubble(draw, local_box, line, font)

    return page_img
