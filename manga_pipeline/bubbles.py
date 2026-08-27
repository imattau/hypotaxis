from __future__ import annotations

from PIL import Image, ImageDraw

from .fonts import load_font, wrap_to_width
from .layouts import boxes_for
from .schema import DialogueLine, Page


def _draw_bubble(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], line: DialogueLine, font) -> None:
    x0, y0, x1, y1 = box
    pad = 8
    text = f"{line.speaker}: {line.text}" if line.kind == "narration" else line.text
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


def render_bubbles(page_img: Image.Image, page: Page, page_size: tuple[int, int]) -> Image.Image:
    draw = ImageDraw.Draw(page_img)
    font = load_font(max(12, page_size[1] // 45))
    boxes = boxes_for(page.layout)

    for (x, y, w, h), panel in zip(boxes, page.panels):
        box_px = (int(x * page_size[0]), int(y * page_size[1]), int((x + w) * page_size[0]), int((y + h) * page_size[1]))
        cursor_y = box_px[1]
        for line in panel.dialogue:
            local_box = (box_px[0], cursor_y, box_px[2], box_px[3])
            _draw_bubble(draw, local_box, line, font)
            cursor_y += (page_size[1] // 45) * 4

    return page_img
