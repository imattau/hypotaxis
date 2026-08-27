from __future__ import annotations

from PIL import Image, ImageOps

from .layouts import boxes_for


def compose_page(layout: str, panel_images: list[Image.Image], page_size: tuple[int, int]) -> Image.Image:
    page = Image.new("RGB", page_size, (255, 255, 255))
    boxes = boxes_for(layout)
    gutter = max(2, page_size[0] // 200)

    for (x, y, w, h), panel_img in zip(boxes, panel_images):
        box_px = (
            int(x * page_size[0]) + gutter,
            int(y * page_size[1]) + gutter,
            int((x + w) * page_size[0]) - gutter,
            int((y + h) * page_size[1]) - gutter,
        )
        box_w = max(1, box_px[2] - box_px[0])
        box_h = max(1, box_px[3] - box_px[1])
        fitted = ImageOps.fit(panel_img, (box_w, box_h), method=Image.LANCZOS)
        page.paste(fitted, (box_px[0], box_px[1]))

    return page
