from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from .assembly import compose_page
from .backends import build_backend
from .bubbles import render_bubbles
from .config import PipelineConfig
from .layouts import boxes_for, panel_count
from .registry import CharacterRegistry
from .schema import Story


def run(story: Story, cfg: PipelineConfig, on_progress: "Callable[[str], None] | None" = None) -> Path:
    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    backend = build_backend(cfg)
    size = (cfg.page_width, cfg.page_height)
    out_dir = Path(cfg.output_dir) / story.id
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}.json")

    report("designing characters")
    backend.prepare_characters(story.id, registry, story.style_prompt)

    page_images: list[Image.Image] = []
    for page_index, page in enumerate(story.pages):
        report(f"generating page {page_index + 1}/{len(story.pages)}")
        expected = panel_count(page.layout)
        if len(page.panels) != expected:
            raise ValueError(
                f"page {page_index}: layout '{page.layout}' expects {expected} panels, got {len(page.panels)}"
            )

        base = backend.generate_base(story.id, page_index, page, size, story.style_prompt)
        boxes = boxes_for(page.layout)
        panel_images = [
            backend.edit_panel(
                base,
                story.id,
                page_index,
                panel_index,
                panel,
                (max(1, round(w * size[0])), max(1, round(h * size[1]))),
                story.style_prompt,
                registry,
            )
            for panel_index, (panel, (_, _, w, h)) in enumerate(zip(page.panels, boxes))
        ]
        composed = compose_page(page.layout, panel_images, size)
        final = render_bubbles(composed, page, size)

        page_path = out_dir / f"page_{page_index:02d}.png"
        final.save(page_path)
        page_images.append(final)

    report("assembling PDF")
    pdf_path = out_dir / f"{story.id}.pdf"
    if page_images:
        page_images[0].save(pdf_path, save_all=True, append_images=page_images[1:])

    report("done")
    return pdf_path
