"""Regression tests for the bubble-overflow fix in manga_pipeline/bubbles.py -
see the pipeline review: bubbles used to advance by a fixed guessed height
regardless of actual wrapped-text size, and never shrank the font for a
dialogue-dense panel, producing overlapping/illegible text.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from manga_pipeline.bubbles import _MIN_FONT_SIZE, _draw_bubble, _font_size_for_panel, render_bubbles
from manga_pipeline.fonts import load_font
from manga_pipeline.schema import DialogueLine, Page, Panel


def _draw() -> ImageDraw.ImageDraw:
    return ImageDraw.Draw(Image.new("RGB", (400, 600), (255, 255, 255)))


def test_draw_bubble_returns_its_actual_rendered_height():
    draw = _draw()
    font = load_font(16)
    short = DialogueLine(speaker="Aiko", text="Hi.", kind="speech")
    long = DialogueLine(
        speaker="Aiko",
        text="This is a much longer line of dialogue that will wrap across several lines of text.",
        kind="speech",
    )
    short_h = _draw_bubble(draw, (0, 0, 200, 600), short, font)
    long_h = _draw_bubble(draw, (0, 0, 200, 600), long, font)
    assert long_h > short_h  # a taller bubble must report a taller height


def test_font_size_for_panel_shrinks_when_dialogue_does_not_fit():
    draw = _draw()
    dialogue = [DialogueLine(speaker="Aiko", text=f"Line number {i} of dialogue text." * 2, kind="speech") for i in range(20)]
    size = _font_size_for_panel(draw, dialogue, panel_w=200, panel_h=300, base_size=24)
    assert size < 24
    assert size >= _MIN_FONT_SIZE


def test_font_size_for_panel_keeps_base_size_when_it_fits():
    draw = _draw()
    dialogue = [DialogueLine(speaker="Aiko", text="Hi.", kind="speech")]
    size = _font_size_for_panel(draw, dialogue, panel_w=400, panel_h=600, base_size=16)
    assert size == 16


def test_render_bubbles_does_not_crash_on_dialogue_dense_panel():
    # this is the exact failure shape found in the real manuscript test: a
    # single panel with far more dialogue lines than could ever visually fit
    page = Page(
        layout="H3",
        panels=[
            Panel(
                scene_description="a",
                dialogue=[DialogueLine(speaker="Nova", text=f"Line {i}", kind="speech") for i in range(20)],
            ),
            Panel(scene_description="b"),
            Panel(scene_description="c"),
        ],
    )
    img = Image.new("RGB", (600, 900), (255, 255, 255))
    result = render_bubbles(img, page, (600, 900))
    assert result.size == (600, 900)


def test_render_bubbles_skips_panels_with_no_dialogue():
    page = Page(layout="H2", panels=[Panel(scene_description="a"), Panel(scene_description="b")])
    blank = Image.new("RGB", (400, 600), (255, 255, 255)).tobytes()
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    # render_bubbles mutates img in place - should simply not raise, and
    # leave it untouched (no dialogue anywhere to draw)
    render_bubbles(img, page, (400, 600))
    assert img.tobytes() == blank
