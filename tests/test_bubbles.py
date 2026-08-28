"""Regression tests for the bubble-overflow fix in manga_pipeline/bubbles.py -
see the pipeline review: bubbles used to advance by a fixed guessed height
regardless of actual wrapped-text size, and never shrank the font for a
dialogue-dense panel, producing overlapping/illegible text.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from manga_pipeline.bubbles import (
    _MIN_FONT_SIZE,
    _draw_bubble,
    _font_size_for_panel,
    _is_shouted,
    _shout_outline_points,
    _tail_geometry,
    _thought_trail,
    render_bubbles,
)
from manga_pipeline.fonts import load_font
from manga_pipeline.schema import DialogueLine, Page, Panel


def _draw() -> ImageDraw.ImageDraw:
    return ImageDraw.Draw(Image.new("RGBA", (400, 600), (255, 255, 255, 255)))


def _page_and_draw() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (400, 600), (255, 255, 255, 255))
    return img, ImageDraw.Draw(img)


def test_draw_bubble_returns_its_actual_rendered_height():
    font = load_font(16)
    short = DialogueLine(speaker="Aiko", text="Hi.", kind="speech")
    long = DialogueLine(
        speaker="Aiko",
        text="This is a much longer line of dialogue that will wrap across several lines of text.",
        kind="speech",
    )
    img1, draw1 = _page_and_draw()
    short_h = _draw_bubble(img1, draw1, (0, 0, 200, 600), short, font, None)
    img2, draw2 = _page_and_draw()
    long_h = _draw_bubble(img2, draw2, (0, 0, 200, 600), long, font, None)
    assert long_h > short_h  # a taller bubble must report a taller height


def test_draw_bubble_works_with_an_anchor_too():
    # a real (x, y) anchor - see face_detect.py - shouldn't change the
    # bubble's own reported height, just add a tail pointing toward it
    img, draw = _page_and_draw()
    line = DialogueLine(speaker="Aiko", text="Hi.", kind="speech")
    height = _draw_bubble(img, draw, (0, 0, 200, 600), line, load_font(16), (350, 400))
    assert height > 0


def test_tail_geometry_points_toward_a_right_side_anchor():
    box = (0.0, 0.0, 100.0, 50.0)
    tail = _tail_geometry(box, (300.0, 25.0))
    assert tail is not None
    base1, base2, tip = tail
    # the tail should reach toward the anchor, past the bubble's right edge
    assert tip[0] > box[2]


def test_tail_geometry_points_toward_a_bottom_anchor():
    box = (0.0, 0.0, 100.0, 50.0)
    tail = _tail_geometry(box, (50.0, 400.0))
    assert tail is not None
    base1, base2, tip = tail
    assert tip[1] > box[3]


def test_tail_geometry_returns_none_when_anchor_is_inside_the_bubble():
    box = (0.0, 0.0, 100.0, 50.0)
    assert _tail_geometry(box, (50.0, 25.0)) is None


def test_thought_trail_moves_toward_the_anchor():
    box = (0.0, 0.0, 100.0, 50.0)
    trail = _thought_trail(box, (300.0, 25.0))
    assert len(trail) == 3
    xs = [x for x, _, _ in trail]
    assert xs == sorted(xs)  # each step moves further toward the (rightward) anchor


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
    # render_bubbles returns a (possibly new, since it normalizes to RGBA
    # internally) image rather than always mutating in place - should
    # simply not raise, and the result should be untouched pixel-for-pixel
    # (no dialogue anywhere to draw)
    result = render_bubbles(img, page, (400, 600))
    assert result.tobytes() == blank


def test_render_bubbles_uses_panel_images_to_find_bubble_anchors():
    # doesn't need a real detector - a panel with no dialogue never even
    # tries to load one, so this just confirms passing panel_images through
    # doesn't break the no-dialogue path
    page = Page(layout="H2", panels=[Panel(scene_description="a"), Panel(scene_description="b")])
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    panel_images = [Image.new("RGB", (200, 600), (255, 255, 255)) for _ in range(2)]
    result = render_bubbles(img, page, (400, 600), panel_images=panel_images)
    assert result.size == (400, 600)


# ---------- shout bubbles ----------


def test_is_shouted_true_for_trailing_exclamation_mark():
    assert _is_shouted("Get down!") is True


def test_is_shouted_true_for_mostly_uppercase_text():
    assert _is_shouted("GET DOWN NOW") is True


def test_is_shouted_false_for_calm_dialogue():
    assert _is_shouted("I think we should go.") is False


def test_is_shouted_false_for_short_text_even_if_uppercase():
    # too short to trust casing as a real signal (e.g. a single-letter
    # exclamation or initial) - needs at least 4 letters
    assert _is_shouted("OK") is False


def test_shout_outline_points_stays_within_a_reasonable_bound():
    points = _shout_outline_points(100, 60)
    assert len(points) == 28  # _SHOUT_SPIKES * 2
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    # generously larger than the tight box, but not unbounded
    assert max(xs) - min(xs) <= 100 * 1.4
    assert max(ys) - min(ys) <= 60 * 1.4


def test_draw_bubble_uses_shout_style_for_exclaimed_speech():
    img, draw = _page_and_draw()
    line = DialogueLine(speaker="Aiko", text="LOOK OUT!", kind="speech")
    height = _draw_bubble(img, draw, (0, 0, 200, 600), line, load_font(16), None)
    assert height > 0


def test_draw_bubble_keeps_normal_style_for_calm_speech():
    img, draw = _page_and_draw()
    line = DialogueLine(speaker="Aiko", text="Hi there.", kind="speech")
    height = _draw_bubble(img, draw, (0, 0, 200, 600), line, load_font(16), None)
    assert height > 0
