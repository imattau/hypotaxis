from __future__ import annotations

from dataclasses import asdict

from manga_pipeline.layouts import LAYOUTS, boxes_for, is_wide_box, panel_count
from manga_pipeline.schema import DialogueLine, Page, Panel, Story


def test_every_layout_box_count_matches_panel_count():
    for name in LAYOUTS:
        assert len(boxes_for(name)) == panel_count(name)


def test_every_layout_box_covers_the_full_page_area():
    # boxes are (x, y, w, h) fractions of the page - a layout with gaps or
    # overlaps would silently misplace panels in compose_page
    for name, boxes in LAYOUTS.items():
        total_area = sum(w * h for (_, _, w, h) in boxes)
        assert abs(total_area - 1.0) < 1e-9, name


def test_is_wide_box_true_for_landscape_oriented_box():
    # V2/V3 boxes span the full page width but a fraction of its height -
    # landscape-oriented once the page's own portrait proportions are
    # accounted for
    assert is_wide_box((0.0, 0.0, 1.0, 0.5)) is True


def test_is_wide_box_false_for_portrait_oriented_box():
    # H3 boxes span a third of the page width at full height - a narrow
    # vertical strip, the shape that rendered a "wide two-shot" panel as a
    # single figure instead of two in a real page-generation test
    assert is_wide_box((0.0, 0.0, 1 / 3, 1.0)) is False


def test_is_wide_box_false_for_near_square_grid_box():
    # a 2x2 grid box is fractionally square (0.5 x 0.5), but the page's
    # portrait proportions still make it narrower than tall in practice
    assert is_wide_box((0.0, 0.0, 0.5, 0.5)) is False


def test_story_roundtrips_through_dict(tmp_path):
    story = Story(
        id="test",
        title="Test Story",
        style_prompt="monochrome manga",
        pages=[
            Page(
                layout="H2",
                panels=[
                    Panel(
                        scene_description="a scene",
                        characters=["Aiko"],
                        locations=["Mill"],
                        props=["Letter"],
                        camera_hint="close-up",
                        dialogue=[DialogueLine(speaker="Aiko", text="Hello", kind="speech")],
                    ),
                    Panel(scene_description="another scene"),
                ],
            )
        ],
    )
    path = tmp_path / "story.json"
    story.save(path)
    reloaded = Story.load(path)
    assert asdict(reloaded) == asdict(story)
