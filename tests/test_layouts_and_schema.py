from __future__ import annotations

from dataclasses import asdict

from manga_pipeline.layouts import LAYOUTS, boxes_for, panel_count
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
