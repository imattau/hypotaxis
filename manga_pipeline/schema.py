from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import atomic_write_text


@dataclass
class DialogueLine:
    speaker: str
    text: str
    kind: str = "speech"  # speech | thought | narration

    @staticmethod
    def from_dict(d: dict) -> "DialogueLine":
        return DialogueLine(speaker=d["speaker"], text=d["text"], kind=d.get("kind", "speech"))


@dataclass
class Panel:
    scene_description: str
    characters: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    props: list[str] = field(default_factory=list)
    camera_hint: str = "medium shot"
    dialogue: list[DialogueLine] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Panel":
        return Panel(
            scene_description=d["scene_description"],
            characters=d.get("characters", []),
            locations=d.get("locations", []),
            props=d.get("props", []),
            camera_hint=d.get("camera_hint", "medium shot"),
            dialogue=[DialogueLine.from_dict(x) for x in d.get("dialogue", [])],
        )


@dataclass
class Page:
    layout: str
    panels: list[Panel]

    @staticmethod
    def from_dict(d: dict) -> "Page":
        panels = [Panel.from_dict(p) for p in d["panels"]]
        return Page(layout=d["layout"], panels=panels)


@dataclass
class Story:
    id: str
    title: str
    style_prompt: str
    pages: list[Page]

    @staticmethod
    def from_dict(d: dict) -> "Story":
        pages = [Page.from_dict(p) for p in d["pages"]]
        return Story(id=d["id"], title=d["title"], style_prompt=d.get("style_prompt", ""), pages=pages)

    @staticmethod
    def load(path: str | Path) -> "Story":
        data = json.loads(Path(path).read_text())
        return Story.from_dict(data)

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(asdict(self), indent=2) + "\n")
