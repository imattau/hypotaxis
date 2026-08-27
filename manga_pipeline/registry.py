from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CharacterEntry:
    name: str
    description: str = ""
    reference_image: str = ""


class CharacterRegistry:
    """Per-story character identity registry (Stage B).

    Keeps one canonical text description per character so every panel that
    references them, on any page, is prompted with the same visual identity
    words instead of re-describing them from scratch each time.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, CharacterEntry] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._entries = {name: CharacterEntry(**entry) for name, entry in data.items()}

    def get(self, name: str) -> CharacterEntry | None:
        return self._entries.get(name)

    def set_description(self, name: str, description: str) -> None:
        existing = self._entries.get(name)
        reference_image = existing.reference_image if existing else ""
        self._entries[name] = CharacterEntry(name=name, description=description, reference_image=reference_image)
        self._save()

    def set_reference_image(self, name: str, path: str) -> None:
        existing = self._entries.get(name) or CharacterEntry(name=name)
        existing.reference_image = path
        self._entries[name] = existing
        self._save()

    def all(self) -> dict[str, CharacterEntry]:
        return dict(self._entries)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(entry) for name, entry in self._entries.items()}
        self.path.write_text(json.dumps(data, indent=2))
