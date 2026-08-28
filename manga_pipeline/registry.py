from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import atomic_write_text


@dataclass
class CharacterEntry:
    name: str
    description: str = ""
    reference_image: str = ""
    is_abstract: bool = False
    lora_path: str = ""


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
        existing = self._entries.get(name) or CharacterEntry(name=name)
        existing.description = description
        self._entries[name] = existing
        self._save()

    def set_reference_image(self, name: str, path: str) -> None:
        existing = self._entries.get(name) or CharacterEntry(name=name)
        existing.reference_image = path
        self._entries[name] = existing
        self._save()

    def set_lora_path(self, name: str, path: str) -> None:
        """Records the directory holding a trained per-character LoRA (see
        manga_pipeline/character_lora.py, train_character_lora.py) -
        DiffusersBackend.generate_panel() loads and activates it for panels
        resolving to this character when cfg.use_character_lora is on."""
        existing = self._entries.get(name) or CharacterEntry(name=name)
        existing.lora_path = path
        self._entries[name] = existing
        self._save()

    def set_is_abstract(self, name: str, value: bool = True) -> None:
        """Marks a character as having no physical/humanoid appearance (an
        AI, a voice, a presence) - see parse_character_profiles' "[no-form]"
        tag. backends.py uses this to skip face/portrait reference-image
        generation and IP-Adapter identity conditioning for this character
        entirely, text-anchoring their description into the prompt instead,
        the same way props are handled."""
        existing = self._entries.get(name) or CharacterEntry(name=name)
        existing.is_abstract = value
        self._entries[name] = existing
        self._save()

    def all(self) -> dict[str, CharacterEntry]:
        return dict(self._entries)

    def delete(self, name: str) -> bool:
        """Removes an entry entirely - e.g. a bogus name NER mistook for a
        person/location/prop (see the pipeline review's alias-merging notes:
        alias merging catches variant spellings of a real name, but not a
        wrong entity-type call like mistaking a capitalized common noun for
        a character - those need to be pruned manually). Returns whether
        anything was actually removed. Does not touch reference_image files
        on disk - the caller (studio API) decides whether to also delete
        those, since a registry doesn't own that filesystem lifecycle.
        """
        if name not in self._entries:
            return False
        del self._entries[name]
        self._save()
        return True

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(entry) for name, entry in self._entries.items()}
        atomic_write_text(self.path, json.dumps(data, indent=2) + "\n")
