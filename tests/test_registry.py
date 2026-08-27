from __future__ import annotations

from manga_pipeline.registry import CharacterRegistry


def test_set_description_preserves_other_fields(tmp_path):
    """Regression: set_description() used to reconstruct a fresh
    CharacterEntry and silently drop any other already-set field (it only
    special-cased reference_image) - found while adding is_abstract, which
    would have been dropped by a second set_description call the same way.
    """
    registry = CharacterRegistry(tmp_path / "chars.json")
    registry.set_reference_image("Nova", "/some/path.png")
    registry.set_is_abstract("Nova", True)
    registry.set_description("Nova", "updated description")

    entry = registry.get("Nova")
    assert entry.description == "updated description"
    assert entry.reference_image == "/some/path.png"
    assert entry.is_abstract is True


def test_registry_roundtrips_through_disk(tmp_path):
    path = tmp_path / "chars.json"
    registry = CharacterRegistry(path)
    registry.set_description("Jules", "young woman, dark hair")
    registry.set_is_abstract("Nova", True)

    reloaded = CharacterRegistry(path)
    assert reloaded.get("Jules").description == "young woman, dark hair"
    assert reloaded.get("Nova").is_abstract is True
    assert reloaded.get("Nova").description == ""


def test_get_missing_name_returns_none(tmp_path):
    registry = CharacterRegistry(tmp_path / "chars.json")
    assert registry.get("Nobody") is None


def test_delete_removes_entry_and_persists(tmp_path):
    path = tmp_path / "chars.json"
    registry = CharacterRegistry(path)
    registry.set_description("Desire", "a bogus NER hit, not a real character")

    assert registry.delete("Desire") is True
    assert registry.get("Desire") is None
    assert CharacterRegistry(path).get("Desire") is None  # gone on disk too


def test_delete_missing_name_returns_false(tmp_path):
    registry = CharacterRegistry(tmp_path / "chars.json")
    assert registry.delete("Nobody") is False
