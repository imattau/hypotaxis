import json

from package_adapter import main


def test_package_adapter_creates_verified_bundle_and_event(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    output = tmp_path / "release"
    assert main(
        [
            str(source),
            str(output),
            "--name",
            "grounding",
            "--version",
            "1.0.0",
            "--base-model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--license",
            "MIT",
            "--file",
            "adapter_model.safetensors",
            "--blossom",
            "https://mirror.example/" + "a" * 64,
            "--magnet",
            "magnet:?xt=urn:btih:test",
            "--nostr-pubkey",
            "1" * 64,
            "--created-at",
            "123",
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    event = json.loads((output / "nostr-release-event.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["path"] == "adapter_model.safetensors"
    assert event["kind"] == 30078
    assert (output / "adapter_model.safetensors").exists()


def test_package_adapter_does_not_include_unselected_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "release"
    main(
        [
            str(source),
            str(output),
            "--name",
            "grounding",
            "--version",
            "1.0.0",
            "--base-model",
            "base",
            "--license",
            "MIT",
            "--file",
            "adapter_model.safetensors",
        ]
    )
    assert not (output / "adapter_config.json").exists()
