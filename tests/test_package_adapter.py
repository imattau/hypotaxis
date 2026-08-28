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
            "--training-method",
            "qlora",
            "--training-rank",
            "16",
            "--training-dataset",
            "caption-corpus-v1",
            "--training-dataset-sha256",
            "b" * 64,
            "--training-examples",
            "2400",
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    event = json.loads((output / "nostr-release-event.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["path"] == "adapter_model.safetensors"
    assert manifest["training"]["dataset_sha256"] == "b" * 64
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


def test_package_adapter_reads_evaluation_records(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    evaluations = tmp_path / "evaluations.json"
    evaluations.write_text(json.dumps([{"name": "heldout", "dataset": "corpus-v1", "score": 0.84}]), encoding="utf-8")

    main([
        str(source), str(tmp_path / "release"), "--name", "grounding", "--version", "1.0.0",
        "--base-model", "base", "--license", "MIT", "--evaluations-json", str(evaluations),
    ])

    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluations"][0]["score"] == 0.84


def test_package_adapter_fingerprints_training_dataset_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    dataset = tmp_path / "curated.jsonl"
    dataset.write_text('{"text":"one"}\n\n{"text":"two"}\n', encoding="utf-8")

    main([
        str(source), str(tmp_path / "release"), "--name", "grounding", "--version", "1.0.0",
        "--base-model", "base", "--license", "MIT", "--training-dataset-file", str(dataset),
    ])

    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["dataset"] == "curated.jsonl"
    assert manifest["training"]["examples"] == 2
    assert len(manifest["training"]["dataset_sha256"]) == 64


def test_package_adapter_imports_training_metadata_sidecar(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    sidecar = tmp_path / "training-metadata.json"
    sidecar.write_text(json.dumps({"method": "character-lora", "character": "Jules", "rank": 8, "examples": 7}), encoding="utf-8")

    main([
        str(source), str(tmp_path / "release"), "--name", "jules", "--version", "1.0.0",
        "--base-model", "base", "--license", "MIT", "--training-metadata-json", str(sidecar),
    ])

    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training"]["method"] == "character-lora"
    assert manifest["training"]["character"] == "Jules"
