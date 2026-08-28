import hashlib
import json

import pytest

from manga_pipeline.adapter_manifest import load_manifest, manifest_bytes, sha256_file, validate_evaluations, validate_manifest, validate_training_metadata


def _manifest():
    return {
        "schema": "hypotaxis.adapter.v1",
        "name": "grounded-captioner",
        "version": "1.0.0",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "license": "CC-BY-4.0",
        "files": [{"path": "adapter_model.safetensors", "sha256": "a" * 64}],
        "distribution": {
            "torrent": {"magnet": "magnet:?xt=urn:btih:example"},
            "blossom": ["https://blossom.example/" + "b" * 64],
        },
    }


def test_manifest_validates_and_canonicalizes():
    manifest = _manifest()
    validate_manifest(manifest)
    assert manifest_bytes(manifest).endswith(b"\n")
    assert json.loads(manifest_bytes(manifest))["name"] == "grounded-captioner"


def test_manifest_rejects_path_escape():
    manifest = _manifest()
    manifest["files"][0]["path"] = "../adapter_model.safetensors"
    with pytest.raises(ValueError, match="inside the bundle"):
        validate_manifest(manifest)


def test_manifest_rejects_multiline_or_oversized_license():
    manifest = _manifest()
    manifest["license"] = "MIT\nnotice"
    with pytest.raises(ValueError, match="single-line"):
        validate_manifest(manifest)
    manifest["license"] = "x" * 201
    with pytest.raises(ValueError, match="short"):
        validate_manifest(manifest)


def test_load_manifest_and_hash_file(tmp_path):
    payload = b"adapter payload"
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(payload)
    assert sha256_file(adapter) == hashlib.sha256(payload).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert load_manifest(path)["version"] == "1.0.0"


def test_training_metadata_validates_reproducibility_fields():
    validate_training_metadata(
        {
            "method": "qlora",
            "rank": 16,
            "dataset": "community-caption-set-v2",
            "dataset_sha256": "a" * 64,
            "examples": 2400,
        }
    )
    with pytest.raises(ValueError, match="positive integer"):
        validate_training_metadata({"rank": 0})
    with pytest.raises(ValueError, match="dataset_sha256"):
        validate_training_metadata({"dataset_sha256": "not-a-digest"})
    with pytest.raises(ValueError, match="training.steps"):
        validate_training_metadata({"steps": 0})
    with pytest.raises(ValueError, match="learning_rate"):
        validate_training_metadata({"learning_rate": 0})
    with pytest.raises(ValueError, match="seed"):
        validate_training_metadata({"seed": True})


def test_training_metadata_accepts_character_lora_reproducibility_fields():
    validate_training_metadata({
        "method": "character-lora", "base_model": "sdxl", "rank": 8,
        "steps": 300, "learning_rate": 1e-4, "resolution": 1024,
        "examples": 8, "seed": 23,
    })


def test_manifest_accepts_training_metadata():
    manifest = _manifest()
    manifest["training"] = {"method": "lora", "rank": 8, "examples": 12}
    validate_manifest(manifest)


def test_manifest_rejects_training_base_model_mismatch():
    manifest = _manifest()
    manifest["training"] = {"method": "lora", "base_model": "different-base"}
    with pytest.raises(ValueError, match="training.base_model"):
        validate_manifest(manifest)


def test_manifest_accepts_matching_training_base_model():
    manifest = _manifest()
    manifest["training"] = {"method": "lora", "base_model": manifest["base_model"]}
    validate_manifest(manifest)


def test_evaluations_validate_scores_and_dataset_identity():
    validate_evaluations([{"name": "caption-bleu", "dataset": "corpus-v1", "dataset_sha256": "b" * 64, "score": 0.82}])
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_evaluations([{"name": "x", "dataset": "d", "score": 1.1}])
    with pytest.raises(ValueError, match="missing score"):
        validate_evaluations([{"name": "x", "dataset": "d"}])


def test_manifest_accepts_evaluations():
    manifest = _manifest()
    manifest["evaluations"] = [{"name": "caption-bleu", "dataset": "corpus-v1", "score": 0.8}]
    validate_manifest(manifest)
