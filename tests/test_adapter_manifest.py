import hashlib
import json

import pytest

from manga_pipeline.adapter_manifest import load_manifest, manifest_bytes, sha256_file, validate_manifest, validate_training_metadata


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


def test_manifest_accepts_training_metadata():
    manifest = _manifest()
    manifest["training"] = {"method": "lora", "rank": 8, "examples": 12}
    validate_manifest(manifest)
