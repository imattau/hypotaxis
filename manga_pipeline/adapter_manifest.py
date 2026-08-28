"""Portable metadata and integrity helpers for shared Hypotaxis adapters.

The manifest is deliberately transport-neutral: Nostr events can publish it,
Blossom can host the referenced blobs, and BitTorrent can distribute the same
bundle.  Cryptographic signing is kept outside this module so the app can use
the creator's preferred Nostr signing implementation without making a crypto
dependency mandatory for local training.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict[str, Any]) -> bytes:
    """Serialize manifest data deterministically for signing or hashing."""

    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def validate_training_metadata(training: Any) -> None:
    """Validate optional reproducibility metadata for a trained adapter."""

    if not isinstance(training, dict):
        raise ValueError("training metadata must be an object")
    if "method" in training and (not isinstance(training["method"], str) or not training["method"].strip()):
        raise ValueError("training.method must be a non-empty string")
    if "rank" in training and (not isinstance(training["rank"], int) or isinstance(training["rank"], bool) or training["rank"] <= 0):
        raise ValueError("training.rank must be a positive integer")
    if "dataset" in training and (not isinstance(training["dataset"], str) or not training["dataset"].strip()):
        raise ValueError("training.dataset must be a non-empty string")
    if "dataset_sha256" in training:
        _require_sha256(training["dataset_sha256"], "training.dataset_sha256")
    if "examples" in training and (not isinstance(training["examples"], int) or isinstance(training["examples"], bool) or training["examples"] < 0):
        raise ValueError("training.examples must be a non-negative integer")


def validate_evaluations(evaluations: Any) -> None:
    """Validate optional benchmark results published with an adapter."""

    if not isinstance(evaluations, list):
        raise ValueError("evaluations must be a list")
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ValueError("each evaluation must be an object")
        for field in ("name", "dataset", "score"):
            if field not in evaluation:
                raise ValueError(f"evaluation missing {field}")
        if not isinstance(evaluation["name"], str) or not evaluation["name"].strip():
            raise ValueError("evaluation.name must be a non-empty string")
        if not isinstance(evaluation["dataset"], str) or not evaluation["dataset"].strip():
            raise ValueError("evaluation.dataset must be a non-empty string")
        score = evaluation["score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError("evaluation.score must be between 0 and 1")
        if "dataset_sha256" in evaluation:
            _require_sha256(evaluation["dataset_sha256"], "evaluation.dataset_sha256")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the interoperable, security-relevant manifest fields."""

    required = ("schema", "name", "version", "base_model", "files", "license")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
    if manifest["schema"] != "hypotaxis.adapter.v1":
        raise ValueError("unsupported adapter manifest schema")
    if not all(isinstance(manifest[field], str) and manifest[field].strip() for field in required if field != "files"):
        raise ValueError("manifest string fields must be non-empty strings")
    license_value = manifest["license"]
    if len(license_value) > 200 or any(ord(character) < 32 for character in license_value):
        raise ValueError("manifest license must be a short single-line value")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ValueError("manifest files must be a non-empty list")
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("each manifest file needs a path")
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError("manifest file paths must stay inside the bundle")
        _require_sha256(entry.get("sha256"), f"sha256 for {entry['path']}")
    distribution = manifest.get("distribution", {})
    if not isinstance(distribution, dict):
        raise ValueError("distribution must be an object")
    torrent = distribution.get("torrent")
    if torrent is not None and not isinstance(torrent, dict):
        raise ValueError("distribution.torrent must be an object")
    blossom = distribution.get("blossom")
    if blossom is not None and not isinstance(blossom, list):
        raise ValueError("distribution.blossom must be a list")
    if "training" in manifest:
        validate_training_metadata(manifest["training"])
    if "evaluations" in manifest:
        validate_evaluations(manifest["evaluations"])


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON adapter manifest."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("adapter manifest must contain a JSON object")
    validate_manifest(manifest)
    return manifest


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Return canonical bytes for signing; excludes the optional signature."""

    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    validate_manifest(unsigned)
    return canonical_json(unsigned)
