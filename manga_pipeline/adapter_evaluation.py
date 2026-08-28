"""Small, model-agnostic evaluators for shared adapter test corpora."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .adapter_manifest import validate_evaluations


def normalize_text(value: str) -> str:
    """Normalize generated text for a stable exact-match comparison."""

    if not isinstance(value, str):
        raise ValueError("evaluation text must be a string")
    return re.sub(r"\s+", " ", value).strip().casefold()


def build_exact_match_evaluation(
    predictions: Iterable[dict[str, Any]],
    *,
    dataset: str,
    dataset_sha256: str | None = None,
    name: str = "caption-exact-match",
) -> dict[str, Any]:
    """Build a manifest evaluation record from reference/prediction pairs.

    Each input row must contain string ``reference`` and ``prediction`` keys.
    Matching is deliberately strict after whitespace/case normalization so the
    resulting score is deterministic and easy to reproduce across clients.
    """

    rows = list(predictions)
    if not rows:
        raise ValueError("at least one prediction is required")
    matches = 0
    for row in rows:
        if not isinstance(row, dict) or "reference" not in row or "prediction" not in row:
            raise ValueError("each prediction needs reference and prediction text")
        if normalize_text(row["reference"]) == normalize_text(row["prediction"]):
            matches += 1
    record: dict[str, Any] = {
        "name": name,
        "dataset": dataset,
        "score": matches / len(rows),
        "examples": len(rows),
    }
    if dataset_sha256 is not None:
        record["dataset_sha256"] = dataset_sha256
    validate_evaluations([record])
    return record


def build_camera_accuracy_evaluation(
    predictions: Iterable[dict[str, Any]],
    *,
    dataset: str,
    dataset_sha256: str | None = None,
    name: str = "caption-camera-accuracy",
) -> dict[str, Any]:
    """Build an evaluation record for structured camera-hint predictions."""

    rows = list(predictions)
    if not rows:
        raise ValueError("at least one prediction is required")
    matches = 0
    for row in rows:
        if not isinstance(row, dict) or "reference_camera" not in row or "prediction_camera" not in row:
            raise ValueError("each prediction needs reference_camera and prediction_camera")
        if normalize_text(row["reference_camera"]) == normalize_text(row["prediction_camera"]):
            matches += 1
    record: dict[str, Any] = {"name": name, "dataset": dataset, "score": matches / len(rows), "examples": len(rows)}
    if dataset_sha256 is not None:
        record["dataset_sha256"] = dataset_sha256
    validate_evaluations([record])
    return record
