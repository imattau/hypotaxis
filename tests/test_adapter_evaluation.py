import pytest

from manga_pipeline.adapter_evaluation import build_camera_accuracy_evaluation, build_exact_match_evaluation, normalize_text


def test_normalize_text_is_case_and_whitespace_insensitive():
    assert normalize_text("  A bright   room\n") == "a bright room"


def test_build_exact_match_evaluation_is_manifest_compatible():
    result = build_exact_match_evaluation(
        [
            {"reference": "A bright room.", "prediction": " a BRIGHT room. "},
            {"reference": "Rain falls.", "prediction": "Snow falls."},
        ],
        dataset="caption-corpus-v1",
        dataset_sha256="a" * 64,
    )
    assert result == {
        "name": "caption-exact-match",
        "dataset": "caption-corpus-v1",
        "dataset_sha256": "a" * 64,
        "score": 0.5,
        "examples": 2,
    }


def test_build_exact_match_evaluation_rejects_empty_or_malformed_rows():
    with pytest.raises(ValueError, match="at least one"):
        build_exact_match_evaluation([], dataset="corpus")
    with pytest.raises(ValueError, match="reference and prediction"):
        build_exact_match_evaluation([{"reference": "caption"}], dataset="corpus")


def test_build_camera_accuracy_evaluation_scores_structured_hints():
    result = build_camera_accuracy_evaluation(
        [
            {"reference_camera": "Wide Two-Shot", "prediction_camera": "wide two-shot"},
            {"reference_camera": "close-up", "prediction_camera": "medium shot"},
        ],
        dataset="caption-corpus-v1",
    )
    assert result["name"] == "caption-camera-accuracy"
    assert result["score"] == 0.5
    assert result["examples"] == 2
