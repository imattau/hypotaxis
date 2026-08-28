"""Regression tests for the pure-logic pieces of manga_pipeline/train_captioner.py -
building training examples from harvested/curated records. Doesn't touch
transformers/peft/datasets (those are only imported lazily inside train()).
"""

from __future__ import annotations

from manga_pipeline.train_captioner import build_captioner_training_metadata, to_examples


def test_to_examples_builds_joint_caption_camera_target():
    records = [{"input": "Jules walked in.", "characters": ["Jules"], "target": "Jules walks in.", "camera": "medium shot"}]
    examples = to_examples(records)
    assert examples == [
        {
            "source": "caption: characters: Jules\nJules walked in.",
            "target": "CAPTION: Jules walks in.\nCAMERA: medium shot",
        }
    ]


def test_to_examples_falls_back_to_heuristic_when_camera_missing():
    # pre-camera-harvest records have no "camera" field at all
    records = [{"input": "They stood together in the yard.", "characters": ["Jules", "Priya"], "target": "They stand together."}]
    examples = to_examples(records)
    assert examples[0]["target"] == "CAPTION: They stand together.\nCAMERA: wide two-shot"


def test_to_examples_uses_none_for_missing_characters():
    records = [{"input": "Rain fell.", "target": "Rain falls.", "camera": "wide establishing shot"}]
    examples = to_examples(records)
    assert examples[0]["source"] == "caption: characters: none\nRain fell."


def test_captioner_training_metadata_fingerprints_dataset(tmp_path):
    dataset = tmp_path / "caption_pairs.jsonl"
    dataset.write_text('{"input":"A","target":"A"}\n', encoding="utf-8")

    metadata = build_captioner_training_metadata(
        dataset_path=dataset, base_model="google-t5/t5-base", epochs=8,
        total_examples=10, grounded_examples=9, train_examples=8, eval_examples=1,
    )

    assert metadata["method"] == "captioner-lora"
    assert len(metadata["dataset_sha256"]) == 64
    assert metadata["seed"] == 0
