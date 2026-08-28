import hashlib
import json

from evaluate_adapter import main


def test_evaluate_adapter_writes_manifest_evaluations(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"reference":"A bright room.","prediction":"a BRIGHT room."}\n'
        '{"reference":"Rain falls.","prediction":"Snow falls."}\n',
        encoding="utf-8",
    )
    output = tmp_path / "evaluations.json"

    assert main([str(predictions), "--dataset", "caption-corpus-v1", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == [{"name": "caption-exact-match", "dataset": "caption-corpus-v1", "score": 0.5, "examples": 2}]


def test_evaluate_adapter_can_include_camera_accuracy(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"reference":"A","prediction":"A","reference_camera":"close-up","prediction_camera":"close-up"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "evaluations.json"

    main([str(predictions), "--dataset", "caption-corpus-v1", "--include-camera", "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [record["name"] for record in payload] == ["caption-exact-match", "caption-camera-accuracy"]
    assert payload[1]["score"] == 1.0


def test_evaluate_adapter_can_fingerprint_dataset_file(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"reference":"A","prediction":"A"}\n', encoding="utf-8")
    dataset = tmp_path / "corpus.jsonl"
    dataset.write_text('{"text":"A"}\n', encoding="utf-8")
    output = tmp_path / "evaluations.json"

    assert main([str(predictions), "--dataset", "caption-corpus-v1", "--dataset-file", str(dataset), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
