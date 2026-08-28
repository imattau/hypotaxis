"""Create manifest-compatible evaluation records from caption predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_pipeline.adapter_evaluation import build_camera_accuracy_evaluation, build_exact_match_evaluation
from manga_pipeline.adapter_manifest import sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate adapter predictions against a shared corpus")
    parser.add_argument("predictions", type=Path, help="JSONL file with reference and prediction fields")
    parser.add_argument("--dataset", required=True, help="shared corpus name or version")
    parser.add_argument("--dataset-sha256", default=None, help="optional lowercase SHA-256 digest of the corpus")
    parser.add_argument("--dataset-file", type=Path, default=None, help="corpus file to fingerprint when publishing reproducible evaluations")
    parser.add_argument("--include-camera", action="store_true", help="also score reference_camera/prediction_camera fields")
    parser.add_argument("--output", type=Path, default=None, help="write an evaluation JSON array to this file")
    args = parser.parse_args(argv)
    if args.dataset_sha256 and args.dataset_file:
        parser.error("use either --dataset-sha256 or --dataset-file, not both")
    dataset_sha256 = args.dataset_sha256 or (sha256_file(args.dataset_file) if args.dataset_file else None)

    rows = []
    for line_number, line in enumerate(args.predictions.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid JSON on predictions line {line_number}: {exc.msg}")
        rows.append(row)
    evaluations = [build_exact_match_evaluation(rows, dataset=args.dataset, dataset_sha256=dataset_sha256)]
    if args.include_camera:
        evaluations.append(build_camera_accuracy_evaluation(rows, dataset=args.dataset, dataset_sha256=dataset_sha256))
    payload = json.dumps(evaluations, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
