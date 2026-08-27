from __future__ import annotations

import argparse
import json

from manga_pipeline.train_captioner import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: LoRA-fine-tune a small seq2seq model on harvested {prose,caption} pairs."
    )
    parser.add_argument("--dataset", default="data/caption_pairs.jsonl")
    parser.add_argument("--output-dir", default="models/captioner")
    parser.add_argument("--base-model", default="google-t5/t5-small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    args = parser.parse_args()

    metrics = train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        min_similarity=args.min_similarity,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
