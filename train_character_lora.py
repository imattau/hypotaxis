from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_pipeline.backends import DiffusersBackend
from manga_pipeline.character_lora import (
    TRAINING_VIEW_PROMPTS,
    build_training_caption,
    default_lora_output_dir,
    train_character_lora,
)
from manga_pipeline.config import PipelineConfig
from manga_pipeline.registry import CharacterRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a per-character SDXL LoRA (Stage B/Phase 4 identity, opt-in) from a small "
            "set of bootstrapped portrait images - see README's character-LoRA section."
        )
    )
    parser.add_argument("story_id", help="story id whose registry the character belongs to")
    parser.add_argument("character", help="character name, exactly as it appears in the registry")
    parser.add_argument("--style-prompt", default="", help="story's style prompt, used for the bootstrap images too")
    parser.add_argument("--checkpoint", default="stabilityai/sdxl-turbo")
    parser.add_argument("--registry-dir", default="registry")
    parser.add_argument("--output-dir", default="output", help="where bootstrap training images are written")
    parser.add_argument("--models-dir", default="models", help="where the trained LoRA is saved")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--bootstrap-count", type=int, default=len(TRAINING_VIEW_PROMPTS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()

    registry = CharacterRegistry(Path(args.registry_dir) / f"{args.story_id}.json")
    entry = registry.get(args.character)
    if entry is None:
        raise SystemExit(
            f"character '{args.character}' not found in {args.registry_dir}/{args.story_id}.json - "
            "run Stage A/B (adapt + prepare-cast) first"
        )

    cfg = PipelineConfig(backend="diffusers", checkpoint=args.checkpoint, device=args.device, output_dir=args.output_dir)
    backend = DiffusersBackend(cfg)

    print(f"generating {args.bootstrap_count} bootstrap training images for '{args.character}'...")
    image_paths = backend.generate_character_lora_images(
        args.story_id, args.character, args.style_prompt, registry, count=args.bootstrap_count
    )
    captions = [
        build_training_caption(args.character, entry.description, args.style_prompt, view)
        for view in TRAINING_VIEW_PROMPTS[: len(image_paths)]
    ]

    output_dir = default_lora_output_dir(args.models_dir, args.story_id, args.character)
    metrics = train_character_lora(
        image_paths,
        captions,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        rank=args.rank,
        steps=args.steps,
        learning_rate=args.lr,
        resolution=args.resolution,
        device=args.device,
        on_progress=print,
    )

    registry.set_lora_path(args.character, str(output_dir))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
