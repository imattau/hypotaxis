from __future__ import annotations

import argparse
from pathlib import Path

from manga_pipeline.llm import SmallLLM
from manga_pipeline.registry import CharacterRegistry
from manga_pipeline.story_adapt import adapt_story, parse_character_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A: convert a prose story into a Story JSON script.")
    parser.add_argument("prose", help="path to a plain-text story file")
    parser.add_argument("--id", required=True, help="story id, used for output paths")
    parser.add_argument("--title", required=True)
    parser.add_argument("--style-prompt", default="")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--registry-dir", default="registry")
    parser.add_argument("--output", default=None, help="output Story JSON path (default: stories/<id>.json)")
    parser.add_argument(
        "--dataset", default="data/caption_pairs.jsonl", help="append harvested {input,target} caption pairs here for Phase 3 training"
    )
    parser.add_argument("--no-dataset", action="store_true", help="don't harvest training pairs from this run")
    parser.add_argument(
        "--character-profiles",
        default=None,
        help="path to a cast sheet file, one 'Name: description' per line, one character per line - "
        "guarantees these names are recognized and seeds their Stage B descriptions "
        "(see stories/character_profiles.example.txt for the format and an example)",
    )
    args = parser.parse_args()

    text = Path(args.prose).read_text()
    registry = CharacterRegistry(Path(args.registry_dir) / f"{args.id}.json")
    llm = SmallLLM(model_id=args.llm, device=args.device)

    character_profiles = None
    if args.character_profiles:
        character_profiles = parse_character_profiles(Path(args.character_profiles).read_text())

    dataset_path = None if args.no_dataset else args.dataset
    story = adapt_story(
        text,
        args.id,
        args.title,
        args.style_prompt,
        registry,
        llm,
        dataset_path=dataset_path,
        character_profiles=character_profiles,
    )

    output_path = Path(args.output) if args.output else Path("stories") / f"{args.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    story.save(output_path)
    print(f"wrote {output_path}")
    print(f"characters: {list(registry.all())}")


if __name__ == "__main__":
    main()
