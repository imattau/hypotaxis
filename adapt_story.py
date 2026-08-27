from __future__ import annotations

import argparse
from pathlib import Path

from manga_pipeline.llm import SmallLLM
from manga_pipeline.registry import CharacterRegistry
from manga_pipeline.story_adapt import (
    adapt_story,
    parse_character_profiles,
    parse_location_profiles,
    parse_prop_profiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A: convert a prose story into a Story JSON script.")
    parser.add_argument("prose", help="path to a plain-text story file")
    parser.add_argument("--id", required=True, help="story id, used for output paths")
    parser.add_argument("--title", required=True)
    parser.add_argument("--style-prompt", default="")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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
    parser.add_argument(
        "--location-profiles",
        default=None,
        help="path to a location sheet file, same 'Name: description' format as --character-profiles - "
        "locations have no automatic detection at all, so this is the only way to register them "
        "(see stories/location_profiles.example.txt for the format and an example)",
    )
    parser.add_argument(
        "--prop-profiles",
        default=None,
        help="path to a prop sheet file, same 'Name: description' format - unlike locations, a prop is only "
        "ever text-anchored into the generation prompt, not image-conditioned (see stories/prop_profiles.example.txt)",
    )
    args = parser.parse_args()

    text = Path(args.prose).read_text()
    registry = CharacterRegistry(Path(args.registry_dir) / f"{args.id}.json")
    location_registry = CharacterRegistry(Path(args.registry_dir) / f"{args.id}_locations.json")
    prop_registry = CharacterRegistry(Path(args.registry_dir) / f"{args.id}_props.json")
    llm = SmallLLM(model_id=args.llm, device=args.device)

    character_profiles = None
    abstract_characters = None
    if args.character_profiles:
        character_profiles, abstract_characters = parse_character_profiles(Path(args.character_profiles).read_text())
    location_profiles = None
    if args.location_profiles:
        location_profiles = parse_location_profiles(Path(args.location_profiles).read_text())
    prop_profiles = None
    if args.prop_profiles:
        prop_profiles = parse_prop_profiles(Path(args.prop_profiles).read_text())

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
        abstract_characters=abstract_characters,
        location_registry=location_registry,
        location_profiles=location_profiles,
        prop_registry=prop_registry,
        prop_profiles=prop_profiles,
    )

    output_path = Path(args.output) if args.output else Path("stories") / f"{args.id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    story.save(output_path)
    print(f"wrote {output_path}")
    print(f"characters: {list(registry.all())}")
    print(f"locations: {list(location_registry.all())}")
    print(f"props: {list(prop_registry.all())}")


if __name__ == "__main__":
    main()
