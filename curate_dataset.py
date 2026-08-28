from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from manga_pipeline.llm import SmallLLM
from manga_pipeline.registry import CharacterRegistry
from manga_pipeline.story_adapt import adapt_story

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate teacher-quality caption candidates for LoRA dataset curation "
            "(Phase 3). Runs a stronger LLM than the production bridge model over one "
            "or more source stories and harvests {input, characters, target} pairs, "
            "the same way normal Stage A use does - but from a better teacher, and "
            "written to a separate 'candidates' file for human review rather than "
            "straight into the training set - open the studio UI's Dataset tab to review them."
        )
    )
    parser.add_argument("stories", nargs="+", help="path(s) to plain-text story files to harvest from")
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="teacher LLM - deliberately stronger than the production bridge model (3B default), "
        "since candidates only need to be good enough for a human to review, not run in production",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize",
        help="load the teacher at full precision instead of 4-bit - needs more VRAM "
        "(~14GB for a 7B model vs. ~4-5GB in 4-bit), only worth it if you have room to spare",
    )
    parser.add_argument(
        "--out", default="data/caption_candidates.jsonl", help="candidates file to append to (see the studio's Dataset tab)"
    )
    args = parser.parse_args()

    llm = SmallLLM(model_id=args.teacher, device=args.device, quantize=args.quantize)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for story_path in args.stories:
        story_path = Path(story_path)
        print(f"harvesting from {story_path}...")
        text = story_path.read_text()
        # a throwaway registry, scoped to a temp dir - candidate generation
        # doesn't need to persist a character registry the way a real Stage A
        # adaptation does, it only cares about the harvested {input,target} pairs
        with tempfile.TemporaryDirectory() as tmp:
            registry = CharacterRegistry(Path(tmp) / "registry.json")
            location_registry = CharacterRegistry(Path(tmp) / "locations.json")
            prop_registry = CharacterRegistry(Path(tmp) / "props.json")
            try:
                adapt_story(
                    text,
                    story_path.stem,
                    story_path.stem,
                    "",
                    registry,
                    llm,
                    dataset_path=out_path,
                    location_registry=location_registry,
                    prop_registry=prop_registry,
                )
            except ValueError as e:
                print(f"  skipped ({e})")
                continue
        print(f"  done")

    print(f"wrote candidates to {out_path} - run curate_review.py to review them")


if __name__ == "__main__":
    main()
