from __future__ import annotations

import argparse

from manga_pipeline.config import PipelineConfig
from manga_pipeline.pipeline import run
from manga_pipeline.schema import Story


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 manga pipeline prototype on a Story JSON file.")
    parser.add_argument("story", help="path to a Story JSON file")
    parser.add_argument("--backend", choices=["mock", "diffusers"], default="mock")
    parser.add_argument("--checkpoint", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--page-width", type=int, default=1024)
    parser.add_argument("--page-height", type=int, default=1536)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--registry-dir", default="registry")
    parser.add_argument("--no-identity-adapter", action="store_true", help="disable the Phase 4 IP-Adapter identity conditioning")
    parser.add_argument("--identity-adapter-scale", type=float, default=0.6)
    args = parser.parse_args()

    cfg = PipelineConfig(
        backend=args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        page_width=args.page_width,
        page_height=args.page_height,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        use_identity_adapter=not args.no_identity_adapter,
        identity_adapter_scale=args.identity_adapter_scale,
    )

    story = Story.load(args.story)
    pdf_path = run(story, cfg)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
