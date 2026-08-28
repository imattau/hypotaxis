"""Create a verified Hypotaxis adapter bundle for community distribution."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from manga_pipeline.adapter_distribution import build_manifest, build_release_event, write_bundle
from manga_pipeline.adapter_manifest import canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a Hypotaxis adapter for distribution")
    parser.add_argument("source", type=Path, help="directory containing adapter artifacts")
    parser.add_argument("output", type=Path, help="directory to create or replace as the release bundle")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--file", dest="files", action="append", help="artifact path relative to source (repeatable)")
    parser.add_argument("--blossom", action="append", default=[], help="Blossom mirror URL (repeatable)")
    parser.add_argument("--magnet", default=None, help="BitTorrent magnet link")
    parser.add_argument("--nostr-pubkey", default=None, help="creator's 64-character Nostr public key")
    parser.add_argument("--created-at", type=int, default=None, help="Nostr event timestamp (defaults to current time)")
    args = parser.parse_args(argv)

    distribution = {}
    if args.blossom:
        distribution["blossom"] = args.blossom
    if args.magnet:
        distribution["torrent"] = {"magnet": args.magnet}
    manifest = build_manifest(
        args.source,
        name=args.name,
        version=args.version,
        base_model=args.base_model,
        license=args.license,
        files=args.files,
        distribution=distribution or None,
    )
    manifest_path = write_bundle(args.source, args.output, manifest)
    if args.nostr_pubkey:
        event = build_release_event(manifest, args.nostr_pubkey, args.created_at if args.created_at is not None else int(time.time()))
        (args.output / "nostr-release-event.json").write_bytes(canonical_json(event))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
