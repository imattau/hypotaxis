# Community adapter distribution

This design uses three complementary layers:

- Nostr relays publish signed Hypotaxis adapter metadata and community
  discussions. Relays carry references, not model weights.
- Blossom servers provide content-addressed HTTPS mirrors for adapter bundles
  and manifests. Clients verify every file against its SHA-256 digest.
- BitTorrent distributes large bundles peer-to-peer and allows users to opt in
  to seeding them.

The first manifest format is `hypotaxis.adapter.v1`, implemented in
`manga_pipeline.adapter_manifest`. It records the base model, adapter files,
license, and optional Blossom/torrent locations. Manifest bytes are canonical
JSON with the optional `signature` field removed, so a Nostr or other signing
implementation can sign the same payload reproducibly.

`manga_pipeline.adapter_distribution` now builds unsigned Nostr release-event
templates, calculates their NIP-01 event IDs, orders Blossom mirrors before a
torrent magnet, and verifies every declared bundle file before installation.
Signing and network transfers remain explicit integration points; importing
Hypotaxis does not require a relay, Blossom, or BitTorrent client.

It also provides `build_manifest` and `write_bundle` for preparing a release
locally. Files must be explicitly selected (or use the supported default file
set), only safe adapter/document extensions are accepted, and the finished
bundle is hash-verified before it is returned to a publishing step.

Release events can be created as unsigned Nostr templates, then validated and
parsed after an external Nostr signer attaches the Schnorr signature. Blossom
`kind:10063` server lists are parsed into deduplicated mirror origins, and
content-addressed blob URLs are generated only from valid HTTP(S) server
locations.

`download_verified_blob` provides the first transport client. It streams a
Blossom blob to a temporary file, enforces a configurable size limit, verifies
the expected SHA-256, and atomically replaces the destination only after
verification. It is intentionally usable with an injected opener for tests;
upload authorization, relay publishing, and BitTorrent remain separate steps.

`download_from_mirrors` retries the same verified download across a deduplicated
list of Blossom URLs and reports each failed mirror if the complete set is
unavailable. `upload_blob` and `mirror_blob` implement Blossom BUD-02 and
BUD-04, including `X-SHA-256`, content metadata, and optional BUD-11
authorization. Private-key signing remains outside Python: callers pass an
already signed kind-24242 event or a `Nostr ...` header from the browser signer.
`upload_bundle_to_servers` verifies the complete bundle before attempting all
configured servers, deduplicates the list, and returns successful descriptors
alongside per-server failures so a temporary mirror outage is recoverable.
`check_blossom_servers` provides a lightweight HEAD probe for configured mirror
origins; Studio exposes it at `POST /api/adapters/blossom/health` for UI and
future source-selection logic.

The Studio browser now uses bundled `nostr-tools` for discovery. Its
`SimplePool` queries multiple relays and `verifyEvent` validates NIP-01 event
signatures before release metadata is displayed. The Python relay query remains
available for headless/integration use, while the browser is the primary
user-facing Nostr client.

The frontend uses a checked-in npm lockfile and a local bundle rather than a
runtime CDN dependency, which keeps the desktop wrapper usable offline after
the UI assets have been built.

Studio's Community Discovery form now queries user-supplied relays and displays
release metadata after browser-side Nostr signature verification.

When a discovered release advertises Blossom mirrors, Studio can now request a
verified install. Each declared file is downloaded to a temporary directory,
checked against its manifest digest, and atomically moved into the local shared
adapter registry only after the complete bundle verifies.

BitTorrent support is isolated behind the optional `libtorrent` dependency.
`create_torrent` creates a torrent for a verified bundle, while
`download_torrent` downloads a magnet, reports progress, and verifies the
resulting bundle before returning. The rest of Hypotaxis remains usable when
`libtorrent` is not installed; callers receive `TorrentUnavailableError` and
can fall back to Blossom mirrors.

Studio's Adapters tab now reports local BitTorrent availability and metadata
state, and can create a `.torrent` for a packaged bundle. The torrent is stored
beside (not inside) the bundle and the UI reports the resulting path.

For resilience, the initial implementation should mirror complete bundles to
multiple Blossom servers. BitTorrent already splits transfers into pieces;
custom sharding across Blossom servers can be considered later if storage
limits require it. Downloads should retain an HTTPS fallback and verify the
manifest signature, file hashes, base-model compatibility, and license before
installation.

The registry should preserve adapter lineage and evaluation results. A merged
community adapter must identify its source adapters and weights, while the
original adapters remain independently downloadable and reversible.

`build_composition` provides the first composition format,
`hypotaxis.adapter-composition.v1`. It records the shared base model and each
component's name, version, manifest digest, and runtime weight. Components must
be unique and use weights from 0 through 2. `compatible_manifests` rejects
merges across different base models before a composition is published.
Studio exposes `POST /api/adapters/composition` to create these manifests from
local bundles. It refuses mismatched base models and retains each source
manifest digest so a composition can be audited or reversed later.

To prepare a local release bundle:

```bash
python package_adapter.py path/to/adapter release/grounding-1.0.0 \
  --name grounding --version 1.0.0 \
  --base-model Qwen/Qwen2.5-7B-Instruct --license CC-BY-4.0 \
  --file adapter_model.safetensors \
  --nostr-pubkey <64-character-hex-pubkey>
```

The command writes a verified `manifest.json` and, when a public key is
provided, an unsigned `nostr-release-event.json`. It does not sign anything;
that boundary remains with the browser/NIP-07 signer. The Studio API exposes
authenticated Blossom upload and mirror routes for a signed `Nostr ...`
authorization header, while relay publication remains a browser-side
`nostr-tools` operation.
