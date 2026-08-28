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
The headless `POST /api/adapters/discover` response now includes verified,
newest-per-creator composition events alongside adapter releases.
It also returns latest-per-author rating aggregates and deduplicated report
summaries for active releases and compositions, keeping headless clients in
step with the Studio browser.
All discovery subqueries, including revocation and moderation lookups, map
relay transport failures to an explicit HTTP 502 response.

The Studio browser now uses bundled `nostr-tools` for discovery. Its
`SimplePool` queries multiple relays and `verifyEvent` validates NIP-01 event
signatures before release metadata is displayed. The Python relay query remains
available for headless/integration use, while the browser is the primary
user-facing Nostr client.

The frontend uses a checked-in npm lockfile and a local bundle rather than a
runtime CDN dependency, which keeps the desktop wrapper usable offline after
the UI assets have been built.
CI rebuilds the bundle and checks for a clean diff, so changes to
`studio/static/app.js` cannot silently ship with stale browser output.
GitHub Actions also runs a separate Ubuntu job with the optional distribution
dependencies installed, exercising the real Schnorr, relay, and BitTorrent
integration paths in addition to the lightweight compatibility tests.

Studio's Community Discovery form now queries user-supplied relays and displays
release metadata after browser-side Nostr signature verification.
It can also load the signed NIP-65 kind-10002 read-relay list for the current
NIP-07 identity, using the manually entered relays as bootstrap sources.
Discovery keeps only the newest verified event for each creator and parameterized
adapter identity, avoiding duplicate cards for superseded release versions.
Equal timestamps are resolved by event ID, making relay ordering unable to change
which replaceable record is selected.

When a discovered release advertises Blossom mirrors, Studio can now request a
verified install. The request must include the complete signed Nostr release
event; Studio checks that the event's manifest exactly matches the requested
manifest and verifies its Schnorr signature whenever `coincurve` is available.
Each declared file is then downloaded to a temporary directory, checked against
its manifest digest, and atomically moved into the local shared adapter registry
only after the complete bundle verifies. This prevents a client from replacing
trusted release metadata with a different manifest at install time.

For production deployments, set `HYPOTAXIS_REQUIRE_NOSTR_SIGNATURES=1`. In this
strict mode discovery and every remote release or composition installation fail
closed unless `coincurve` is installed and the Nostr event signatures verify.
Revocation events are also signature-checked before they can hide an artifact.
The default remains compatibility-oriented for lightweight local environments;
those environments report signatures as unavailable rather than claiming they
are verified.

BitTorrent support is isolated behind the optional `libtorrent` dependency.
`create_torrent` creates a torrent for a verified bundle, while
`download_torrent` downloads a magnet, reports progress, and verifies the
resulting bundle before returning. The rest of Hypotaxis remains usable when
`libtorrent` is not installed; callers receive `TorrentUnavailableError` and
can fall back to Blossom mirrors.
For optional server-side relay queries, Schnorr verification, and BitTorrent,
install `requirements-distribution.txt`; the default Studio requirements stay
lightweight and do not pull these packages in.
`TorrentSeeder` is an explicit opt-in long-lived session for keeping a
completed bundle available to peers; downloading alone does not start seeding.
Studio persists only explicit seeding choices in
`models/shared_adapters/.seeding.json` and restores those choices on startup;
the application lifespan closes active sessions cleanly on shutdown.
Studio exposes Start/Stop Seeding controls for verified local bundles and
reports the active seeder's peer and upload-rate state.
Removing a local bundle also stops its matching seeder before deleting the
bundle and adjacent torrent metadata.
Torrent downloads initiated by Studio also require the signed release event and
use the same manifest/event and Schnorr checks as Blossom installation before a
job is launched.

Studio's Adapters tab now reports local BitTorrent availability and metadata
state, and can create a `.torrent` for a packaged bundle. The torrent is stored
beside (not inside) the bundle and the UI reports the resulting path.

Community Discovery compares release versions with the local registry. A newer
verified release is labeled as an Update and can be installed through Blossom
or BitTorrent; the prior local version is retained so updates remain reversible.
Already-installed or older releases do not show an install action.

For resilience, the initial implementation should mirror complete bundles to
multiple Blossom servers. BitTorrent already splits transfers into pieces;
custom sharding across Blossom servers can be considered later if storage
limits require it. Downloads should retain an HTTPS fallback and verify the
manifest signature, file hashes, base-model compatibility, and license before
installation.
In the Studio browser, a failed BitTorrent download automatically falls back
to the release's advertised Blossom mirrors when available. The fallback uses
the same signed release event and verified, atomic Blossom installation path.
Composition installs apply the inverse fallback as well: when a component
advertises both transports and its Blossom mirrors fail, Studio tries its
verified BitTorrent magnet before aborting the composition transaction.

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
Compositions can be marked as `community_merge`; that flag requires at least
one validated evaluation record with a SHA-256 digest for the evaluated
dataset. This prevents an evaluation-free or non-reproducible merge from
being presented as a community release. Local compositions remain opt-in and
may omit the flag and evaluations.
Studio can publish a composition as a signed Nostr kind-30079 event. The event
content contains the complete composition, including component manifest
digests, weights, lineage, and evaluation evidence, so other clients can audit
the merge before reconstructing it locally.
Community Discovery also shows verified remote compositions and offers
component installation when every referenced adapter release is present with a
Blossom source or torrent magnet. The server verifies all signed release events
and lineage digests before downloading any component, preferring Blossom and
falling back to BitTorrent for torrent-only components.

Adapter manifests may also include a `training` object with reproducibility
fields such as `method`, `rank`, `dataset`, `dataset_sha256`, and `examples`.
These fields are optional for backward compatibility but are validated when
present and are accepted by the Studio packaging API.
Studio's packaging form now exposes both training metadata and evaluation
records as optional JSON fields, and the local registry summarizes them after
the bundle is written.

Release trust helpers support NIP-09 revocation requests for both adapter
releases and compositions, plus NIP-32 report labels. Revocations require the
artifact author's pubkey and clients can use `release_is_revoked` to hide
matching artifacts; Studio exposes reporting for both artifact types. Report
labels preserve a separate moderation trail without modifying the release
manifest.
Generated revocations include both the exact event (`e`) reference and the
parameterized replaceable-event (`a`) address when available, allowing clients
to handle either NIP-09 form.
The report and rating builders only target Hypotaxis release (kind 30078) and
composition (kind 30079) events, preventing accidental labels on unrelated
Nostr events.
Manifest license metadata is required to be a short, single-line value; this
preserves support for SPDX identifiers, URLs, and custom license references
without allowing control-character or oversized metadata injection.
Studio discovery also supports signed 1–5 ratings using the
`hypotaxis.adapter.rating` label namespace for both releases and compositions.
It keeps the latest rating from each author and displays average/count for
active releases and compositions.
Discovery also summarizes verified NIP-32 report labels per active artifact,
deduplicating repeated reports from the same author and showing the reported
reasons alongside the artifact. Reports are informational; revocation remains
an author-controlled NIP-09 action.
The browser applies the same lowercase 2–64 character report-label validation
as the Python event builders and requires at least one configured relay before
publishing a report.
The browser also honors both NIP-09 event-reference (`e`) and replaceable
address (`a`) revocations, matching the headless client behavior.
Before downloading a remote release or composition, Studio asks the user to
acknowledge the declared license(s). The acknowledgement covers both direct
Blossom installs and BitTorrent downloads, including Blossom fallback after a
torrent failure; it does not impose a fixed SPDX allow-list.
The API also requires an explicit `license_acknowledged: true` field on remote
install and torrent-download requests, so callers cannot bypass this safeguard
by skipping the browser UI.
The packaging form can load the author's verified NIP-10063 Blossom server
list from the configured Nostr relays and use the resulting servers as mirror
targets. Only signed events and HTTP(S) server URLs are accepted.

To prepare a local release bundle:

```bash
python package_adapter.py path/to/adapter release/grounding-1.0.0 \
  --name grounding --version 1.0.0 \
  --base-model Qwen/Qwen2.5-7B-Instruct --license CC-BY-4.0 \
  --file adapter_model.safetensors \
  --training-method lora --training-rank 16 \
  --training-dataset curated-caption-v1 \
  --evaluations-json evaluations.json \
  --nostr-pubkey <64-character-hex-pubkey>
```

The command writes a verified `manifest.json` and, when a public key is
provided, an unsigned `nostr-release-event.json`. It does not sign anything;
that boundary remains with the browser/NIP-07 signer. The Studio API exposes
authenticated Blossom upload and mirror routes for a signed `Nostr ...`
authorization header, while relay publication remains a browser-side
`nostr-tools` operation.
Training declaration flags and `--evaluations-json` use the same validation as
the Studio packaging API, including optional dataset SHA-256 and example
counts. `--training-dataset-file` can compute the dataset digest and count
non-empty JSONL/text records automatically, rejecting conflicting overrides.
Existing sidecars can be imported with `--training-metadata-json`; explicit
training flags override matching sidecar fields and all resulting metadata is
validated before the bundle is written.

Shared-corpus comparisons can use `manga_pipeline.adapter_evaluation`:
`build_exact_match_evaluation` accepts reference/prediction pairs, performs a
deterministic case/whitespace-normalized comparison, and returns a validated
evaluation record suitable for `--evaluations-json` or a Studio manifest.
The `evaluate_adapter.py` command wraps this for JSONL files:

```bash
python evaluate_adapter.py predictions.jsonl \
  --dataset curated-caption-v1 --output evaluations.json
```
Pass `--include-camera` when rows also contain `reference_camera` and
`prediction_camera` fields; the command then emits both caption exact-match
and structured camera-accuracy records.

Character-LoRA runs, including Studio background jobs, also write `training-metadata.json` beside the trained
weights. It records the story and character identity, base checkpoint, rank,
steps, learning rate, seed, resolution, and number of generated training examples so the resulting
adapter can be packaged with auditable training provenance.
