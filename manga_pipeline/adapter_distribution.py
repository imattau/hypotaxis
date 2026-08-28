"""Discovery and verification primitives for community adapter releases.

This module does not perform network I/O.  It produces Nostr event payloads,
orders transport sources, and verifies a downloaded bundle.  HTTP/Blossom,
BitTorrent, and Nostr clients can be added without making those dependencies a
requirement for story adaptation or local training.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .adapter_manifest import canonical_json, load_manifest, manifest_bytes, sha256_file, validate_manifest

NOSTR_RELEASE_KIND = 30078
BLOSSOM_SERVER_LIST_KIND = 10063
COMPOSITION_SCHEMA = "hypotaxis.adapter-composition.v1"


class TorrentUnavailableError(RuntimeError):
    """Raised when the optional libtorrent dependency is not installed."""


class NostrUnavailableError(RuntimeError):
    """Raised when the optional Nostr WebSocket dependency is not installed."""


class NostrCryptoUnavailableError(RuntimeError):
    """Raised when the optional secp256k1 Schnorr backend is not installed."""


def _hex(value: Any, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be {length} hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc


def nostr_event_id(event: dict[str, Any]) -> str:
    """Calculate the NIP-01 event id for an unsigned event template."""

    required = ("pubkey", "created_at", "kind", "tags", "content")
    if any(key not in event for key in required):
        raise ValueError("Nostr event is missing required fields")
    payload = [0, event["pubkey"], event["created_at"], event["kind"], event["tags"], event["content"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_signed_event(event: dict[str, Any]) -> None:
    """Validate an event's shape and self-consistent id/signature fields.

    Schnorr signature verification belongs to the selected Nostr client
    implementation.  This function still rejects malformed events before they
    reach that implementation and ensures the id was calculated from the
    event's signed fields.
    """

    for key in ("id", "pubkey", "sig"):
        _hex(event.get(key), 128 if key == "sig" else 64, key)
    expected_id = nostr_event_id(event)
    if event["id"] != expected_id:
        raise ValueError("Nostr event id does not match its signed fields")
    if not isinstance(event.get("kind"), int) or not isinstance(event.get("created_at"), int):
        raise ValueError("Nostr event kind and created_at must be integers")
    if not isinstance(event.get("tags"), list) or not isinstance(event.get("content"), str):
        raise ValueError("Nostr event tags and content have invalid types")


def verify_schnorr_signature(event: dict[str, Any]) -> bool:
    """Verify a NIP-01 event's BIP-340 Schnorr signature."""

    try:
        validate_signed_event(event)
    except ValueError:
        return False
    try:
        from coincurve import PublicKeyXOnly
    except ImportError as exc:
        raise NostrCryptoUnavailableError("Nostr signature verification requires the optional 'coincurve' package") from exc
    try:
        public_key = PublicKeyXOnly(bytes.fromhex(event["pubkey"]))
        return public_key.verify(bytes.fromhex(event["sig"]), bytes.fromhex(event["id"]))
    except ValueError:
        return False


def schnorr_available() -> bool:
    """Return whether the optional BIP-340 verification backend is available."""

    try:
        from coincurve import PublicKeyXOnly  # noqa: F401
    except ImportError:
        return False
    return True


def build_release_event(manifest: dict[str, Any], pubkey: str, created_at: int) -> dict[str, Any]:
    """Build an unsigned Nostr replaceable-event template for a release.

    The caller must sign the returned event using its Nostr key implementation.
    The manifest remains in event content so its canonical bytes can be signed
    independently by systems that also publish it to Blossom.
    """

    validate_manifest(manifest)
    _hex(pubkey, 64, "pubkey")
    if not isinstance(created_at, int) or created_at < 0:
        raise ValueError("created_at must be a non-negative integer")
    tags = [
        ["d", f"adapter:{manifest['name']}"],
        ["version", manifest["version"]],
        ["base-model", manifest["base_model"]],
        ["t", "hypotaxis-adapter"],
    ]
    event = {
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": NOSTR_RELEASE_KIND,
        "tags": tags,
        "content": canonical_json(manifest).decode("utf-8"),
    }
    event["id"] = nostr_event_id(event)
    return event


def parse_release_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract a Hypotaxis manifest from a signed release event."""

    validate_signed_event(event)
    if event["kind"] != NOSTR_RELEASE_KIND:
        raise ValueError("event is not a Hypotaxis adapter release")
    manifest = json.loads(event["content"])
    if not isinstance(manifest, dict):
        raise ValueError("release event content must contain a manifest object")
    validate_manifest(manifest)
    return manifest


def _load_websocket():
    try:
        import websocket
    except ImportError as exc:
        raise NostrUnavailableError("Nostr relay support requires the optional 'websocket-client' package") from exc
    return websocket


def query_nostr_relays(
    relay_urls: Iterable[str],
    filters: Iterable[dict[str, Any]],
    *,
    timeout: int = 10,
    max_events: int = 100,
    connector=None,
) -> list[dict[str, Any]]:
    """Query Nostr relays and return unique, self-consistent events.

    ``connector`` is injectable for tests and must return an object supporting
    ``send``, ``recv``, and ``close``. Signature verification remains the
    responsibility of a Nostr signer/verifier integration.
    """

    relays = list(dict.fromkeys(relay_urls))
    filters = list(filters)
    if not relays:
        raise ValueError("at least one Nostr relay is required")
    if not filters or not all(isinstance(item, dict) for item in filters):
        raise ValueError("filters must be a non-empty list of objects")
    if not isinstance(timeout, int) or timeout <= 0 or not isinstance(max_events, int) or max_events <= 0:
        raise ValueError("timeout and max_events must be positive")
    if connector is None:
        websocket = _load_websocket()

        def connector(url, timeout):
            return websocket.create_connection(url, timeout=timeout)
    events: dict[str, dict[str, Any]] = {}
    for relay in relays:
        parsed = urlparse(relay)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            continue
        connection = None
        subscription_id = uuid.uuid4().hex[:16]
        try:
            connection = connector(relay, timeout)
            connection.send(json.dumps(["REQ", subscription_id, *filters], separators=(",", ":")))
            while len(events) < max_events:
                message = json.loads(connection.recv())
                if not isinstance(message, list) or not message:
                    continue
                if message[0] == "EOSE" and len(message) > 1 and message[1] == subscription_id:
                    break
                if message[0] != "EVENT" or len(message) < 3 or not isinstance(message[2], dict):
                    continue
                event = message[2]
                try:
                    validate_signed_event(event)
                except ValueError:
                    continue
                events[event["id"]] = event
        except (OSError, ValueError, TimeoutError, json.JSONDecodeError):
            continue
        finally:
            if connection is not None:
                connection.close()
    return list(events.values())[:max_events]


def _source_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"https", "http"} and parsed.netloc:
        return value
    if parsed.scheme == "magnet" and value.startswith("magnet:?xt=urn:btih:"):
        return value
    return None


def blossom_servers(event: dict[str, Any]) -> list[str]:
    """Extract unique HTTP(S) Blossom server URLs from a kind 10063 event."""

    validate_signed_event(event)
    if event["kind"] != BLOSSOM_SERVER_LIST_KIND:
        raise ValueError("event is not a Blossom server-list event")
    servers: list[str] = []
    for tag in event["tags"]:
        if isinstance(tag, list) and len(tag) == 2 and tag[0] == "server":
            server = _source_url(tag[1])
            if server is not None and server not in servers:
                servers.append(server.rstrip("/"))
    return servers


def blossom_blob_urls(sha256: str, server_urls: Iterable[str]) -> list[str]:
    """Build HTTPS/HTTP Blossom blob URLs for a validated SHA-256 digest."""

    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("sha256 must be a 64-character digest")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError("sha256 must be hexadecimal") from exc
    urls: list[str] = []
    for server in server_urls:
        parsed = urlparse(server)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Blossom server URL must be an HTTP(S) origin/path without query or fragment")
        url = server.rstrip("/") + "/" + sha256
        if url not in urls:
            urls.append(url)
    return urls


def distribution_sources(manifest: dict[str, Any]) -> list[str]:
    """Return deduplicated sources, preferring Blossom HTTPS mirrors."""

    validate_manifest(manifest)
    distribution = manifest.get("distribution", {})
    sources: list[str] = []
    for value in distribution.get("blossom", []):
        if (source := _source_url(value)) is not None and source not in sources:
            sources.append(source)
    torrent = distribution.get("torrent", {})
    if isinstance(torrent, dict) and (source := _source_url(torrent.get("magnet"))) is not None:
        if source not in sources:
            sources.append(source)
    return sources


def verify_bundle(manifest: dict[str, Any] | str | Path, root: str | Path) -> list[str]:
    """Verify every declared file and return the verified relative paths."""

    manifest = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    validate_manifest(manifest)
    root = Path(root).resolve()
    verified: list[str] = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"manifest file escapes bundle root: {entry['path']}")
        if not path.is_file():
            raise FileNotFoundError(entry["path"])
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"sha256 mismatch for {entry['path']}")
        verified.append(entry["path"])
    return verified


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the digest of the canonical unsigned manifest."""

    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def validate_composition(composition: dict[str, Any]) -> None:
    """Validate a runtime adapter-bank or community-merge description."""

    required = ("schema", "name", "version", "base_model", "components")
    missing = [field for field in required if field not in composition]
    if missing:
        raise ValueError(f"composition missing required fields: {', '.join(missing)}")
    if composition["schema"] != COMPOSITION_SCHEMA:
        raise ValueError("unsupported adapter composition schema")
    for field in required[1:4]:
        if not isinstance(composition[field], str) or not composition[field].strip():
            raise ValueError(f"composition {field} must be a non-empty string")
    components = composition["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("composition components must be a non-empty list")
    names: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("each composition component must be an object")
        for field in ("name", "version", "manifest_sha256", "weight"):
            if field not in component:
                raise ValueError(f"composition component missing {field}")
        if not isinstance(component["name"], str) or not component["name"].strip():
            raise ValueError("composition component name must be non-empty")
        if component["name"] in names:
            raise ValueError(f"duplicate composition component: {component['name']}")
        names.add(component["name"])
        _hex(component["manifest_sha256"], 64, f"manifest_sha256 for {component['name']}")
        weight = component["weight"]
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 2:
            raise ValueError(f"weight for {component['name']} must be between 0 and 2")


def build_composition(
    name: str,
    version: str,
    base_model: str,
    components: Iterable[dict[str, Any]],
    *,
    description: str = "",
) -> dict[str, Any]:
    """Build a deterministic, lineage-preserving adapter composition."""

    composition: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "name": name,
        "version": version,
        "base_model": base_model,
        "components": list(components),
    }
    if description:
        composition["description"] = description
    validate_composition(composition)
    return composition


def compatible_manifests(manifests: Iterable[dict[str, Any]]) -> str:
    """Ensure adapter manifests share a base model and return that model id."""

    manifests = list(manifests)
    if not manifests:
        raise ValueError("at least one adapter manifest is required")
    for manifest in manifests:
        validate_manifest(manifest)
    base_models = {manifest["base_model"] for manifest in manifests}
    if len(base_models) != 1:
        raise ValueError("adapters target incompatible base models")
    return base_models.pop()


def resolve_composition_paths(
    composition: dict[str, Any],
    adapter_root: str | Path,
) -> list[dict[str, Any]]:
    """Resolve and verify local adapter bundles referenced by a composition.

    The returned order and weights are ready for a diffusion pipeline's
    ``load_lora_weights``/``set_adapters`` calls.  Every on-disk manifest must
    match the digest recorded in the composition, preventing silent lineage
    substitution.
    """

    validate_composition(composition)
    adapter_root = Path(adapter_root).resolve()
    resolved: list[dict[str, Any]] = []
    for component in composition["components"]:
        bundle = (adapter_root / f"{component['name']}-{component['version']}").resolve()
        if adapter_root not in bundle.parents:
            raise ValueError("composition component escapes adapter root")
        manifest_path = bundle / "manifest.json"
        manifest = load_manifest(manifest_path)
        if manifest["name"] != component["name"] or manifest["version"] != component["version"]:
            raise ValueError(f"manifest identity mismatch for {component['name']}")
        if manifest_digest(manifest) != component["manifest_sha256"]:
            raise ValueError(f"manifest digest mismatch for {component['name']}")
        if manifest["base_model"] != composition["base_model"]:
            raise ValueError(f"base model mismatch for {component['name']}")
        verify_bundle(manifest, bundle)
        resolved.append({"name": component["name"], "version": component["version"], "path": str(bundle), "weight": component["weight"]})
    return resolved


def available_sources(manifest: dict[str, Any], available: Iterable[str]) -> list[str]:
    """Filter advertised sources by availability while preserving preference."""

    available = set(available)
    return [source for source in distribution_sources(manifest) if source in available]


def build_manifest(
    source_dir: str | Path,
    *,
    name: str,
    version: str,
    base_model: str,
    license: str,
    files: Iterable[str] | None = None,
    distribution: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest from explicitly selected files in an adapter directory."""

    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    selected = list(files) if files is not None else [path.name for path in sorted(source_dir.iterdir()) if path.is_file()]
    entries = []
    for relative_name in selected:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"adapter file path escapes source directory: {relative_name}")
        path = (source_dir / relative).resolve()
        if source_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(relative_name)
        if path.suffix.lower() not in {".safetensors", ".json", ".txt", ".md", ".model", ".bin"}:
            raise ValueError(f"unsupported adapter artifact type: {relative_name}")
        entries.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest: dict[str, Any] = {
        "schema": "hypotaxis.adapter.v1",
        "name": name,
        "version": version,
        "base_model": base_model,
        "license": license,
        "files": entries,
    }
    if distribution:
        manifest["distribution"] = distribution
    if metadata:
        manifest["metadata"] = metadata
    validate_manifest(manifest)
    return manifest


def write_bundle(
    source_dir: str | Path,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Copy declared artifacts and write a canonical manifest into a bundle."""

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    validate_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        source = (source_dir / relative).resolve()
        target = (output_dir / relative).resolve()
        if source_dir not in source.parents or output_dir not in target.parents:
            raise ValueError("bundle file path escapes its root")
        if not source.is_file():
            raise FileNotFoundError(entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    verify_bundle(manifest, output_dir)
    return manifest_path


def download_verified_blob(
    url: str,
    destination: str | Path,
    expected_sha256: str,
    *,
    max_bytes: int = 4 * 1024 * 1024 * 1024,
    opener=urllib.request.urlopen,
) -> Path:
    """Download one Blossom blob, verify it, and atomically install it."""

    _hex(expected_sha256, 64, "expected_sha256")
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("blob URL must be an HTTP(S) URL")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with opener(url) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError("remote blob exceeds maximum size")
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("remote blob exceeds maximum size")
                    digest.update(chunk)
                    handle.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded blob sha256 does not match manifest")
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def download_from_mirrors(
    urls: Iterable[str],
    destination: str | Path,
    expected_sha256: str,
    *,
    max_bytes: int = 4 * 1024 * 1024 * 1024,
    opener=urllib.request.urlopen,
) -> Path:
    """Try Blossom mirrors until one provides a verified blob."""

    urls = list(dict.fromkeys(urls))
    if not urls:
        raise ValueError("at least one mirror URL is required")
    failures: list[str] = []
    for url in urls:
        try:
            return download_verified_blob(
                url,
                destination,
                expected_sha256,
                max_bytes=max_bytes,
                opener=opener,
            )
        except (OSError, ValueError) as exc:
            failures.append(f"{url}: {exc}")
    raise RuntimeError("all adapter mirrors failed: " + "; ".join(failures))


def file_blossom_urls(manifest: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Resolve Blossom server roots or exact blob URLs for one manifest file."""

    validate_manifest(manifest)
    _hex(entry.get("sha256"), 64, f"sha256 for {entry.get('path', 'file')}")
    urls: list[str] = []
    for source in manifest.get("distribution", {}).get("blossom", []):
        if not isinstance(source, str):
            continue
        parsed = urlparse(source)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.query or parsed.fragment:
            continue
        last_segment = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if len(last_segment) == 64 and all(character in "0123456789abcdef" for character in last_segment):
            if last_segment != entry["sha256"]:
                continue
            url = source
        else:
            url = source.rstrip("/") + "/" + entry["sha256"]
        if url not in urls:
            urls.append(url)
    return urls


def blossom_authorization_header(event: dict[str, Any]) -> str:
    """Encode a signed BUD-11 event as an HTTP Authorization header.

    Signing is deliberately left to the caller (the Studio browser uses
    ``nostr-tools``/a NIP-07 signer).  This helper only validates the event
    shape and performs the protocol's base64url encoding.
    """

    validate_signed_event(event)
    if event.get("kind") != 24242:
        raise ValueError("Blossom authorization event must have kind 24242")
    encoded = base64.urlsafe_b64encode(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"Nostr {encoded}"


def _authorization_header(event: dict[str, Any] | None, authorization: str | None) -> str | None:
    if event is not None and authorization is not None:
        raise ValueError("provide either an authorization event or header, not both")
    if event is not None:
        return blossom_authorization_header(event)
    if authorization is not None:
        if not isinstance(authorization, str) or not authorization.startswith("Nostr "):
            raise ValueError("authorization must be a BUD-11 Nostr header")
        return authorization
    return None


def _json_response(response: Any) -> dict[str, Any]:
    payload = response.read()
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Blossom server returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Blossom server response must be an object")
    return value


def upload_blob(
    server_url: str,
    blob_path: str | Path,
    *,
    content_type: str = "application/octet-stream",
    authorization_event: dict[str, Any] | None = None,
    authorization: str | None = None,
    opener=urllib.request.urlopen,
) -> dict[str, Any]:
    """Upload one blob to a Blossom server using BUD-02.

    ``authorization_event`` is an already signed kind-24242 event.  Passing
    the pre-built ``authorization`` header is useful for browser/NIP-07
    signers that keep the private key outside Python.
    """

    blob_path = Path(blob_path).resolve()
    if not blob_path.is_file():
        raise FileNotFoundError(blob_path)
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Blossom server URL must be an HTTP(S) origin/path without query or fragment")
    if not isinstance(content_type, str) or not content_type:
        raise ValueError("content_type must be non-empty")
    digest = sha256_file(blob_path)
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(blob_path.stat().st_size),
        "X-SHA-256": digest,
    }
    auth_header = _authorization_header(authorization_event, authorization)
    if auth_header:
        headers["Authorization"] = auth_header
    with blob_path.open("rb") as body:
        request = urllib.request.Request(
            server_url.rstrip("/") + "/upload",
            data=body,
            headers=headers,
            method="PUT",
        )
        with opener(request) as response:
            descriptor = _json_response(response)
    if descriptor.get("sha256") not in {None, digest}:
        raise ValueError("Blossom descriptor hash does not match uploaded blob")
    return descriptor


def mirror_blob(
    server_url: str,
    source_url: str,
    *,
    authorization_event: dict[str, Any] | None = None,
    authorization: str | None = None,
    opener=urllib.request.urlopen,
) -> dict[str, Any]:
    """Ask a Blossom server to mirror a remote blob using BUD-04."""

    parsed_server = urlparse(server_url)
    parsed_source = urlparse(source_url)
    if parsed_server.scheme not in {"http", "https"} or not parsed_server.netloc or parsed_server.query or parsed_server.fragment:
        raise ValueError("Blossom server URL must be an HTTP(S) origin/path without query or fragment")
    if parsed_source.scheme not in {"http", "https"} or not parsed_source.netloc:
        raise ValueError("source URL must be an HTTP(S) URL")
    headers = {"Content-Type": "application/json"}
    auth_header = _authorization_header(authorization_event, authorization)
    if auth_header:
        headers["Authorization"] = auth_header
    request = urllib.request.Request(
        server_url.rstrip("/") + "/mirror",
        data=json.dumps({"url": source_url}, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    with opener(request) as response:
        return _json_response(response)


def upload_bundle_to_servers(
    manifest: dict[str, Any],
    bundle_root: str | Path,
    server_urls: Iterable[str],
    *,
    authorization: str | None = None,
    authorizations: dict[str, str] | None = None,
    opener=urllib.request.urlopen,
) -> dict[str, list[dict[str, Any]]]:
    """Publish every verified bundle file to multiple Blossom servers.

    Results are grouped by server. A server is only reported as successful
    after all files have uploaded and returned hash-consistent descriptors;
    failures on one server do not prevent attempts on the remaining mirrors.
    """

    validate_manifest(manifest)
    bundle_root = Path(bundle_root).resolve()
    verify_bundle(manifest, bundle_root)
    servers = list(dict.fromkeys(server_urls))
    if not servers:
        raise ValueError("at least one Blossom server is required")
    if authorization is not None and authorizations is not None:
        raise ValueError("provide either authorization or authorizations, not both")
    authorizations = authorizations or {}
    results: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    for server in servers:
        descriptors: list[dict[str, Any]] = []
        try:
            for entry in manifest["files"]:
                content_type = "application/octet-stream"
                file_authorization = authorizations.get(entry["sha256"], authorization)
                descriptors.append(upload_blob(server, bundle_root / entry["path"], content_type=content_type, authorization=file_authorization, opener=opener))
            results[server] = descriptors
        except (OSError, RuntimeError, ValueError) as exc:
            failures[server] = str(exc)
    if failures:
        results["_failures"] = [{"server": server, "error": error} for server, error in failures.items()]
    if not results or all(key == "_failures" for key in results):
        raise RuntimeError("all Blossom bundle uploads failed: " + "; ".join(f"{server}: {error}" for server, error in failures.items()))
    return results


def check_blossom_server(
    server_url: str,
    *,
    timeout: int = 10,
    opener=urllib.request.urlopen,
) -> dict[str, Any]:
    """Probe a Blossom server origin and normalize its HTTP health result."""

    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Blossom server URL must be an HTTP(S) origin/path without query or fragment")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be positive")
    request = urllib.request.Request(server_url.rstrip("/") + "/", method="HEAD")
    try:
        with opener(request, timeout=timeout) as response:
            status_value = getattr(response, "status", None)
            status = int(status_value if status_value is not None else response.getcode())
            return {"server": server_url.rstrip("/"), "healthy": 200 <= status < 500, "status": status}
    except urllib.error.HTTPError as exc:
        return {"server": server_url.rstrip("/"), "healthy": 400 <= exc.code < 500, "status": exc.code, "error": str(exc.reason)}
    except (OSError, TimeoutError) as exc:
        return {"server": server_url.rstrip("/"), "healthy": False, "status": None, "error": str(exc)}


def check_blossom_servers(
    server_urls: Iterable[str],
    *,
    timeout: int = 10,
    opener=urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Probe unique Blossom servers while preserving configured order."""

    servers = list(dict.fromkeys(server_urls))
    if not servers:
        raise ValueError("at least one Blossom server is required")
    return [check_blossom_server(server, timeout=timeout, opener=opener) for server in servers]


def install_from_blossom(
    manifest: dict[str, Any],
    install_root: str | Path,
    *,
    opener=urllib.request.urlopen,
) -> Path:
    """Download and atomically install a complete adapter from Blossom mirrors."""

    validate_manifest(manifest)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", manifest["name"]):
        raise ValueError("adapter name contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", manifest["version"]):
        raise ValueError("adapter version contains unsupported characters")
    install_root = Path(install_root).resolve()
    install_root.mkdir(parents=True, exist_ok=True)
    target = install_root / f"{manifest['name']}-{manifest['version']}"
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(dir=install_root, prefix=".adapter-install-"))
    try:
        for entry in manifest["files"]:
            urls = file_blossom_urls(manifest, entry)
            if not urls:
                raise ValueError(f"no Blossom source for {entry['path']}")
            download_from_mirrors(urls, temporary / entry["path"], entry["sha256"], opener=opener)
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        verify_bundle(manifest, temporary)
        temporary.replace(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_libtorrent():
    try:
        import libtorrent
    except ImportError as exc:
        raise TorrentUnavailableError("BitTorrent support requires the optional 'libtorrent' package") from exc
    return libtorrent


def torrent_available() -> bool:
    """Return whether the optional BitTorrent backend can be imported."""

    try:
        _load_libtorrent()
    except TorrentUnavailableError:
        return False
    return True


def create_torrent(
    bundle_dir: str | Path,
    torrent_path: str | Path,
    *,
    trackers: Iterable[str] = (),
) -> Path:
    """Create a .torrent for a verified bundle using optional libtorrent."""

    libtorrent = _load_libtorrent()
    bundle_dir = Path(bundle_dir).resolve()
    torrent_path = Path(torrent_path)
    if not bundle_dir.is_dir():
        raise NotADirectoryError(bundle_dir)
    if bundle_dir in torrent_path.parents:
        raise ValueError("torrent file must be outside the bundle directory")
    storage = libtorrent.file_storage()
    libtorrent.add_files(storage, str(bundle_dir))
    creator = libtorrent.create_torrent(storage)
    for tracker in trackers:
        if not isinstance(tracker, str) or not tracker.strip():
            raise ValueError("trackers must be non-empty strings")
        creator.add_tracker(tracker.strip())
    # add_files() records paths below the bundle directory, including the
    # directory name; hashing therefore starts at its parent.
    libtorrent.set_piece_hashes(creator, str(bundle_dir.parent))
    torrent_path.parent.mkdir(parents=True, exist_ok=True)
    torrent_path.write_bytes(libtorrent.bencode(creator.generate()))
    return torrent_path


def download_torrent(
    magnet: str,
    destination_dir: str | Path,
    manifest: dict[str, Any],
    *,
    progress=None,
    status_callback=None,
    timeout: int = 3600,
) -> Path:
    """Download a magnet and verify its resulting adapter bundle.

    The optional backend is deliberately isolated here.  ``progress`` receives
    a float from 0.0 to 1.0 when the backend reports progress.
    """

    if _source_url(magnet) != magnet:
        raise ValueError("magnet must be a valid BitTorrent magnet URI")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be positive")
    libtorrent = _load_libtorrent()
    destination_dir = Path(destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    session = libtorrent.session()
    handle = libtorrent.add_magnet_uri(session, magnet, {"save_path": str(destination_dir)})
    deadline = time.monotonic() + timeout
    while not handle.has_metadata():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for torrent metadata")
        if status_callback is not None:
            status_callback(_torrent_status(handle.status(), seeding=False))
        time.sleep(0.25)
    while not handle.is_seed():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out downloading torrent")
        status = handle.status()
        if progress is not None:
            progress(max(0.0, min(1.0, float(status.progress))))
        if status_callback is not None:
            status_callback(_torrent_status(status, seeding=False))
        time.sleep(0.25)
    if progress is not None:
        progress(1.0)
    if status_callback is not None:
        status_callback(_torrent_status(handle.status(), seeding=True))
    bundle_roots = [destination_dir] if (destination_dir / "manifest.json").is_file() else list(destination_dir.glob("*/manifest.json"))
    if len(bundle_roots) != 1:
        raise ValueError("torrent download did not produce exactly one adapter bundle")
    bundle_root = bundle_roots[0].parent
    verify_bundle(manifest, bundle_root)
    return bundle_root


def _torrent_status(status: Any, *, seeding: bool) -> dict[str, Any]:
    """Normalize a libtorrent status object for UI/API consumers."""

    progress = max(0.0, min(1.0, float(getattr(status, "progress", 0.0))))
    return {
        "progress": progress,
        "peers": int(getattr(status, "num_peers", 0)),
        "download_rate": int(getattr(status, "download_rate", 0)),
        "upload_rate": int(getattr(status, "upload_rate", 0)),
        "state": str(getattr(status, "state", "downloading")),
        "seeding": bool(seeding),
    }
