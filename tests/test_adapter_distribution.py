import hashlib
import json

import pytest

from manga_pipeline.adapter_distribution import (
    BLOSSOM_SERVER_LIST_KIND,
    NOSTR_RELEASE_KIND,
    available_sources,
    blossom_authorization_header,
    blossom_blob_urls,
    blossom_servers,
    build_manifest,
    build_composition,
    build_release_event,
    compatible_manifests,
    distribution_sources,
    download_verified_blob,
    download_from_mirrors,
    file_blossom_urls,
    install_from_blossom,
    mirror_blob,
    upload_blob,
    TorrentUnavailableError,
    create_torrent,
    download_torrent,
    torrent_available,
    manifest_digest,
    nostr_event_id,
    parse_release_event,
    query_nostr_relays,
    schnorr_available,
    verify_schnorr_signature,
    validate_composition,
    validate_signed_event,
    verify_bundle,
    write_bundle,
    _torrent_status,
)


def _manifest():
    return {
        "schema": "hypotaxis.adapter.v1",
        "name": "grounded-captioner",
        "version": "1.0.0",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "license": "CC-BY-4.0",
        "files": [{"path": "adapter_model.safetensors", "sha256": "a" * 64}],
        "distribution": {
            "blossom": ["https://mirror-a.example/" + "b" * 64, "https://mirror-a.example/" + "b" * 64],
            "torrent": {"magnet": "magnet:?xt=urn:btih:abc"},
        },
    }


def test_release_event_is_deterministic_and_replaceable():
    event = build_release_event(_manifest(), "1" * 64, 123)
    assert event["kind"] == NOSTR_RELEASE_KIND
    assert event["tags"][0] == ["d", "adapter:grounded-captioner"]
    assert event["id"] == nostr_event_id(event)
    assert event["content"].endswith("\n")


def test_release_event_rejects_bad_pubkey():
    with pytest.raises(ValueError, match="pubkey"):
        build_release_event(_manifest(), "not-a-key", 123)


def test_signed_release_event_round_trips_after_signature_is_attached():
    event = build_release_event(_manifest(), "1" * 64, 123)
    event["sig"] = "2" * 128
    validate_signed_event(event)
    assert parse_release_event(event)["name"] == "grounded-captioner"


def test_signed_event_rejects_id_tampering():
    event = build_release_event(_manifest(), "1" * 64, 123)
    event["sig"] = "2" * 128
    event["content"] = "tampered"
    with pytest.raises(ValueError, match="id"):
        validate_signed_event(event)


@pytest.mark.skipif(not schnorr_available(), reason="optional coincurve is not installed")
def test_schnorr_signature_verification_accepts_valid_event_and_rejects_tampering():
    from coincurve import PrivateKey

    private_key = PrivateKey.from_int(7)
    event = build_release_event(_manifest(), private_key.public_key_xonly.format().hex(), 123)
    event["sig"] = private_key.sign_schnorr(bytes.fromhex(event["id"])).hex()
    assert verify_schnorr_signature(event) is True
    event["content"] = "tampered"
    assert verify_schnorr_signature(event) is False


class _FakeRelay:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        try:
            return next(self.messages)
        except StopIteration:
            subscription_id = json.loads(self.sent[0])[1]
            return json.dumps(["EOSE", subscription_id])

    def close(self):
        self.closed = True


def test_query_nostr_relays_deduplicates_and_ignores_invalid_events():
    event = build_release_event(_manifest(), "1" * 64, 123)
    event["sig"] = "2" * 128
    invalid = dict(event, id="f" * 64)
    connection = _FakeRelay(
        [
            json.dumps(["EVENT", "subscription", event]),
            json.dumps(["EVENT", "subscription", invalid]),
            json.dumps(["EVENT", "subscription", event]),
            json.dumps(["EOSE", "subscription"]),
        ]
    )
    result = query_nostr_relays(
        ["wss://relay.example"],
        [{"kinds": [NOSTR_RELEASE_KIND]}],
        connector=lambda url, timeout: connection,
    )
    assert result == [event]
    assert connection.closed is True
    assert json.loads(connection.sent[0])[0:2] == ["REQ", json.loads(connection.sent[0])[1]]


def test_query_nostr_relays_skips_bad_relay_url():
    assert query_nostr_relays(["https://not-a-websocket-relay"], [{"kinds": [NOSTR_RELEASE_KIND]}], connector=lambda *_: None) == []


def test_blossom_servers_and_blob_urls_are_deduplicated():
    event = {
        "pubkey": "1" * 64,
        "created_at": 123,
        "kind": BLOSSOM_SERVER_LIST_KIND,
        "tags": [
            ["server", "https://one.example/"],
            ["server", "https://one.example"],
            ["server", "https://two.example/blobs"],
            ["other", "ignored"],
        ],
        "content": "",
    }
    event["id"] = nostr_event_id(event)
    event["sig"] = "2" * 128
    servers = blossom_servers(event)
    assert servers == ["https://one.example", "https://two.example/blobs"]
    assert blossom_blob_urls("a" * 64, servers) == [
        "https://one.example/" + "a" * 64,
        "https://two.example/blobs/" + "a" * 64,
    ]


def test_blossom_blob_urls_reject_query_strings():
    with pytest.raises(ValueError, match="without query"):
        blossom_blob_urls("a" * 64, ["https://one.example?token=secret"])


def test_blossom_authorization_header_uses_unpadded_base64url():
    event = {
        "pubkey": "1" * 64,
        "created_at": 123,
        "kind": 24242,
        "tags": [["t", "upload"], ["expiration", "9999999999"]],
        "content": "Upload adapter",
    }
    event["id"] = nostr_event_id(event)
    event["sig"] = "2" * 128
    header = blossom_authorization_header(event)
    assert header.startswith("Nostr ")
    assert "=" not in header


class _JsonResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_upload_blob_uses_bud02_headers_and_authorization(tmp_path):
    blob = tmp_path / "adapter.safetensors"
    blob.write_bytes(b"weights")
    requests = []
    event = {
        "pubkey": "1" * 64,
        "created_at": 123,
        "kind": 24242,
        "tags": [["t", "upload"], ["expiration", "9999999999"]],
        "content": "Upload adapter",
    }
    event["id"] = nostr_event_id(event)
    event["sig"] = "2" * 128

    def opener(request):
        body = request.data.read()
        requests.append((request, body))
        return _JsonResponse({"sha256": hashlib.sha256(b"weights").hexdigest(), "url": "https://cdn.example/blob"})

    descriptor = upload_blob("https://blossom.example", blob, authorization_event=event, opener=opener)
    assert descriptor["url"] == "https://cdn.example/blob"
    request, body = requests[0]
    assert request.method == "PUT"
    assert request.full_url == "https://blossom.example/upload"
    assert request.headers["X-sha-256"] == hashlib.sha256(b"weights").hexdigest()
    assert request.headers["Authorization"].startswith("Nostr ")
    assert body == b"weights"


def test_mirror_blob_sends_bud04_json_and_header():
    requests = []

    def opener(request):
        requests.append(request)
        return _JsonResponse({"sha256": "a" * 64, "url": "https://mirror.example/" + "a" * 64})

    descriptor = mirror_blob(
        "https://mirror.example",
        "https://origin.example/" + "a" * 64 + ".safetensors",
        authorization="Nostr signed-token",
        opener=opener,
    )
    assert descriptor["sha256"] == "a" * 64
    assert requests[0].full_url == "https://mirror.example/mirror"
    assert json.loads(requests[0].data) == {"url": "https://origin.example/" + "a" * 64 + ".safetensors"}
    assert requests[0].headers["Authorization"] == "Nostr signed-token"


class _FakeResponse:
    def __init__(self, payload):
        self.headers = {"Content-Length": str(len(payload))}
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        payload, self._payload = self._payload[:size], self._payload[size:]
        return payload


def test_download_verified_blob_streams_and_replaces_atomically(tmp_path):
    payload = b"weights"
    destination = tmp_path / "adapter_model.safetensors"
    result = download_verified_blob(
        "https://mirror.example/" + "a" * 64,
        destination,
        hashlib.sha256(payload).hexdigest(),
        opener=lambda url: _FakeResponse(payload),
    )
    assert result == destination
    assert destination.read_bytes() == payload
    assert list(tmp_path.glob(".*")) == []


def test_download_verified_blob_rejects_bad_hash_and_size(tmp_path):
    destination = tmp_path / "adapter_model.safetensors"
    with pytest.raises(ValueError, match="does not match"):
        download_verified_blob(
            "https://mirror.example/blob",
            destination,
            "a" * 64,
            opener=lambda url: _FakeResponse(b"tampered"),
        )
    with pytest.raises(ValueError, match="maximum size"):
        download_verified_blob(
            "https://mirror.example/blob",
            destination,
            "a" * 64,
            max_bytes=3,
            opener=lambda url: _FakeResponse(b"too large"),
        )
    assert not destination.exists()


def test_download_from_mirrors_falls_back_after_failed_mirror(tmp_path):
    payload = b"weights"
    destination = tmp_path / "adapter_model.safetensors"
    good_url = "https://good.example/blob"

    def opener(url):
        if url == "https://bad.example/blob":
            raise OSError("connection refused")
        return _FakeResponse(payload)

    assert download_from_mirrors(
        ["https://bad.example/blob", good_url, good_url],
        destination,
        hashlib.sha256(payload).hexdigest(),
        opener=opener,
    ) == destination
    assert destination.read_bytes() == payload


def test_download_from_mirrors_reports_all_failures(tmp_path):
    destination = tmp_path / "adapter_model.safetensors"

    def opener(url):
        raise OSError("offline")

    with pytest.raises(RuntimeError, match="all adapter mirrors failed") as error:
        download_from_mirrors(
            ["https://one.example/blob", "https://two.example/blob"],
            destination,
            "a" * 64,
            opener=opener,
        )
    assert "one.example" in str(error.value)
    assert "two.example" in str(error.value)
    assert not destination.exists()


def test_file_blossom_urls_support_server_roots_and_exact_blob_urls():
    manifest = _manifest()
    entry = manifest["files"][0]
    entry["sha256"] = "c" * 64
    manifest["distribution"]["blossom"] = [
        "https://server.example/blobs",
        "https://server.example/" + "c" * 64,
        "https://server.example/" + "d" * 64,
    ]
    assert file_blossom_urls(manifest, entry) == [
        "https://server.example/blobs/" + "c" * 64,
        "https://server.example/" + "c" * 64,
    ]


def test_install_from_blossom_is_atomic_and_verified(tmp_path):
    payload = b"weights"
    manifest = _manifest()
    manifest["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["distribution"]["blossom"] = ["https://server.example/blobs"]

    def opener(url):
        assert url.endswith(manifest["files"][0]["sha256"])
        return _FakeResponse(payload)

    target = install_from_blossom(manifest, tmp_path / "installed", opener=opener)
    assert (target / "adapter_model.safetensors").read_bytes() == payload
    assert (target / "manifest.json").exists()
    assert list((tmp_path / "installed").glob(".adapter-install-*")) == []


def test_install_from_blossom_does_not_leave_partial_bundle(tmp_path):
    manifest = _manifest()
    manifest["distribution"]["blossom"] = []
    with pytest.raises(ValueError, match="no Blossom source"):
        install_from_blossom(manifest, tmp_path / "installed")
    assert list((tmp_path / "installed").glob(".adapter-install-*")) == []


def test_torrent_backend_is_optional_when_libtorrent_is_missing(tmp_path):
    if torrent_available():
        pytest.skip("libtorrent is installed in this environment")
    assert torrent_available() is False
    with pytest.raises(TorrentUnavailableError, match="libtorrent"):
        create_torrent(tmp_path, tmp_path / "adapter.torrent")
    with pytest.raises(TorrentUnavailableError, match="libtorrent"):
        download_torrent("magnet:?xt=urn:btih:abc", tmp_path, _manifest())


@pytest.mark.skipif(not torrent_available(), reason="optional libtorrent is not installed")
def test_create_torrent_generates_metadata_for_bundle(tmp_path):
    source = tmp_path / "grounding-1.0.0"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    torrent_path = tmp_path / "grounding-1.0.0.torrent"
    assert create_torrent(source, torrent_path).exists()
    import libtorrent

    info = libtorrent.torrent_info(str(torrent_path))
    assert info.num_files() == 1
    assert info.files().file_path(0).endswith("grounding-1.0.0/adapter_model.safetensors")


def test_sources_prefer_unique_blossom_mirrors_then_torrent():
    manifest = _manifest()
    expected = ["https://mirror-a.example/" + "b" * 64, "magnet:?xt=urn:btih:abc"]
    assert distribution_sources(manifest) == expected
    assert available_sources(manifest, {expected[1]}) == [expected[1]]


def test_verify_bundle_checks_declared_hashes(tmp_path):
    payload = b"adapter payload"
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(payload)
    manifest = _manifest()
    manifest["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    assert verify_bundle(manifest, tmp_path) == ["adapter_model.safetensors"]
    adapter.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_bundle(manifest, tmp_path)


def test_manifest_digest_changes_with_metadata():
    manifest = _manifest()
    first = manifest_digest(manifest)
    manifest["version"] = "1.0.1"
    assert manifest_digest(manifest) != first


def test_composition_preserves_components_and_lineage():
    composition = build_composition(
        "community-captioner",
        "1.0.0",
        "Qwen/Qwen2.5-7B-Instruct",
        [
            {"name": "grounding", "version": "3.0.0", "manifest_sha256": "a" * 64, "weight": 0.8},
            {"name": "format", "version": "2.0.0", "manifest_sha256": "b" * 64, "weight": 0.6},
        ],
        description="Community composition",
    )
    validate_composition(composition)
    assert composition["components"][1]["manifest_sha256"] == "b" * 64


def test_composition_rejects_duplicate_components_and_bad_weights():
    component = {"name": "grounding", "version": "1", "manifest_sha256": "a" * 64, "weight": 0.8}
    with pytest.raises(ValueError, match="duplicate"):
        build_composition("x", "1", "base", [component, dict(component)])
    invalid = dict(component, weight=2.1)
    with pytest.raises(ValueError, match="between 0 and 2"):
        build_composition("x", "1", "base", [invalid])


def test_compatible_manifests_requires_one_base_model():
    first = _manifest()
    second = _manifest()
    second["name"] = "other"
    second["base_model"] = "different-base"
    with pytest.raises(ValueError, match="incompatible"):
        compatible_manifests([first, second])
    assert compatible_manifests([first]) == "Qwen/Qwen2.5-7B-Instruct"


def test_blossom_kind_is_exposed_for_future_server_discovery():
    assert BLOSSOM_SERVER_LIST_KIND == 10063


def test_build_manifest_and_bundle_only_selected_safe_files(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source / "notes.py").write_text("should not ship", encoding="utf-8")
    manifest = build_manifest(
        source,
        name="grounded-captioner",
        version="1.0.0",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        license="CC-BY-4.0",
        files=["adapter_model.safetensors", "adapter_config.json"],
    )
    bundle = tmp_path / "bundle"
    assert write_bundle(source, bundle, manifest).name == "manifest.json"
    assert (bundle / "adapter_model.safetensors").read_bytes() == b"weights"
    assert not (bundle / "notes.py").exists()
    assert verify_bundle(manifest, bundle) == ["adapter_model.safetensors", "adapter_config.json"]


def test_build_manifest_rejects_executable_artifacts(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    script = source / "install.sh"
    script.write_text("echo unsafe", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported adapter artifact type"):
        build_manifest(
            source,
            name="test",
            version="1.0.0",
            base_model="base",
            license="MIT",
            files=[script.name],
        )


def test_build_manifest_rejects_path_escape(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    with pytest.raises(ValueError, match="escapes source"):
        build_manifest(
            source,
            name="test",
            version="1.0.0",
            base_model="base",
            license="MIT",
            files=["../adapter_model.safetensors"],
        )


def test_torrent_status_normalizes_transfer_metrics():
    class Status:
        progress = 1.4
        num_peers = 3
        download_rate = 1200
        upload_rate = 80
        state = "seeding"

    assert _torrent_status(Status(), seeding=True) == {
        "progress": 1.0,
        "peers": 3,
        "download_rate": 1200,
        "upload_rate": 80,
        "state": "seeding",
        "seeding": True,
    }
