import hashlib
import json

import pytest

from studio import app as studio_app


def test_package_adapter_endpoint_creates_bundle(tmp_path, monkeypatch):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    models = tmp_path / "models"
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", models)
    request = studio_app.PackageAdapterRequest(
        source=str(source),
        name="grounding",
        version="1.0.0",
        base_model="base",
        license="MIT",
        files=["adapter_model.safetensors"],
        nostr_pubkey="1" * 64,
        created_at=123,
        training={"method": "lora", "rank": 16, "dataset": "curated-v1"},
        evaluations=[{"name": "heldout", "dataset": "corpus-v1", "score": 0.82}],
    )
    result = studio_app.package_adapter(request)
    assert result["manifest"]["name"] == "grounding"
    assert result["event"]["kind"] == 30078
    assert result["manifest"]["training"]["rank"] == 16
    assert result["manifest"]["evaluations"][0]["score"] == 0.82
    assert (models / "shared_adapters" / "grounding-1.0.0" / "manifest.json").exists()


def test_package_adapter_endpoint_rejects_source_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    request = studio_app.PackageAdapterRequest(
        source="/tmp/outside",
        name="grounding",
        version="1.0.0",
        base_model="base",
        license="MIT",
    )
    with pytest.raises(studio_app.HTTPException, match="inside the project"):
        studio_app.package_adapter(request)


def test_list_local_adapters_returns_only_valid_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    bundle = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "hypotaxis.adapter.v1",
                "name": "grounding",
                "version": "1.0.0",
                "base_model": "base",
                "license": "MIT",
                "files": [{"path": "adapter_model.safetensors", "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )
    invalid = tmp_path / "models" / "shared_adapters" / "invalid"
    invalid.mkdir()
    (invalid / "manifest.json").write_text("not json", encoding="utf-8")
    result = studio_app.list_local_adapters()
    assert len(result["adapters"]) == 1
    assert result["adapters"][0]["manifest_sha256"]
    assert result["adapters"][0]["torrent_exists"] is False


def test_remove_local_adapter_deletes_bundle_and_torrent(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    bundle = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    torrent = bundle.parent / "grounding-1.0.0.torrent"
    torrent.write_bytes(b"torrent")
    result = studio_app.remove_local_adapter("grounding", "1.0.0")
    assert result == {"removed": True, "name": "grounding", "version": "1.0.0"}
    assert not bundle.exists()
    assert not torrent.exists()


def test_create_adapter_torrent_endpoint_reports_created(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    bundle = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    torrent = bundle.parent / "grounding-1.0.0.torrent"
    monkeypatch.setattr(studio_app, "torrent_available", lambda: True)
    monkeypatch.setattr(studio_app, "create_torrent", lambda source, target, trackers: target.write_bytes(b"torrent"))
    result = studio_app.create_adapter_torrent(studio_app.CreateTorrentRequest(name="grounding", version="1.0.0"))
    assert result["status"] == "created"
    assert result["torrent_path"].endswith("grounding-1.0.0.torrent")
    assert torrent.read_bytes() == b"torrent"


def test_discover_adapters_returns_valid_release_metadata(monkeypatch):
    manifest = {
        "schema": "hypotaxis.adapter.v1",
        "name": "grounding",
        "version": "1.0.0",
        "base_model": "base",
        "license": "MIT",
        "files": [{"path": "adapter_model.safetensors", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    monkeypatch.setattr(studio_app, "query_nostr_relays", lambda relays, filters, max_events: [event])
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    result = studio_app.discover_adapters(studio_app.DiscoverAdaptersRequest(relays=["wss://relay.example"]))
    assert result["releases"][0]["manifest"]["name"] == "grounding"
    assert result["releases"][0]["signature_verified"] is False


def test_discover_adapters_requires_relay(monkeypatch):
    with pytest.raises(studio_app.HTTPException, match="relay URL"):
        studio_app.discover_adapters(studio_app.DiscoverAdaptersRequest(relays=[]))


def test_discover_adapters_returns_newest_parameterized_release(monkeypatch):
    from manga_pipeline.adapter_distribution import build_release_event

    manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    older = build_release_event(manifest, "1" * 64, 123)
    older["sig"] = "2" * 128
    newer_manifest = {**manifest, "version": "1.1.0"}
    newer = build_release_event(newer_manifest, "1" * 64, 124)
    newer["sig"] = "2" * 128
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    monkeypatch.setattr(studio_app, "query_nostr_relays", lambda *_args, **_kwargs: [older, newer])
    result = studio_app.discover_adapters(studio_app.DiscoverAdaptersRequest(relays=["wss://relay.example"]))
    assert len(result["releases"]) == 1
    assert result["releases"][0]["manifest"]["version"] == "1.1.0"


def test_install_adapter_endpoint_returns_installed_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    target = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda manifest, root: target)
    event_manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(event_manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    result = studio_app.install_adapter(studio_app.InstallAdapterRequest(manifest=event_manifest, release_event=event))
    assert result["installed"] is True


def test_install_adapter_rejects_manifest_event_mismatch(monkeypatch):
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda *_args: pytest.fail("install should not start"))
    event_manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(event_manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    requested_manifest = {**event_manifest, "version": "2.0.0"}
    with pytest.raises(studio_app.HTTPException, match="does not match"):
        studio_app.install_adapter(studio_app.InstallAdapterRequest(manifest=requested_manifest, release_event=event))


def test_install_adapter_rejects_invalid_signature(monkeypatch):
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: True)
    monkeypatch.setattr(studio_app, "verify_schnorr_signature", lambda _event: False)
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda *_args: pytest.fail("install should not start"))
    manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    with pytest.raises(studio_app.HTTPException, match="signature is invalid"):
        studio_app.install_adapter(studio_app.InstallAdapterRequest(manifest=manifest, release_event=event))


def test_install_composition_installs_verified_components(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "style", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_composition, build_release_event, manifest_digest

    event = build_release_event(manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    composition = build_composition(
        "community", "1.0.0", "base",
        [{"name": "style", "version": "1.0.0", "manifest_sha256": manifest_digest(manifest), "weight": 1.0}],
    )
    installed = tmp_path / "models" / "shared_adapters" / "style-1.0.0"
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda _manifest, _root: installed)
    result = studio_app.install_composition(studio_app.InstallCompositionRequest(composition=composition, release_events=[event]))
    assert result == {"installed": ["models/shared_adapters/style-1.0.0"], "composition": "community"}


def test_install_composition_rolls_back_components_after_later_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    from manga_pipeline.adapter_distribution import build_composition, build_release_event, manifest_digest

    manifests = []
    events = []
    for name in ("style", "format"):
        manifest = {
            "schema": "hypotaxis.adapter.v1", "name": name, "version": "1.0.0", "base_model": "base", "license": "MIT",
            "files": [{"path": "x.bin", "sha256": "a" * 64}],
        }
        manifests.append(manifest)
        event = build_release_event(manifest, "1" * 64, 123)
        event["sig"] = "2" * 128
        events.append(event)
    composition = build_composition("community", "1.0.0", "base", [
        {"name": m["name"], "version": m["version"], "manifest_sha256": manifest_digest(m), "weight": 1.0} for m in manifests
    ])
    first = tmp_path / "models" / "shared_adapters" / "style-1.0.0"
    calls = 0

    def install(_manifest, _root):
        nonlocal calls
        calls += 1
        if calls == 1:
            first.mkdir(parents=True)
            return first
        raise OSError("mirror unavailable")

    monkeypatch.setattr(studio_app, "install_from_blossom", install)
    with pytest.raises(studio_app.HTTPException, match="mirror unavailable"):
        studio_app.install_composition(studio_app.InstallCompositionRequest(composition=composition, release_events=events))
    assert not first.exists()


def test_upload_adapter_endpoint_verifies_bundle_and_uploads_each_file(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    bundle = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    bundle.mkdir(parents=True)
    artifact = bundle / "adapter_model.safetensors"
    artifact.write_bytes(b"weights")
    from manga_pipeline.adapter_distribution import build_manifest, write_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / artifact.name).write_bytes(b"weights")
    manifest = build_manifest(source, name="grounding", version="1.0.0", base_model="base", license="MIT")
    write_bundle(source, bundle, manifest)
    calls = []
    monkeypatch.setattr(
        studio_app,
        "upload_bundle_to_servers",
        lambda manifest, bundle, servers, authorization, authorizations: calls.append((manifest, bundle, servers, authorization, authorizations))
        or {servers[0]: [{"sha256": manifest["files"][0]["sha256"], "url": "https://cdn.example/blob"}]},
    )
    result = studio_app.upload_adapter(
        studio_app.UploadAdapterRequest(
            name="grounding", version="1.0.0", server_urls=["https://blossom.example"], authorization="Nostr token"
        )
    )
    assert result["uploaded"] is True
    assert result["manifest_sha256"]
    assert calls[0][2:] == (["https://blossom.example"], "Nostr token", None)


def test_mirror_adapter_blob_endpoint_returns_descriptor(monkeypatch):
    monkeypatch.setattr(
        studio_app,
        "mirror_blob",
        lambda server, source, authorization: {"url": server + "/" + "a" * 64, "sha256": "a" * 64},
    )
    result = studio_app.mirror_adapter_blob(
        studio_app.MirrorBlobRequest(
            server_url="https://blossom.example",
            source_url="https://origin.example/" + "a" * 64,
            authorization="Nostr token",
        )
    )
    assert result["mirrored"] is True
    assert result["descriptor"]["sha256"] == "a" * 64


def test_check_adapter_blossom_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        studio_app,
        "check_blossom_servers",
        lambda servers, timeout: [{"server": servers[0], "healthy": True, "status": 200}],
    )
    result = studio_app.check_adapter_blossom_health(
        studio_app.BlossomHealthRequest(server_urls=["https://blossom.example"], timeout=3)
    )
    assert result["servers"][0]["healthy"] is True


def test_start_torrent_download_returns_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "schnorr_available", lambda: False)
    started = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(studio_app.threading, "Thread", FakeThread)
    manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    result = studio_app.start_torrent_download(
        studio_app.DownloadTorrentRequest(
            magnet="magnet:?xt=urn:btih:abc",
            manifest=manifest,
            release_event=event,
        )
    )
    assert result["job_id"]
    assert len(started) == 1
    assert started[0][1][0] == result["job_id"]


def test_start_torrent_download_rejects_untrusted_release(monkeypatch):
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda *_args: pytest.fail("not relevant"))
    manifest = {
        "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0", "base_model": "base", "license": "MIT",
        "files": [{"path": "x.bin", "sha256": "a" * 64}],
    }
    from manga_pipeline.adapter_distribution import build_release_event

    event = build_release_event(manifest, "1" * 64, 123)
    event["sig"] = "2" * 128
    requested_manifest = {**manifest, "version": "2.0.0"}
    with pytest.raises(studio_app.HTTPException, match="does not match"):
        studio_app.start_torrent_download(
            studio_app.DownloadTorrentRequest(
                magnet="magnet:?xt=urn:btih:abc", manifest=requested_manifest, release_event=event
            )
        )


def test_create_adapter_composition_preserves_lineage_and_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    from manga_pipeline.adapter_distribution import build_manifest, write_bundle

    source = tmp_path / "source"
    source.mkdir()
    for name in ("one", "two"):
        adapter_source = source / name
        adapter_source.mkdir()
        (adapter_source / "adapter_model.safetensors").write_bytes(name.encode())
        manifest = build_manifest(adapter_source, name=name, version="1.0.0", base_model="base", license="MIT")
        write_bundle(adapter_source, tmp_path / "models" / "shared_adapters" / f"{name}-1.0.0", manifest)
    result = studio_app.create_adapter_composition(
        studio_app.CreateCompositionRequest(
            name="combined",
            version="1.0.0",
            base_model="base",
            components=[
                studio_app.CompositionComponentRequest(name="one", version="1.0.0", weight=0.7),
                studio_app.CompositionComponentRequest(name="two", version="1.0.0", weight=1.2),
            ],
        )
    )
    assert result["composition"]["components"][0]["weight"] == 0.7
    assert result["composition"]["components"][1]["manifest_sha256"]
    assert (tmp_path / result["path"]).exists()


def test_create_adapter_composition_rejects_incompatible_base_models(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    from manga_pipeline.adapter_distribution import build_manifest, write_bundle

    source = tmp_path / "source"
    source.mkdir()
    for name, base in (("one", "base-a"), ("two", "base-b")):
        adapter_source = source / name
        adapter_source.mkdir()
        (adapter_source / "adapter_model.safetensors").write_bytes(name.encode())
        manifest = build_manifest(adapter_source, name=name, version="1.0.0", base_model=base, license="MIT")
        write_bundle(adapter_source, tmp_path / "models" / "shared_adapters" / f"{name}-1.0.0", manifest)
    with pytest.raises(studio_app.HTTPException, match="incompatible"):
        studio_app.create_adapter_composition(
            studio_app.CreateCompositionRequest(
                name="combined",
                version="1.0.0",
                base_model="base-a",
                components=[
                    studio_app.CompositionComponentRequest(name="one", version="1.0.0"),
                    studio_app.CompositionComponentRequest(name="two", version="1.0.0"),
                ],
            )
        )


def test_list_adapter_compositions_returns_only_valid_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    root = tmp_path / "models" / "shared_adapters" / "compositions"
    root.mkdir(parents=True)
    (root / "combined-1.0.0.json").write_text(
        json.dumps(
            {
                "schema": "hypotaxis.adapter-composition.v1",
                "name": "combined",
                "version": "1.0.0",
                "base_model": "base",
                "components": [{"name": "one", "version": "1.0.0", "manifest_sha256": "a" * 64, "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    (root / "invalid.json").write_text("{}", encoding="utf-8")
    result = studio_app.list_adapter_compositions()
    assert result["compositions"] == [
        {"name": "combined", "version": "1.0.0", "base_model": "base", "component_count": 1, "evaluation_count": 0, "community_merge": False, "composition": {"schema": "hypotaxis.adapter-composition.v1", "name": "combined", "version": "1.0.0", "base_model": "base", "components": [{"name": "one", "version": "1.0.0", "manifest_sha256": "a" * 64, "weight": 1.0}]}, "path": "models/shared_adapters/compositions/combined-1.0.0.json"}
    ]


def test_generate_rejects_composition_outside_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "_story_or_404", lambda story_id: object())
    with pytest.raises(studio_app.HTTPException, match="inside the project"):
        studio_app.generate(
            "story",
            studio_app.GenerateRequest(adapter_composition_path=str(tmp_path / "outside.json")),
        )


def test_seed_adapter_endpoint_starts_and_reports_opt_in_seeder(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    bundle = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps({
            "schema": "hypotaxis.adapter.v1", "name": "grounding", "version": "1.0.0",
            "base_model": "base", "license": "MIT", "files": [{"path": "x.bin", "sha256": hashlib.sha256(b"weights").hexdigest()}],
            "distribution": {"torrent": {"magnet": "magnet:?xt=urn:btih:abc"}},
        }), encoding="utf-8"
    )
    (bundle / "x.bin").write_bytes(b"weights")

    class Seeder:
        def start(self, seed_id, bundle_dir, **kwargs):
            return {"seed_id": seed_id, "seeding": True, "peers": 0}

        def status(self, seed_id):
            return {"seed_id": seed_id, "seeding": True}

        def stop(self, seed_id):
            return None

    studio_app._torrent_seeder = Seeder()
    monkeypatch.setattr(studio_app, "torrent_available", lambda: True)
    result = studio_app.seed_adapter(studio_app.SeedAdapterRequest(name="grounding", version="1.0.0"))
    assert result["seeding"] is True
    assert studio_app.adapter_seed_status("grounding", "1.0.0")["seeding"] is True
