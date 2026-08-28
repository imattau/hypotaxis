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
    )
    result = studio_app.package_adapter(request)
    assert result["manifest"]["name"] == "grounding"
    assert result["event"]["kind"] == 30078
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


def test_install_adapter_endpoint_returns_installed_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "ROOT", tmp_path)
    target = tmp_path / "models" / "shared_adapters" / "grounding-1.0.0"
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(studio_app, "install_from_blossom", lambda manifest, root: target)
    manifest = {"name": "grounding", "version": "1.0.0"}
    result = studio_app.install_adapter(studio_app.InstallAdapterRequest(manifest=manifest))
    assert result["installed"] is True


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
        lambda manifest, bundle, servers, authorization: calls.append((manifest, bundle, servers, authorization))
        or {servers[0]: [{"sha256": manifest["files"][0]["sha256"], "url": "https://cdn.example/blob"}]},
    )
    result = studio_app.upload_adapter(
        studio_app.UploadAdapterRequest(
            name="grounding", version="1.0.0", server_urls=["https://blossom.example"], authorization="Nostr token"
        )
    )
    assert result["uploaded"] is True
    assert result["manifest_sha256"]
    assert calls[0][2:] == (["https://blossom.example"], "Nostr token")


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


def test_start_torrent_download_returns_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_app, "MODELS_DIR", tmp_path / "models")
    started = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(studio_app.threading, "Thread", FakeThread)
    result = studio_app.start_torrent_download(
        studio_app.DownloadTorrentRequest(
            magnet="magnet:?xt=urn:btih:abc",
            manifest={"name": "grounding", "version": "1.0.0"},
        )
    )
    assert result["job_id"]
    assert len(started) == 1
    assert started[0][1][0] == result["job_id"]


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
        {"name": "combined", "version": "1.0.0", "base_model": "base", "component_count": 1, "path": "models/shared_adapters/compositions/combined-1.0.0.json"}
    ]
