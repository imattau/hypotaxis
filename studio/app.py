from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manga_pipeline.captioner import Captioner  # noqa: E402
from manga_pipeline.adapter_distribution import (  # noqa: E402
    build_manifest,
    build_composition,
    check_blossom_servers,
    compatible_manifests,
    build_release_event,
    create_torrent,
    download_torrent,
    install_from_blossom,
    install_from_torrent,
    manifest_digest,
    mirror_blob,
    parse_release_event,
    parse_composition_event,
    release_is_revoked,
    query_nostr_relays,
    schnorr_available,
    verify_schnorr_signature,
    upload_bundle_to_servers,
    torrent_available,
    TorrentSeeder,
    verify_bundle,
    write_bundle,
)  # noqa: E402
from manga_pipeline.adapter_manifest import canonical_json, load_manifest  # noqa: E402
from manga_pipeline.character_lora import (  # noqa: E402
    TRAINING_VIEW_PROMPTS,
    build_training_caption,
    default_lora_output_dir,
    train_character_lora,
)
from manga_pipeline.backends import DiffusersBackend  # noqa: E402
from manga_pipeline.config import PipelineConfig  # noqa: E402
from manga_pipeline.llm import SmallLLM  # noqa: E402
from manga_pipeline.pipeline import prepare_cast as prepare_cast_pipeline  # noqa: E402
from manga_pipeline.pipeline import run as run_pipeline  # noqa: E402
from manga_pipeline.registry import CharacterRegistry  # noqa: E402
from manga_pipeline.schema import DialogueLine, Panel, Story  # noqa: E402
from manga_pipeline.train_captioner import CAMERA_HINTS  # noqa: E402
from manga_pipeline.story_adapt import (  # noqa: E402
    adapt_story,
    parse_character_profiles,
    parse_location_profiles,
    parse_prop_profiles,
)

STORIES_DIR = ROOT / "stories"
REGISTRY_DIR = ROOT / "registry"
OUTPUT_DIR = ROOT / "output"
DATASET_PATH = ROOT / "data" / "caption_pairs.jsonl"
# canonical location train_captioner.py's CLI defaults write to
# (--output-dir models/captioner) - see README's "Curating a clean caption
# dataset" for how to produce this
CAPTIONER_ADAPTER_DIR = ROOT / "models" / "captioner" / "adapter"
CAPTIONER_BASE_MODEL = "google-t5/t5-base"
MODELS_DIR = ROOT / "models"
SEED_STATE_PATH = MODELS_DIR / "shared_adapters" / ".seeding.json"

@asynccontextmanager
async def lifespan(_app):
    """Restore explicit seed choices and close the session on shutdown."""

    global _torrent_seeder
    if torrent_available():
        _torrent_seeder = TorrentSeeder(persistence_path=SEED_STATE_PATH)
        _torrent_seeder.restore()
    yield
    if _torrent_seeder is not None:
        _torrent_seeder.close()
        _torrent_seeder = None


app = FastAPI(title="Manga Production Studio", lifespan=lifespan)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="output")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 3600
# shared by both adapt (Stage A, small LLM) and generate (Stage B-D, diffusion) -
# on modest hardware these shouldn't run concurrently either, since Stage A's
# LLM and Stage C's diffusion pipeline compete for the same GPU
_gpu_lock = threading.Lock()
_gpu_busy = False
_API_TOKEN = os.environ.get("HYPOTAXIS_API_TOKEN")
_torrent_seeder = None


@app.middleware("http")
async def protect_api_when_configured(request, call_next):
    if _API_TOKEN and request.url.path.startswith("/api/"):
        if request.headers.get("authorization") != f"Bearer {_API_TOKEN}":
            return JSONResponse({"detail": "authentication required"}, status_code=401)
    return await call_next(request)


def _cleanup_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [
            job_id
            for job_id, job in _jobs.items()
            if job.get("finished_at") is not None and job["finished_at"] < cutoff
        ]
        for job_id in stale:
            del _jobs[job_id]


def _create_job() -> str:
    _cleanup_jobs()
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "message": "queued", "created_at": time.time()}
    return job_id


def _finish_job(job: dict, status: str, message: str = "done") -> None:
    job["status"] = status
    job["message"] = message
    job["finished_at"] = time.time()


def _validate_story_id(story_id: str) -> str:
    if len(story_id) > 80 or not re.fullmatch(r"[A-Za-z0-9_-]+", story_id):
        raise HTTPException(404, "story not found")
    return story_id


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _try_claim_gpu() -> bool:
    global _gpu_busy
    with _gpu_lock:
        if _gpu_busy:
            return False
        _gpu_busy = True
        return True


def _release_gpu() -> None:
    global _gpu_busy
    with _gpu_lock:
        _gpu_busy = False


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


_SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".md", ".docx"}


def _extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")
    if suffix == ".docx":
        import io

        from docx import Document

        document = Document(io.BytesIO(content))
        return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    raise HTTPException(400, f"unsupported file type '{suffix}' - use .txt, .md, or .docx")


@app.post("/api/extract-text")
async def extract_text(file: UploadFile):
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(400, "unsupported file type - use .txt, .md, or .docx")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "uploaded file is too large (maximum 25 MiB)")
    try:
        text = _extract_text(filename, content)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read file: {e}")
    return {"text": text}


class PackageAdapterRequest(BaseModel):
    source: str = Field(max_length=500)
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    base_model: str = Field(max_length=300)
    license: str = Field(max_length=100)
    files: list[str] = Field(default_factory=list, max_length=50)
    blossom: list[str] = Field(default_factory=list, max_length=20)
    magnet: str | None = Field(None, max_length=2000)
    nostr_pubkey: str | None = Field(None, max_length=64)
    created_at: int | None = None
    training: dict | None = None
    evaluations: list[dict] | None = None


class CreateTorrentRequest(BaseModel):
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    trackers: list[str] = Field(default_factory=list, max_length=20)


class DiscoverAdaptersRequest(BaseModel):
    relays: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(20, ge=1, le=100)


class InstallAdapterRequest(BaseModel):
    manifest: dict
    release_event: dict


class InstallCompositionRequest(BaseModel):
    composition: dict
    composition_event: dict
    release_events: list[dict]


class UploadAdapterRequest(BaseModel):
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    server_urls: list[str] = Field(min_length=1, max_length=20)
    authorization: str | None = Field(None, max_length=20000)
    authorizations: dict[str, str] | None = None


class MirrorBlobRequest(BaseModel):
    server_url: str = Field(max_length=500)
    source_url: str = Field(max_length=2000)
    authorization: str | None = Field(None, max_length=20000)


class BlossomHealthRequest(BaseModel):
    server_urls: list[str] = Field(min_length=1, max_length=20)
    timeout: int = Field(10, ge=1, le=60)


class DownloadTorrentRequest(BaseModel):
    magnet: str = Field(max_length=4000)
    manifest: dict
    release_event: dict


class SeedAdapterRequest(BaseModel):
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    magnet: str | None = Field(None, max_length=4000)


class CompositionComponentRequest(BaseModel):
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    weight: float = Field(1.0, ge=0, le=2)


class CreateCompositionRequest(BaseModel):
    name: str = Field(max_length=100)
    version: str = Field(max_length=50)
    base_model: str = Field(max_length=300)
    components: list[CompositionComponentRequest] = Field(min_length=1, max_length=20)
    description: str = Field("", max_length=1000)
    evaluations: list[dict] | None = None
    community_merge: bool = False


def _path_under(path_value: str, root: Path) -> Path:
    path = _project_path(path_value).resolve()
    root = root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "path must be inside the project") from exc
    return path


@app.post("/api/adapters/package")
def package_adapter(req: PackageAdapterRequest):
    """Create a verified local adapter bundle for later publishing."""

    source = _path_under(req.source, ROOT)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", req.name):
        raise HTTPException(400, "adapter name contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", req.version):
        raise HTTPException(400, "adapter version contains unsupported characters")
    output = (MODELS_DIR / "shared_adapters" / f"{req.name}-{req.version}").resolve()
    try:
        output.relative_to(MODELS_DIR.resolve())
        distribution: dict = {}
        if req.blossom:
            distribution["blossom"] = req.blossom
        if req.magnet:
            distribution["torrent"] = {"magnet": req.magnet}
        manifest = build_manifest(
            source,
            name=req.name,
            version=req.version,
            base_model=req.base_model,
            license=req.license,
            files=req.files or None,
            distribution=distribution or None,
            training=req.training,
            evaluations=req.evaluations,
        )
        manifest_path = write_bundle(source, output, manifest)
        event = None
        if req.nostr_pubkey:
            event = build_release_event(manifest, req.nostr_pubkey, req.created_at if req.created_at is not None else int(time.time()))
            (output / "nostr-release-event.json").write_bytes(canonical_json(event))
        return {"bundle_dir": str(output.relative_to(ROOT)), "manifest": manifest, "manifest_path": str(manifest_path.relative_to(ROOT)), "event": event}
    except HTTPException:
        raise
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _adapter_bundle(name: str, version: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version):
        raise HTTPException(400, "invalid adapter name or version")
    bundle = (MODELS_DIR / "shared_adapters" / f"{name}-{version}").resolve()
    try:
        bundle.relative_to(MODELS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(400, "invalid adapter bundle path") from exc
    if not (bundle / "manifest.json").is_file():
        raise HTTPException(404, "local adapter bundle not found")
    return bundle


@app.post("/api/adapters/torrent")
def create_adapter_torrent(req: CreateTorrentRequest):
    """Create a BitTorrent metadata file for a local adapter bundle."""

    if not torrent_available():
        raise HTTPException(503, "BitTorrent support is unavailable; install libtorrent")
    bundle = _adapter_bundle(req.name, req.version)
    torrent_path = bundle.parent / f"{bundle.name}.torrent"
    try:
        create_torrent(bundle, torrent_path, trackers=req.trackers)
    except (OSError, NotADirectoryError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "torrent_path": str(torrent_path.relative_to(ROOT)),
        "torrent_available": True,
        "status": "created",
    }


@app.post("/api/adapters/upload")
def upload_adapter(req: UploadAdapterRequest):
    """Upload every verified file in a local bundle to one Blossom server."""

    bundle = _adapter_bundle(req.name, req.version)
    try:
        manifest = load_manifest(bundle / "manifest.json")
        from manga_pipeline.adapter_distribution import verify_bundle

        verify_bundle(manifest, bundle)
        results = upload_bundle_to_servers(
            manifest,
            bundle,
            req.server_urls,
            authorization=req.authorization,
            authorizations=req.authorizations,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"uploaded": True, "manifest_sha256": manifest_digest(manifest), "servers": results}


@app.post("/api/adapters/mirror")
def mirror_adapter_blob(req: MirrorBlobRequest):
    """Request a Blossom server to mirror one already-published blob."""

    try:
        descriptor = mirror_blob(
            req.server_url,
            req.source_url,
            authorization=req.authorization,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"mirrored": True, "descriptor": descriptor}


@app.post("/api/adapters/blossom/health")
def check_adapter_blossom_health(req: BlossomHealthRequest):
    """Check configured Blossom mirrors before publishing or downloading."""

    try:
        return {"servers": check_blossom_servers(req.server_urls, timeout=req.timeout)}
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _run_torrent_download_job(job_id: str, magnet: str, manifest: dict, destination: Path) -> None:
    job = _jobs[job_id]
    try:
        job.update({"status": "running", "message": "waiting for torrent metadata", "progress": 0.0, "peers": 0, "download_rate": 0, "upload_rate": 0, "seeding": False})

        def report(status: dict) -> None:
            job.update(status)
            job["message"] = "seeding" if status["seeding"] else "downloading"

        bundle = download_torrent(magnet, destination, manifest, status_callback=report)
        target = MODELS_DIR / "shared_adapters" / f"{manifest['name']}-{manifest['version']}"
        if target.exists():
            raise FileExistsError(target)
        bundle.replace(target)
        job.update({"status": "done", "message": "installed", "progress": 1.0, "bundle_dir": str(target.relative_to(ROOT)), "seeding": True, "finished_at": time.time()})
    except FileExistsError:
        _finish_job(job, "error", "adapter version is already installed")
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        _finish_job(job, "error", str(exc))
    finally:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)


@app.post("/api/adapters/torrent/download")
def start_torrent_download(req: DownloadTorrentRequest):
    """Start a verified magnet download and expose progress through /api/jobs."""

    try:
        _validate_release_for_manifest(req.manifest, req.release_event)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = _create_job()
    destination = MODELS_DIR / "shared_adapters" / f".torrent-download-{job_id}"
    thread = threading.Thread(target=_run_torrent_download_job, args=(job_id, req.magnet, req.manifest, destination), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/adapters/torrent/seed")
def seed_adapter(req: SeedAdapterRequest):
    """Start opt-in seeding for a verified local adapter bundle."""

    global _torrent_seeder
    bundle = _adapter_bundle(req.name, req.version)
    manifest = load_manifest(bundle / "manifest.json")
    try:
        verify_bundle(manifest, bundle)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    torrent = bundle.parent / f"{bundle.name}.torrent"
    magnet = req.magnet or manifest.get("distribution", {}).get("torrent", {}).get("magnet")
    if not torrent.is_file() and not magnet:
        raise HTTPException(400, "adapter has no torrent metadata or magnet")
    if not torrent_available():
        raise HTTPException(503, "BitTorrent support is unavailable; install libtorrent")
    try:
        if _torrent_seeder is None:
            _torrent_seeder = TorrentSeeder(persistence_path=SEED_STATE_PATH)
        return _torrent_seeder.start(f"{req.name}-{req.version}", bundle, torrent_path=torrent if torrent.is_file() else None, magnet=magnet)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/adapters/torrent/seed/{name}/{version}")
def adapter_seed_status(name: str, version: str):
    if _torrent_seeder is None:
        return {"seeding": False, "seed_id": f"{name}-{version}"}
    try:
        return _torrent_seeder.status(f"{name}-{version}")
    except KeyError:
        return {"seeding": False, "seed_id": f"{name}-{version}"}


@app.post("/api/adapters/torrent/seed/{name}/{version}/stop")
def stop_adapter_seeding(name: str, version: str):
    if _torrent_seeder is None:
        return {"stopped": False, "seed_id": f"{name}-{version}"}
    try:
        _torrent_seeder.stop(f"{name}-{version}")
    except KeyError:
        return {"stopped": False, "seed_id": f"{name}-{version}"}
    return {"stopped": True, "seed_id": f"{name}-{version}"}


@app.post("/api/adapters/composition")
def create_adapter_composition(req: CreateCompositionRequest):
    """Create a lineage-preserving composition manifest from local adapters."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", req.name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", req.version):
        raise HTTPException(400, "composition name or version contains unsupported characters")
    manifests = []
    components = []
    try:
        for component in req.components:
            manifest = load_manifest(_adapter_bundle(component.name, component.version) / "manifest.json")
            manifests.append(manifest)
            components.append(
                {
                    "name": manifest["name"],
                    "version": manifest["version"],
                    "manifest_sha256": manifest_digest(manifest),
                    "weight": component.weight,
                }
            )
        compatible_base = compatible_manifests(manifests)
        if req.base_model != compatible_base:
            raise ValueError("composition base model does not match its components")
        composition = build_composition(
            req.name,
            req.version,
            compatible_base,
            components,
            description=req.description,
            evaluations=req.evaluations,
            community_merge=req.community_merge,
        )
        output = MODELS_DIR / "shared_adapters" / "compositions"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{req.name}-{req.version}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(canonical_json(composition))
    except HTTPException:
        raise
    except FileExistsError:
        raise HTTPException(409, "composition version already exists")
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"composition": composition, "path": str(path.relative_to(ROOT))}


@app.get("/api/adapters/compositions")
def list_adapter_compositions():
    """List valid local adapter-bank manifests."""

    root = MODELS_DIR / "shared_adapters" / "compositions"
    compositions = []
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        try:
            composition = json.loads(path.read_text(encoding="utf-8"))
            from manga_pipeline.adapter_distribution import validate_composition

            validate_composition(composition)
            compositions.append({"name": composition["name"], "version": composition["version"], "base_model": composition["base_model"], "component_count": len(composition["components"]), "evaluation_count": len(composition.get("evaluations", [])), "community_merge": composition.get("community_merge", False), "composition": composition, "path": str(path.relative_to(ROOT))})
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {"compositions": compositions}


@app.post("/api/adapters/discover")
def discover_adapters(req: DiscoverAdaptersRequest):
    """Discover Hypotaxis release metadata from configured Nostr relays."""

    relays = [relay.strip() for relay in req.relays if relay.strip()]
    if not relays:
        raise HTTPException(400, "at least one Nostr relay URL is required")
    try:
        from manga_pipeline.adapter_distribution import NOSTR_RELEASE_KIND

        events = query_nostr_relays(relays, [{"kinds": [NOSTR_RELEASE_KIND], "limit": req.limit}], max_events=req.limit)
        composition_events = query_nostr_relays(relays, [{"kinds": [30079], "limit": req.limit}], max_events=req.limit)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    releases_by_address = {}
    can_verify_signatures = schnorr_available()
    for event in events:
        try:
            manifest = parse_release_event(event)
        except (ValueError, json.JSONDecodeError):
            continue
        signature_verified = False
        if can_verify_signatures:
            signature_verified = verify_schnorr_signature(event)
            if not signature_verified:
                continue
        address = next((tag[1] for tag in event.get("tags", []) if isinstance(tag, list) and len(tag) >= 2 and tag[0] == "d"), f"adapter:{manifest['name']}")
        key = f"{event['pubkey']}:{address}"
        release = {
            "event_id": event["id"],
            "creator_pubkey": event["pubkey"],
            "signature_verified": signature_verified,
            "manifest": manifest,
        }
        previous = releases_by_address.get(key)
        if previous is None or event["created_at"] > previous["_created_at"]:
            releases_by_address[key] = {**release, "event": event, "_created_at": event["created_at"]}
    releases = [{key: value for key, value in release.items() if key != "_created_at"} for release in releases_by_address.values()]
    compositions_by_address = {}
    for event in composition_events:
        try:
            composition = parse_composition_event(event)
        except (ValueError, json.JSONDecodeError):
            continue
        address = next((tag[1] for tag in event.get("tags", []) if isinstance(tag, list) and len(tag) >= 2 and tag[0] == "d"), f"composition:{composition['name']}")
        key = f"{event['pubkey']}:{address}"
        previous = compositions_by_address.get(key)
        if previous is None or event["created_at"] > previous["_created_at"]:
            compositions_by_address[key] = {"event_id": event["id"], "creator_pubkey": event["pubkey"], "signature_verified": can_verify_signatures and verify_schnorr_signature(event), "composition": composition, "_created_at": event["created_at"]}
    compositions = [{key: value for key, value in item.items() if key != "_created_at"} for item in compositions_by_address.values()]
    if compositions:
        deletion_events = query_nostr_relays(relays, [{"kinds": [5], "#e": [item["event_id"] for item in compositions], "limit": req.limit * 5}], max_events=req.limit * 5)
        compositions = [item for item in compositions if not release_is_revoked(next(event for event in composition_events if event["id"] == item["event_id"]), deletion_events)]
    return {
        "releases": releases,
        "compositions": compositions,
        "signature_verification": "verified" if can_verify_signatures else "not available",
    }


def _validate_release_for_manifest(manifest: dict, release_event: dict) -> None:
    """Require transport requests to retain the signed release provenance."""

    signed_manifest = parse_release_event(release_event)
    if signed_manifest != manifest:
        raise ValueError("release event manifest does not match install manifest")
    if schnorr_available() and not verify_schnorr_signature(release_event):
        raise ValueError("release event signature is invalid")


@app.post("/api/adapters/install")
def install_adapter(req: InstallAdapterRequest):
    """Install a discovered adapter after verified Blossom downloads."""

    try:
        _validate_release_for_manifest(req.manifest, req.release_event)
        target = install_from_blossom(req.manifest, MODELS_DIR / "shared_adapters")
    except FileExistsError:
        raise HTTPException(409, "adapter version is already installed")
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"installed": True, "bundle_dir": str(target.relative_to(ROOT))}


@app.post("/api/adapters/composition/install")
def install_composition(req: InstallCompositionRequest):
    """Install every verified Blossom component referenced by a composition."""

    installed_targets = []
    try:
        from manga_pipeline.adapter_distribution import validate_composition

        validate_composition(req.composition)
        signed_composition = parse_composition_event(req.composition_event)
        if signed_composition != req.composition:
            raise ValueError("composition event does not match install composition")
        if schnorr_available() and not verify_schnorr_signature(req.composition_event):
            raise ValueError("composition event signature is invalid")
        events_by_component = {}
        for event in req.release_events:
            manifest = parse_release_event(event)
            _validate_release_for_manifest(manifest, event)
            events_by_component[(manifest["name"], manifest["version"])] = (manifest, event)
        manifests = []
        for component in req.composition["components"]:
            key = (component["name"], component["version"])
            if key not in events_by_component:
                raise ValueError(f"missing signed release for {component['name']}@{component['version']}")
            manifest, _event = events_by_component[key]
            if manifest_digest(manifest) != component["manifest_sha256"]:
                raise ValueError(f"manifest digest mismatch for {component['name']}@{component['version']}")
            manifests.append(manifest)
        if len({manifest["base_model"] for manifest in manifests}) != 1 or manifests[0]["base_model"] != req.composition["base_model"]:
            raise ValueError("composition components target incompatible base models")
        installed = []
        for manifest in manifests:
            if manifest.get("distribution", {}).get("blossom"):
                target = install_from_blossom(manifest, MODELS_DIR / "shared_adapters")
            else:
                magnet = manifest.get("distribution", {}).get("torrent", {}).get("magnet")
                if not magnet:
                    raise ValueError(f"component {manifest['name']} has no Blossom source or torrent magnet")
                target = install_from_torrent(magnet, manifest, MODELS_DIR / "shared_adapters")
            installed_targets.append(target)
            installed.append(str(target.relative_to(ROOT)))
    except FileExistsError:
        for target in installed_targets:
            shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(409, "a composition component is already installed")
    except (OSError, RuntimeError, ValueError) as exc:
        for target in installed_targets:
            shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return {"installed": installed, "composition": req.composition["name"]}


@app.get("/api/adapters/local")
def list_local_adapters():
    """List locally packaged, valid adapter bundles."""

    adapters = []
    root = MODELS_DIR / "shared_adapters"
    if not root.exists():
        return {"adapters": adapters}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = load_manifest(manifest_path)
            torrent_path = manifest_path.parent.parent / f"{manifest_path.parent.name}.torrent"
            adapters.append(
                {
                    "name": manifest["name"],
                    "version": manifest["version"],
                    "base_model": manifest["base_model"],
                    "file_count": len(manifest["files"]),
                    "manifest_sha256": manifest_digest(manifest),
                    "manifest": manifest,
                    "bundle_dir": str(manifest_path.parent.relative_to(ROOT)),
                    "torrent_available": torrent_available(),
                    "torrent_exists": torrent_path.is_file(),
                    "torrent_path": str(torrent_path.relative_to(ROOT)) if torrent_path.is_file() else None,
                }
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {"adapters": adapters}


@app.delete("/api/adapters/{name}/{version}")
def remove_local_adapter(name: str, version: str):
    """Remove one validated local adapter bundle and its adjacent torrent."""

    bundle = _adapter_bundle(name, version)
    try:
        if _torrent_seeder is not None:
            try:
                _torrent_seeder.stop(f"{name}-{version}")
            except KeyError:
                pass
        shutil.rmtree(bundle)
        torrent = bundle.parent / f"{bundle.name}.torrent"
        torrent.unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"removed": True, "name": name, "version": version}


class AdaptRequest(BaseModel):
    id: str = Field(max_length=80)
    title: str = Field(max_length=500)
    prose: str = Field(max_length=2_000_000)
    style_prompt: str = Field("monochrome manga, screentone shading, dynamic ink linework", max_length=2000)
    llm: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"
    character_profiles: str = Field("", max_length=100_000)
    location_profiles: str = Field("", max_length=100_000)
    prop_profiles: str = Field("", max_length=100_000)
    use_trained_captioner: bool = False


def _run_adapt_job(job_id: str, req: AdaptRequest, story_id: str) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
        location_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_locations.json")
        prop_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_props.json")
        llm = SmallLLM(model_id=req.llm, device=req.device)
        character_profiles, abstract_characters = (
            parse_character_profiles(req.character_profiles) if req.character_profiles else (None, None)
        )
        location_profiles = parse_location_profiles(req.location_profiles) if req.location_profiles else None
        prop_profiles = parse_prop_profiles(req.prop_profiles) if req.prop_profiles else None
        captioner = (
            Captioner(CAPTIONER_ADAPTER_DIR, base_model=CAPTIONER_BASE_MODEL, device=req.device)
            if req.use_trained_captioner
            else None
        )
        story = adapt_story(
            req.prose,
            story_id,
            req.title,
            req.style_prompt,
            registry,
            llm,
            dataset_path=DATASET_PATH,
            on_progress=on_progress,
            character_profiles=character_profiles,
            abstract_characters=abstract_characters,
            location_registry=location_registry,
            location_profiles=location_profiles,
            prop_registry=prop_registry,
            prop_profiles=prop_profiles,
            captioner=captioner,
        )
        STORIES_DIR.mkdir(parents=True, exist_ok=True)
        story.save(STORIES_DIR / f"{story_id}.json")
        _finish_job(job, "done")
        job["story_id"] = story_id
    except ValueError as e:
        _finish_job(job, "error", str(e))
    except Exception as e:  # noqa: BLE001
        _finish_job(job, "error", str(e))
    finally:
        _release_gpu()


@app.post("/api/stories/adapt")
def adapt(req: AdaptRequest):
    story_id = req.id.strip()
    if not story_id or not story_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "story id must be alphanumeric (with - or _)")
    if req.use_trained_captioner and not CAPTIONER_ADAPTER_DIR.exists():
        raise HTTPException(
            400,
            f"no trained captioner found at {CAPTIONER_ADAPTER_DIR} - run train_captioner.py first "
            "(see README's \"Curating a clean caption dataset\")",
        )

    if not _try_claim_gpu():
        raise HTTPException(409, "another job (adapt or generate) is already running - only one at a time on this GPU")

    job_id = _create_job()
    thread = threading.Thread(target=_run_adapt_job, args=(job_id, req, story_id), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/stories")
def list_stories():
    stories = []
    for path in sorted(STORIES_DIR.glob("*.json")):
        try:
            story = Story.load(path)
        except Exception:
            continue
        panel_count = sum(len(p.panels) for p in story.pages)
        pdf_path = OUTPUT_DIR / story.id / f"{story.id}.pdf"
        stories.append(
            {
                "id": story.id,
                "title": story.title,
                "page_count": len(story.pages),
                "panel_count": panel_count,
                "has_output": pdf_path.exists(),
            }
        )
    return {"stories": stories}


def _story_or_404(story_id: str) -> Story:
    story_id = _validate_story_id(story_id)
    path = STORIES_DIR / f"{story_id}.json"
    if not path.exists():
        raise HTTPException(404, "story not found")
    return Story.load(path)


@app.get("/api/stories/{story_id}")
def get_story(story_id: str):
    story = _story_or_404(story_id)
    return asdict(story)


class DialogueLineUpdate(BaseModel):
    speaker: str
    text: str
    kind: str = "speech"


class PanelUpdate(BaseModel):
    scene_description: str = Field(max_length=10_000)
    camera_hint: str = Field("medium shot", max_length=200)
    characters: list[str] = []
    locations: list[str] = []
    props: list[str] = []
    dialogue: list[DialogueLineUpdate] = []


@app.put("/api/stories/{story_id}/pages/{page_index}/panels/{panel_index}")
def update_panel(story_id: str, page_index: int, panel_index: int, req: PanelUpdate):
    """Manual-correction step: Stage A's caption/speaker/tag output is
    generated, not guaranteed, and this session's own testing found real
    hallucinated captions, misattributed speakers, and bogus tags on an
    actual manuscript - previously the only fix was hand-editing the Story
    JSON file directly. Lets the studio UI edit and persist a single
    panel's fields without re-running the whole adapt step (which would
    also risk different/worse LLM output the second time)."""
    story = _story_or_404(story_id)
    if not 0 <= page_index < len(story.pages):
        raise HTTPException(404, "page not found")
    page = story.pages[page_index]
    if not 0 <= panel_index < len(page.panels):
        raise HTTPException(404, "panel not found")

    page.panels[panel_index] = Panel(
        scene_description=req.scene_description,
        characters=req.characters,
        locations=req.locations,
        props=req.props,
        camera_hint=req.camera_hint,
        dialogue=[DialogueLine(speaker=line.speaker, text=line.text, kind=line.kind) for line in req.dialogue],
    )
    story.save(STORIES_DIR / f"{story_id}.json")
    return {"updated": True}


@app.delete("/api/stories/{story_id}")
def delete_story(story_id: str):
    _story_or_404(story_id)  # 404 before touching anything
    (STORIES_DIR / f"{story_id}.json").unlink(missing_ok=True)
    for suffix in ("", "_locations", "_props"):
        (REGISTRY_DIR / f"{story_id}{suffix}.json").unlink(missing_ok=True)
    shutil.rmtree(OUTPUT_DIR / story_id, ignore_errors=True)
    return {"deleted": story_id}


@app.delete("/api/stories/{story_id}/pages")
def delete_pages(story_id: str):
    """Clears generated page images and the PDF only - leaves the script and
    any designed character/location/prop reference images alone, so
    "Generate Pages" can be re-run from a clean slate without redesigning
    the cast."""
    _story_or_404(story_id)
    out_dir = OUTPUT_DIR / story_id
    if out_dir.exists():
        for page_path in out_dir.glob("page_*.png"):
            page_path.unlink()
        (out_dir / f"{story_id}.pdf").unlink(missing_ok=True)
    return {"cleared": story_id}


def _registry_payload(registry: CharacterRegistry) -> dict:
    result = {}
    for name, entry in registry.all().items():
        ref_url = None
        if entry.reference_image:
            ref_path = _project_path(entry.reference_image)
            try:
                rel = ref_path.resolve().relative_to(OUTPUT_DIR.resolve())
                ref_url = f"/output/{rel.as_posix()}"
            except ValueError:
                ref_url = None
        result[name] = {
            "description": entry.description,
            "reference_image_url": ref_url,
            "has_lora": bool(entry.lora_path and _project_path(entry.lora_path).exists()),
        }
    return result


@app.get("/api/stories/{story_id}/registry")
def get_registry(story_id: str):
    _story_or_404(story_id)
    story_id = _validate_story_id(story_id)
    registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
    return {"characters": _registry_payload(registry)}


@app.get("/api/stories/{story_id}/locations")
def get_locations(story_id: str):
    _story_or_404(story_id)
    story_id = _validate_story_id(story_id)
    location_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_locations.json")
    return {"locations": _registry_payload(location_registry)}


@app.get("/api/stories/{story_id}/props")
def get_props(story_id: str):
    _story_or_404(story_id)
    story_id = _validate_story_id(story_id)
    prop_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_props.json")
    return {"props": _registry_payload(prop_registry)}


def _delete_registry_entry(registry: CharacterRegistry, name: str) -> None:
    """Deletes a single entry - e.g. a bogus name NER mistook for a
    character/location/prop (alias merging catches variant spellings of a
    real name, but not a wrong entity-type call entirely - those need
    pruning by hand). 404s if the name isn't actually in this registry, so
    a stale/mistyped request doesn't silently no-op. Also removes the
    reference image file from disk, if any, since the registry itself
    doesn't own that filesystem lifecycle."""
    entry = registry.get(name)
    if entry is None:
        raise HTTPException(404, f"'{name}' not found in this registry")
    if entry.reference_image:
        reference = _project_path(entry.reference_image).resolve()
        try:
            reference.relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            raise HTTPException(400, "reference image is outside the output directory")
        reference.unlink(missing_ok=True)
    registry.delete(name)


@app.delete("/api/stories/{story_id}/registry/{name}")
def delete_character(story_id: str, name: str):
    _story_or_404(story_id)
    registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
    _delete_registry_entry(registry, name)
    return {"deleted": name}


@app.delete("/api/stories/{story_id}/locations/{name}")
def delete_location(story_id: str, name: str):
    _story_or_404(story_id)
    location_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_locations.json")
    _delete_registry_entry(location_registry, name)
    return {"deleted": name}


@app.delete("/api/stories/{story_id}/props/{name}")
def delete_prop(story_id: str, name: str):
    _story_or_404(story_id)
    prop_registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}_props.json")
    _delete_registry_entry(prop_registry, name)
    return {"deleted": name}


@app.get("/api/stories/{story_id}/pages")
def get_pages(story_id: str):
    story_id = _validate_story_id(story_id)
    out_dir = OUTPUT_DIR / story_id
    if not out_dir.exists():
        return {"pages": [], "pdf_url": None}
    pages = sorted(p.name for p in out_dir.glob("page_*.png"))
    pdf_path = out_dir / f"{story_id}.pdf"
    return {
        "pages": [f"/output/{story_id}/{name}" for name in pages],
        "pdf_url": f"/output/{story_id}/{story_id}.pdf" if pdf_path.exists() else None,
    }


class PrepareCastRequest(BaseModel):
    backend: str = "mock"
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "auto"
    steps: int = Field(4, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0, le=30)
    use_identity_adapter: bool = True
    identity_adapter_scale: float = Field(0.6, ge=0, le=2)
    force: bool = False


def _run_prepare_cast_job(job_id: str, story: Story, cfg: PipelineConfig, force: bool) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        prepare_cast_pipeline(story, cfg, on_progress=on_progress, force=force)
        _finish_job(job, "done")
    except Exception as e:  # noqa: BLE001
        _finish_job(job, "error", str(e))
    finally:
        _release_gpu()


@app.post("/api/stories/{story_id}/prepare-cast")
def prepare_cast(story_id: str, req: PrepareCastRequest):
    story = _story_or_404(story_id)

    if not _try_claim_gpu():
        raise HTTPException(409, "another job (adapt or generate) is already running - only one at a time on this GPU")

    cfg = PipelineConfig(
        backend=req.backend,
        checkpoint=req.checkpoint,
        device=req.device,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        use_identity_adapter=req.use_identity_adapter,
        identity_adapter_scale=req.identity_adapter_scale,
        output_dir=str(OUTPUT_DIR),
        registry_dir=str(REGISTRY_DIR),
    )

    job_id = _create_job()
    thread = threading.Thread(target=_run_prepare_cast_job, args=(job_id, story, cfg, req.force), daemon=True)
    thread.start()
    return {"job_id": job_id}


class TrainCharacterLoraRequest(BaseModel):
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "auto"
    bootstrap_count: int = Field(len(TRAINING_VIEW_PROMPTS), ge=1, le=len(TRAINING_VIEW_PROMPTS))
    steps: int = Field(300, ge=10, le=5000)
    rank: int = Field(8, ge=1, le=64)
    lr: float = Field(1e-4, gt=0, le=1e-2)
    resolution: int = Field(768, ge=256, le=1024)


def _run_train_lora_job(
    job_id: str, story_id: str, name: str, story: Story, req: TrainCharacterLoraRequest
) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
        entry = registry.get(name)
        if entry is None:
            raise ValueError(f"'{name}' not found in this story's registry")

        cfg = PipelineConfig(
            backend="diffusers", checkpoint=req.checkpoint, device=req.device, output_dir=str(OUTPUT_DIR)
        )
        backend = DiffusersBackend(cfg)

        on_progress(f"generating {req.bootstrap_count} bootstrap training images")
        image_paths = backend.generate_character_lora_images(
            story_id, name, story.style_prompt, registry, count=req.bootstrap_count
        )
        captions = [
            build_training_caption(name, entry.description, story.style_prompt, view)
            for view in TRAINING_VIEW_PROMPTS[: len(image_paths)]
        ]

        output_dir = default_lora_output_dir(MODELS_DIR, story_id, name)
        train_character_lora(
            image_paths,
            captions,
            checkpoint=req.checkpoint,
            output_dir=output_dir,
            rank=req.rank,
            steps=req.steps,
            learning_rate=req.lr,
            resolution=req.resolution,
            device=req.device,
            on_progress=on_progress,
        )
        registry.set_lora_path(name, str(output_dir))
        _finish_job(job, "done")
    except Exception as e:  # noqa: BLE001
        _finish_job(job, "error", str(e))
    finally:
        _release_gpu()


@app.post("/api/stories/{story_id}/characters/{name}/train-lora")
def train_lora(story_id: str, name: str, req: TrainCharacterLoraRequest):
    story = _story_or_404(story_id)
    registry = CharacterRegistry(REGISTRY_DIR / f"{_validate_story_id(story_id)}.json")
    if registry.get(name) is None:
        raise HTTPException(404, f"'{name}' not found in this story's registry - generate cast first")

    if not _try_claim_gpu():
        raise HTTPException(409, "another job (adapt, cast, or generate) is already running - only one at a time on this GPU")

    job_id = _create_job()
    thread = threading.Thread(target=_run_train_lora_job, args=(job_id, story_id, name, story, req), daemon=True)
    thread.start()
    return {"job_id": job_id}


class GenerateRequest(BaseModel):
    backend: str = "mock"
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "auto"
    steps: int = Field(4, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0, le=30)
    use_identity_adapter: bool = True
    identity_adapter_scale: float = Field(0.6, ge=0, le=2)
    use_character_lora: bool = False
    character_lora_scale: float = Field(0.8, ge=0, le=2)
    adapter_composition_path: str = Field("", max_length=500)
    use_pose_controlnet: bool = False
    pose_controlnet_scale: float = Field(0.5, ge=0, le=2)
    force: bool = False


def _run_generation_job(job_id: str, story: Story, cfg: PipelineConfig, force: bool) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        pdf_path = run_pipeline(story, cfg, on_progress=on_progress, force=force)
        _finish_job(job, "done")
        job["pdf_url"] = f"/output/{story.id}/{Path(pdf_path).name}"
    except Exception as e:  # noqa: BLE001
        _finish_job(job, "error", str(e))
    finally:
        _release_gpu()


@app.post("/api/stories/{story_id}/generate")
def generate(story_id: str, req: GenerateRequest):
    story = _story_or_404(story_id)

    composition_path = ""
    if req.adapter_composition_path:
        composition = _path_under(req.adapter_composition_path, MODELS_DIR / "shared_adapters" / "compositions")
        if not composition.is_file():
            raise HTTPException(404, "adapter composition not found")
        composition_path = str(composition)

    if not _try_claim_gpu():
        raise HTTPException(409, "another job (adapt or generate) is already running - only one at a time on this GPU")

    cfg = PipelineConfig(
        backend=req.backend,
        checkpoint=req.checkpoint,
        device=req.device,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        use_identity_adapter=req.use_identity_adapter,
        identity_adapter_scale=req.identity_adapter_scale,
        use_character_lora=req.use_character_lora,
        character_lora_scale=req.character_lora_scale,
        adapter_composition_path=composition_path,
        use_pose_controlnet=req.use_pose_controlnet,
        pose_controlnet_scale=req.pose_controlnet_scale,
        output_dir=str(OUTPUT_DIR),
        registry_dir=str(REGISTRY_DIR),
    )

    job_id = _create_job()
    thread = threading.Thread(target=_run_generation_job, args=(job_id, story, cfg, req.force), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _cleanup_jobs()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/dataset/stats")
def dataset_stats():
    if not DATASET_PATH.exists():
        return {"total_pairs": 0}
    with DATASET_PATH.open() as f:
        total = sum(1 for line in f if line.strip())
    return {"total_pairs": total}


CANDIDATES_PATH = ROOT / "data" / "caption_candidates.jsonl"
CURATED_PATH = ROOT / "data" / "caption_pairs_curated.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _append_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/api/dataset/candidates")
def list_candidates(limit: int = 20):
    """Pending teacher-generated caption candidates awaiting human review -
    see curate_dataset.py, which generates these with a stronger teacher LLM
    than the production bridge model. Each candidate's position in the file
    is its id for accept/reject, since a local single-user review queue
    doesn't need anything sturdier than that."""
    candidates = _read_jsonl(CANDIDATES_PATH)
    curated_count = len(_read_jsonl(CURATED_PATH))
    return {
        "candidates": [{"index": i, **c} for i, c in enumerate(candidates[:limit])],
        "pending_count": len(candidates),
        "curated_count": curated_count,
    }


class CandidateDecision(BaseModel):
    target: str | None = None  # edited caption text; None keeps the candidate's own target
    characters: list[str] | None = None
    camera: str | None = None  # edited camera hint; None keeps the candidate's own camera


@app.post("/api/dataset/candidates/{index}/accept")
def accept_candidate(index: int, decision: CandidateDecision):
    candidates = _read_jsonl(CANDIDATES_PATH)
    if not 0 <= index < len(candidates):
        raise HTTPException(404, "candidate not found")
    candidate = candidates.pop(index)
    camera = decision.camera if decision.camera is not None else candidate.get("camera")
    if camera not in CAMERA_HINTS:
        raise HTTPException(400, f"camera must be one of {CAMERA_HINTS}")
    curated = {
        "input": candidate["input"],
        "characters": decision.characters if decision.characters is not None else candidate.get("characters", []),
        "target": decision.target.strip() if decision.target is not None else candidate["target"],
        "camera": camera,
    }
    if not curated["target"]:
        raise HTTPException(400, "caption text cannot be empty")
    _append_jsonl_record(CURATED_PATH, curated)
    _write_jsonl(CANDIDATES_PATH, candidates)
    return {"accepted": True, "remaining": len(candidates)}


@app.post("/api/dataset/candidates/{index}/reject")
def reject_candidate(index: int):
    candidates = _read_jsonl(CANDIDATES_PATH)
    if not 0 <= index < len(candidates):
        raise HTTPException(404, "candidate not found")
    candidates.pop(index)
    _write_jsonl(CANDIDATES_PATH, candidates)
    return {"rejected": True, "remaining": len(candidates)}
