from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manga_pipeline.captioner import Captioner  # noqa: E402
from manga_pipeline.config import PipelineConfig  # noqa: E402
from manga_pipeline.llm import SmallLLM  # noqa: E402
from manga_pipeline.pipeline import prepare_cast as prepare_cast_pipeline  # noqa: E402
from manga_pipeline.pipeline import run as run_pipeline  # noqa: E402
from manga_pipeline.registry import CharacterRegistry  # noqa: E402
from manga_pipeline.schema import DialogueLine, Panel, Story  # noqa: E402
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

app = FastAPI(title="Manga Production Studio")
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
        result[name] = {"description": entry.description, "reference_image_url": ref_url}
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


class GenerateRequest(BaseModel):
    backend: str = "mock"
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "auto"
    steps: int = Field(4, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0, le=30)
    use_identity_adapter: bool = True
    identity_adapter_scale: float = Field(0.6, ge=0, le=2)
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


@app.post("/api/dataset/candidates/{index}/accept")
def accept_candidate(index: int, decision: CandidateDecision):
    candidates = _read_jsonl(CANDIDATES_PATH)
    if not 0 <= index < len(candidates):
        raise HTTPException(404, "candidate not found")
    candidate = candidates.pop(index)
    curated = {
        "input": candidate["input"],
        "characters": decision.characters if decision.characters is not None else candidate.get("characters", []),
        "target": decision.target.strip() if decision.target is not None else candidate["target"],
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
