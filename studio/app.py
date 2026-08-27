from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manga_pipeline.config import PipelineConfig  # noqa: E402
from manga_pipeline.llm import SmallLLM  # noqa: E402
from manga_pipeline.pipeline import run as run_pipeline  # noqa: E402
from manga_pipeline.registry import CharacterRegistry  # noqa: E402
from manga_pipeline.schema import Story  # noqa: E402
from manga_pipeline.story_adapt import adapt_story, parse_character_profiles  # noqa: E402

STORIES_DIR = ROOT / "stories"
REGISTRY_DIR = ROOT / "registry"
OUTPUT_DIR = ROOT / "output"
DATASET_PATH = ROOT / "data" / "caption_pairs.jsonl"

app = FastAPI(title="Manga Production Studio")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="output")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_jobs: dict[str, dict] = {}
# shared by both adapt (Stage A, small LLM) and generate (Stage B-D, diffusion) -
# on modest hardware these shouldn't run concurrently either, since Stage A's
# LLM and Stage C's diffusion pipeline compete for the same GPU
_gpu_lock = threading.Lock()
_gpu_busy = False


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
    if Path(file.filename).suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(400, "unsupported file type - use .txt, .md, or .docx")
    content = await file.read()
    try:
        text = _extract_text(file.filename, content)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read file: {e}")
    return {"text": text}


class AdaptRequest(BaseModel):
    id: str
    title: str
    prose: str
    style_prompt: str = "monochrome manga, screentone shading, dynamic ink linework"
    llm: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    character_profiles: str = ""


def _run_adapt_job(job_id: str, req: AdaptRequest, story_id: str) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
        llm = SmallLLM(model_id=req.llm, device=req.device)
        character_profiles = parse_character_profiles(req.character_profiles) if req.character_profiles else None
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
        )
        STORIES_DIR.mkdir(parents=True, exist_ok=True)
        story.save(STORIES_DIR / f"{story_id}.json")
        job["status"] = "done"
        job["message"] = "done"
        job["story_id"] = story_id
    except ValueError as e:
        job["status"] = "error"
        job["message"] = str(e)
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = str(e)
    finally:
        _release_gpu()


@app.post("/api/stories/adapt")
def adapt(req: AdaptRequest):
    story_id = req.id.strip()
    if not story_id or not story_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "story id must be alphanumeric (with - or _)")

    if not _try_claim_gpu():
        raise HTTPException(409, "another job (adapt or generate) is already running - only one at a time on this GPU")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "message": "queued"}
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
    path = STORIES_DIR / f"{story_id}.json"
    if not path.exists():
        raise HTTPException(404, "story not found")
    return Story.load(path)


@app.get("/api/stories/{story_id}")
def get_story(story_id: str):
    story = _story_or_404(story_id)
    return asdict(story)


@app.get("/api/stories/{story_id}/registry")
def get_registry(story_id: str):
    registry = CharacterRegistry(REGISTRY_DIR / f"{story_id}.json")
    entries = registry.all()
    result = {}
    for name, entry in entries.items():
        ref_url = None
        if entry.reference_image:
            ref_path = Path(entry.reference_image)
            try:
                rel = ref_path.resolve().relative_to(OUTPUT_DIR.resolve())
                ref_url = f"/output/{rel.as_posix()}"
            except ValueError:
                ref_url = None
        result[name] = {"description": entry.description, "reference_image_url": ref_url}
    return {"characters": result}


@app.get("/api/stories/{story_id}/pages")
def get_pages(story_id: str):
    out_dir = OUTPUT_DIR / story_id
    if not out_dir.exists():
        return {"pages": [], "pdf_url": None}
    pages = sorted(p.name for p in out_dir.glob("page_*.png"))
    pdf_path = out_dir / f"{story_id}.pdf"
    return {
        "pages": [f"/output/{story_id}/{name}" for name in pages],
        "pdf_url": f"/output/{story_id}/{story_id}.pdf" if pdf_path.exists() else None,
    }


class GenerateRequest(BaseModel):
    backend: str = "mock"
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "cuda"
    steps: int = 4
    guidance_scale: float = 1.0
    use_identity_adapter: bool = True
    identity_adapter_scale: float = 0.6


def _run_generation_job(job_id: str, story: Story, cfg: PipelineConfig) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(msg: str) -> None:
        job["message"] = msg

    try:
        pdf_path = run_pipeline(story, cfg, on_progress=on_progress)
        job["status"] = "done"
        job["message"] = "done"
        job["pdf_url"] = f"/output/{story.id}/{Path(pdf_path).name}"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = str(e)
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

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "message": "queued"}
    thread = threading.Thread(target=_run_generation_job, args=(job_id, story, cfg), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
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
