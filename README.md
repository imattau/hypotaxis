# Manga Production Pipeline

A local pipeline that turns a prose story into a manga/comic — panels, dialogue, character
art — built primarily on small embedding models, transformers, and LoRA adapters rather
than a large LLM, so it stays usable on modest consumer GPU hardware. Local-first: no
required cloud APIs.

## Pipeline stages

| Stage | What it does | Key tech |
|---|---|---|
| **A — Story Adaptation** | Prose → structured Story JSON script (panels, dialogue, camera hints) | sentence embeddings for scene segmentation, LexRank (networkx) for panel-budget compression, spaCy NER + dependency parsing for characters/speaker attribution, a small local LLM (Qwen2.5-0.5B-Instruct by default) for panel captions |
| **B — Character Identity** | Persistent per-character visual identity across the whole story | text registry + IP-Adapter reference portraits (generated once, reused everywhere) |
| **C — Generation** | One base image per page, edited into panels | SDXL / SDXL-Turbo via `diffusers`, img2img panel edits |
| **D — Assembly** | Panels → laid-out page → dialogue bubbles → PDF | Pillow, deterministic, no model |

A web studio UI (FastAPI + vanilla JS) sits on top for running the whole flow visually.

## Setup

Install what you need — these are split so you don't have to pull in the diffusion stack
just to run Stage A, etc.

```bash
pip install -r requirements.txt              # Stage D / mock backend (Pillow only)
pip install -r requirements-story-adapt.txt  # Stage A (embeddings, NER, LLM bridge)
python -m spacy download en_core_web_sm      # Stage A's NER model
pip install -r requirements-generation.txt   # Stage C real generation (diffusers/torch)
pip install -r requirements-studio.txt       # web studio UI
pip install -r requirements-training.txt     # Phase 3 LoRA captioner training (optional)
```

On Debian/Ubuntu-style systems with an externally-managed Python, add
`--user --break-system-packages` to the `pip install` calls, or use a virtualenv instead.

## Quick start

**Studio UI** (recommended way to explore the pipeline):

```bash
python run_studio.py --port 8420
```

Open `http://127.0.0.1:8420`, create a new story (paste text or upload a `.txt`/`.md`/`.docx`
chapter), optionally add a [character profile](#character-profiles) cast sheet, then generate
pages with the `mock` backend (instant, no GPU) or `diffusers` (real generation).

**Command line**, the same two stages the UI drives:

```bash
# Stage A: prose -> Story JSON
python adapt_story.py stories/rain_letter.txt --id rain_letter --title "The Letter in the Rain"

# Stage B-D: Story JSON -> pages + PDF
python run_story.py stories/rain_letter.json --backend mock          # instant, no GPU
python run_story.py stories/rain_letter.json --backend diffusers     # real generation
```

Output lands in `output/<story_id>/` (page PNGs, character reference portraits, final PDF).

## Character profiles

Optionally hand the adapter a cast sheet so character names and appearances don't depend
on automatic detection — see `stories/character_profiles.example.txt` for the format
(one `Name: description` per line). Pass it via `--character-profiles <path>` on the CLI,
or the "Character profiles" field in the studio's New Story form.

## Project status

This is an active prototype, not a finished product. Known limitations are tracked
honestly rather than papered over:

- The Stage A caption LLM (0.5B) sometimes hallucinates content not in the source text.
- Dialogue speaker attribution can't resolve pure-pronoun cases ("he called out") without
  coreference resolution, which isn't wired in (the well-known libraries for this —
  `coreferee`, `spacy-experimental`, `BookNLP` — are all currently incompatible with a
  modern Python/spaCy/transformers stack; a character profile cast sheet sidesteps this
  for named characters).
- spaCy's NER (character detection) has blind spots for names outside its training
  distribution — notably under-recognizing some non-Western names, which matters for a
  manga-focused tool. The character profiles feature is the current mitigation.
- Cross-page character identity consistency (Stage B) is a real, visible improvement over
  text-only conditioning but not perfect — an anime-tuned IP-Adapter or per-character LoRA
  would likely close the remaining gap.
- A LoRA-fine-tuned captioner (`train_captioner.py`, intended to shrink/replace the Stage A
  bridge LLM) is implemented and trains successfully, but doesn't yet have enough harvested
  training data (`data/caption_pairs.jsonl`, growing automatically from normal Stage A use)
  to outperform the LLM it's meant to replace.

## Repository layout

```
manga_pipeline/   core library (schema, stages A-D, backends, registry)
studio/           FastAPI + vanilla JS web UI
stories/          example/test prose and generated Story JSON scripts
registry/         per-story character registries (descriptions, reference image paths)
data/             harvested Stage A training pairs (Phase 3)
output/           generated pages, character portraits, PDFs (gitignored)
models/           trained LoRA adapters (gitignored)
```
