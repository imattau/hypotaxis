# Manga Production Pipeline

A local pipeline that turns a prose story into a manga/comic — panels, dialogue, character
art — built primarily on small embedding models, transformers, and LoRA adapters rather
than a large LLM, so it stays usable on modest consumer GPU hardware. Local-first: no
required cloud APIs.

## Pipeline stages

| Stage | What it does | Key tech |
|---|---|---|
| **A — Story Adaptation** | Prose → structured Story JSON script (panels, dialogue, camera hints) | sentence embeddings for scene segmentation, LexRank (networkx) for panel-budget compression, spaCy NER + dependency parsing for characters/speaker attribution, a small local LLM (Qwen2.5-3B-Instruct by default) for panel captions and content-aware camera framing |
| **B — Character/Location/Prop Identity** | Persistent visual identity for characters and recurring locations; consistent wording for recurring props | characters/locations: text registry + IP-Adapter reference images, two simultaneous slots so a panel can be conditioned on a character and a setting at once, plus an optional trained per-character LoRA for a stronger identity lock (see "Character LoRA" below). Props: text-anchored into the generation prompt only — image-conditioning a whole panel on a close-up of a small object distorts the rest of the composition |
| **C — Generation** | Each panel generated independently at its own aspect ratio, then composed into a page | SDXL / SDXL-Turbo via `diffusers`, text2img per panel |
| **D — Assembly** | Panels → laid-out page → dialogue bubbles → PDF | Pillow, deterministic, no model |

A web studio UI (FastAPI + vanilla JS) sits on top for running the whole flow visually.

Stage A respects a manuscript's own structure rather than treating it as one undifferentiated
block of prose: markdown scene breaks (`---`, `***`, `___` alone on a line) are hard panel
boundaries no single panel can straddle, each independently panel-budgeted by its share of the
chapter's word count; chapter headings (`# ...`) and blockquote markers (`> `) are stripped
before segmentation so they don't pollute a caption or a character-name detection pass.

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
pip install -r requirements-dev.txt          # pytest, for running tests/
```

On Debian/Ubuntu-style systems with an externally-managed Python, add
`--user --break-system-packages` to the `pip install` calls, or use a virtualenv instead.

## Running tests

```bash
pip install -r requirements-story-adapt.txt -r requirements-dev.txt
python -m spacy download en_core_web_sm
pytest
```

Covers the pure-logic pieces of Stage A/B/D (dialogue speaker attribution, NER
alias merging, panel-budget packing, bubble sizing/overflow, identity-conditioning
policy) - no GPU or model download beyond spaCy's small English model is needed.
Most of these tests were written directly from real bugs found while testing the
pipeline against an actual manuscript, so they double as a changelog of past
regressions.

## Quick start

**Studio UI** (recommended way to explore the pipeline):

```bash
python run_studio.py --port 8420
```

Open `http://127.0.0.1:8420`, create a new story (paste text or upload a `.txt`/`.md`/`.docx`
chapter), optionally add a [character profile](#character-profiles) cast sheet, then generate
pages with the `mock` backend (instant, no GPU) or `diffusers` (real generation).

Stage A's output (captions, camera hints, character/location/prop tags, dialogue lines) is
generated, not guaranteed — an "Edit" button on every panel in the script view lets you
correct it by hand (wrong caption, misattributed speaker, a tag that should or shouldn't be
there) without re-running the whole adaptation step. Character/location/prop sheets also have
a delete button per entry, for pruning a name automatic detection got wrong.

Page generation is resumable: a page whose image already exists on disk is reused rather
than redrawn, so a job that stops partway (an out-of-memory error, a crash) can be continued
by clicking "Generate Pages" again instead of redoing every page from the start. Check
"Regenerate existing pages too" to force a clean redo instead.

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

A character with no physical/humanoid body at all — an AI, a voice, a presence — can
start their description with the tag `[no-form]` (e.g. `Nova: [no-form] no physical
body; visualise only as...`). This skips reference-portrait generation and IP-Adapter
identity conditioning for them entirely and text-anchors their description into the
prompt instead, the same way props are handled — the normal reference-portrait template
forces a human figure regardless of what the description says, so without this tag a
character explicitly written as bodiless still gets rendered (and IP-Adapter-conditioned)
as a person.

## Location profiles

Same format and mechanism as character profiles (`stories/location_profiles.example.txt`,
`--location-profiles <path>`), but for recurring settings. Unlike characters, locations have
**no automatic detection at all** — a generic-but-important setting like "the abandoned mill"
isn't a named entity the way a person's name is — so a location only ever appears in the
story if it's listed in a profile. Once registered, a location gets its own reference image
and identity conditioning, running simultaneously alongside whichever character is in the
same panel (two IP-Adapter slots active at once).

## Prop profiles

Same file format again (`stories/prop_profiles.example.txt`, `--prop-profiles <path>`), for
small portable objects — a letter, a key, a sword. Props are **not** handled like locations:
testing showed that generating an image reference the same way (a wide establishing shot)
produces a cluttered, unusable scene for a small object, and conditioning a whole panel's
image on a close-up of one incidental item would distort the rest of the composition around
it. So a prop is only ever **text-anchored** — its description is woven into the generation
prompt on exactly the panels it's tagged on, the same trick already used for character
appearance notes. This also means a location and a prop can appear in the same panel without
conflict, since text-anchoring doesn't compete for an IP-Adapter slot.

## Character LoRA (experimental, opt-in)

IP-Adapter identity conditioning (above) is a lightweight, always-available fallback, but it's
a soft nudge rather than a strong identity lock. `train_character_lora.py` trains a small
rank-8 SDXL LoRA on the UNet for one character, giving a noticeably stronger, more consistent
likeness once trained - complementary to IP-Adapter, not a replacement for it.

There's no photo set to train from - a character only exists in prose - so the trainer
bootstraps its own training images: it generates a handful of portraits of the character
across several fixed camera angles/expressions (`manga_pipeline/character_lora.py`'s
`TRAINING_VIEW_PROMPTS`) using the same base checkpoint, then trains a LoRA on those.

```bash
python train_character_lora.py <story_id> "<character name>" \
  --style-prompt "monochrome manga, screentone shading" \
  --checkpoint stabilityai/sdxl-turbo
```

This needs a real GPU and takes real wall-clock time - a few minutes per character on a 16GB
card (RTX 5060 Ti), most of it fixed overhead (loading the base checkpoint, generating the
bootstrap images) rather than the training loop itself, which runs at roughly 0.27s/step. It's
not something to run casually per character. Also available from the studio: a "Train LoRA"
button on each character card (Stage B cast view), running as a background job the same way
page generation does. The trained adapter is saved to
`models/character_loras/<story_id>/<character>/` and recorded on the character's registry
entry; `generate_panel()` picks it up automatically once `use_character_lora` is turned on
(a per-generation toggle, off by default - a story with nothing trained yet behaves exactly as
before). Camera framing and character descriptions are unaffected either way - the LoRA only
ever influences the panel's rendered likeness.

Default step count (300) was chosen from a real comparison, not guessed: 250 and 500 steps were
both trained end-to-end on real story characters and compared against the IP-Adapter-only
baseline across three different scenes/seeds each. Both step counts produced a visibly more
consistent likeness than baseline (eye shape, nose, eyebrow style, and face silhouette all read
as clearly "the same person" across different poses, where the baseline set showed more
per-panel drift in those same features) - since the gap between 250 and 500 steps looked
small relative to the fixed per-run overhead, 300 was picked as a reasonable default rather
than defaulting to the more expensive end. `--rank` (8) and `--bootstrap-count` (8) weren't
separately tuned - 8 is a conventional default for a small character LoRA, not verified against
alternatives here.

Implementation note: this is a deliberately minimal DreamBooth-style trainer (UNet LoRA only,
no text-encoder LoRA, batch size 1, fixed per-image captions) rather than a full-featured one -
reasonable for a first pass on a handful of bootstrapped images, at some cost to final quality
versus a larger, more careful setup. One real bug worth flagging for anyone extending this:
training the LoRA parameters in fp16 (matching the frozen base model) produces NaN loss within
1-2 optimizer steps - fp16 AdamW state on a small parameter count is numerically unstable. The
fix (already applied) is `diffusers.training_utils.cast_training_params(unet, dtype=torch.float32)`
right after `add_adapter()`, keeping the LoRA weights in fp32 while the frozen base stays fp16.

## Project status

This is an active prototype, not a finished product. Known limitations are tracked
honestly rather than papered over:

- The Stage A caption LLM (3B by default) still sometimes hallucinates content not in the source text on passages with little concrete visual detail, though noticeably less than the 0.5B model it replaced. It also occasionally echoes its own prompt instructions back as if they were caption text (a known small-model failure mode) - a sanitization pass catches and strips the clearest cases (leaked instruction phrases, invented screenplay-style scene slugs) before the caption ever reaches image generation, but this is a guard against the worst outcomes, not a guarantee the model never misbehaves.
- Dialogue speaker attribution can't resolve pure-pronoun cases ("he called out") without
  coreference resolution, which isn't wired in (the well-known libraries for this —
  `coreferee`, `spacy-experimental`, `BookNLP` — are all currently incompatible with a
  modern Python/spaCy/transformers stack; a character profile cast sheet sidesteps this
  for named characters).
- spaCy's NER (character detection) has blind spots for names outside its training
  distribution — notably under-recognizing some non-Western names, which matters for a
  manga-focused tool. The character profiles feature is the current mitigation.
- Markdown-italicized text (`*like this*`) is treated as an internal thought (rendered as a
  thought bubble instead of a speech bubble), attributed via the same speaker-resolution
  pipeline as quoted dialogue. Prose italicizes plenty of things that aren't a thought too —
  a title, reported speech, emphasis — so this is a heuristic, not a guarantee; a length
  filter (3+ words) cuts the clearest false positives (single emphasized words, short
  titles) but can't tell a genuine thought from italicized reported speech.
- Cross-page character identity consistency (Stage B) is a real, visible improvement over
  text-only conditioning but not perfect with IP-Adapter alone. A per-character LoRA (see
  "Character LoRA" above) closes more of the gap once trained, but it's opt-in, one-character-
  at-a-time, and real GPU time per character - not a default anyone gets for free.
- A LoRA-fine-tuned captioner (`train_captioner.py`, `manga_pipeline/captioner.py`) is
  implemented, trained, and wired into Stage A as an opt-in alternative to the bridge LLM for
  panel captions. It trains on `data/caption_pairs_curated.jsonl`, a 2,000-example
  human-reviewed dataset built via `curate_dataset.py` (see "Curating a clean caption dataset"
  below). Default base model is `google-t5/t5-base`, chosen over `t5-small` after a direct
  comparison at the same dataset size showed a clear quality gap (eval loss 1.72 vs. 2.17,
  and t5-small visibly confusing subjects in multi-character sentences that t5-base got right).
  To use it: `python train_captioner.py --dataset data/caption_pairs_curated.jsonl` (writes
  to `models/captioner/adapter` by default), then check "Use trained captioner" on the studio's
  New Story form, or pass a `Captioner` instance to `adapt_story()` directly. The captioner only
  replaces caption text - camera framing still uses the `guess_camera_hint()` keyword heuristic
  in this path (the captioner was never trained to output it), and character descriptions still
  go through the bridge LLM either way. `data/caption_pairs.jsonl`, the older auto-harvested
  dataset that still grows from normal Stage A use, remains unreviewed and untrusted for
  training (its targets are the bridge LLM's own, sometimes-hallucinated output).

## Curating a clean caption dataset

The auto-harvested dataset above trains a LoRA to imitate the bridge LLM's mistakes along with
everything else, since its targets were never reviewed. `curate_dataset.py` generates review
candidates instead: it runs a deliberately *stronger* teacher model (`Qwen/Qwen2.5-7B-Instruct`
by default, loaded in 4-bit so it fits alongside everything else on a modest card) over one or
more story text files and writes `{input, characters, target}` candidates to
`data/caption_candidates.jsonl` - the same shape `train_captioner.py` expects, but not yet
trusted.

```bash
pip install -r requirements-story-adapt.txt -r requirements-training.txt
python curate_dataset.py stories/*.txt
```

Review candidates in the studio's **Dataset** tab: each one shows the source passage and an
editable caption, with Accept (optionally after editing) or Reject. Accepted captions go to
`data/caption_pairs_curated.jsonl` - the clean dataset `train_captioner.py` trains on by
default. It currently holds 2,000 reviewed examples across ~250 short original stories
(`stories/*.txt`), spanning a deliberately wide range of settings and character names.

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
