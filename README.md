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

## Extracting cast sheets from long novels with a large LLM

The three profile mechanisms above (character/location/prop) exist specifically to sidestep
automatic-detection weak points: spaCy's NER has blind spots for names outside its training
distribution (notably under-recognizing some non-Western names - see "Project status" below),
there's no coreference resolution across a long span, and locations/props have no automatic
detection at all. Those gaps compound with length - a short story might only ever mention a
character by one name, but a full novel will use a name, a nickname, a title, and a bare
pronoun across hundreds of pages, and NER alone won't reliably tie those back to one canonical
person the way `_merge_person_aliases` needs to.

A large LLM (whatever you have access to - Claude, GPT-4-class, etc.) is a good fit for
producing these profile files as a one-time **offline authoring step**, before the novel ever
reaches hypotaxis - not as a runtime dependency of the pipeline itself. Have it read the whole
manuscript and write out `character_profiles.txt`/`location_profiles.txt`/`prop_profiles.txt`
in the plain `Name: description` format `stories/*.example.txt` shows, resolving name variants
to one canonical form per character (the form spaCy is most likely to catch, or whichever you
prefer - `_merge_person_aliases` still handles minor variants found in-text, but starting from
a clean canonical list removes most of the burden), tagging any bodiless/non-physical entity
with `[no-form]`, and noting settings/objects that recur often enough to be worth a profile
rather than just prose description. This is squarely a "read a lot of text, extract a
structured list" task a large model is well suited for, and doing it once per novel is a small,
bounded cost regardless of model size or where it runs.

What this deliberately isn't: a large LLM rewriting or simplifying the manuscript's prose
before it reaches Stage A. Stage A's caption model (bridge LLM or the trained captioner) is
tuned against real prose text, not a pre-summarized version of it, and rewriting would risk
introducing a second layer of hallucination on top of whatever the summarization step gets
wrong - the cast-sheet extraction above targets exactly the gap that's actually there (canonical
identity across a long span) rather than changing what the rest of the pipeline reads.

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

## Pose ControlNet (experimental, opt-in)

Identity conditioning (IP-Adapter, Character LoRA above) is skipped entirely for any panel
tagged with 2+ characters - blending several identities into one panel is a separate unsolved
problem (see `DiffusersBackend._resolve_target`). That gap turned out to have a second,
independent symptom worth its own fix: without any identity conditioning steering it, SDXL
would routinely drop or duplicate figures in such a panel - a "wide two-shot" tagged for two
characters coming out as four indistinct figures, or as one. No amount of prompt engineering
fixed this reliably (tested for real: explicit "exactly two people, nobody else in frame"
wording made no measurable difference across repeated generations) - explicit numeric/counting
instructions are a well-known weak point for SDXL-class models, distinct from the compositional
framing language that *did* fix camera-hint adherence (see "Camera hint prompt expansion"
below).

`use_pose_controlnet` (off by default) fixes this with real structural conditioning instead of a
prompt hope: `manga_pipeline/pose_skeleton.py` generates a synthetic OpenPose skeleton image -
not ML pose estimation, since there's no source photo to estimate a pose from, but a
deterministic, evenly-spaced standing-figure skeleton with exactly as many figures as the panel
has tagged characters, drawn in the exact keypoint/color convention `controlnet_aux` (and the
OpenPose-trained ControlNet checkpoint) expects.

**This substantially improves headcount reliability, but it is not a guarantee** - worth being
precise about, since an earlier draft of this section overclaimed one. Across real generations
at the default `pose_controlnet_scale=0.5`, roughly 5 in 6 came out with the right figure count;
the rest still dropped or lost a figure, the same failure mode this feature exists to fix, just
less often. Raising the scale to compensate was tried and made things *worse*, not better: at
0.7, one previously-good seed now rendered zero figures at all - a new, more severe failure mode
that hadn't existed at 0.5. Conditioning strength isn't a dial that moves monotonically toward
"more reliable" here, so 0.5 (the better-tested value) is the default, not a stronger-sounding
number that turned out not to help. A retry-with-detection approach (verify the output with a
real OpenPose detector, regenerate with a different seed if the count comes up short) was also
tried and didn't move the measured success rate either - it traded which specific case failed
for another at ~3x the generation cost, so it isn't part of this feature. Net: this is a real,
large improvement over prompt-only phrasing (which was reliably wrong, not occasionally), but
still probabilistic, not absolute - budget for the occasional wrong headcount even with this on.

Real cost: the ControlNet checkpoint (`thibaud/controlnet-openpose-sdxl-1.0`) is trained
against full SDXL base, not the lighter `sdxl-turbo` checkpoint this project defaults to
elsewhere, so this loads a second, separate SDXL pipeline the first time a panel actually needs
it (`DiffusersBackend._load_pose_pipe`) - real extra VRAM, and full SDXL base's normal
(non-turbo) 30 steps/8.5 guidance run at roughly 7-8x the per-image time of the rest of the
pipeline. Only panels with 2+ tagged characters ever take this path; everything else is
unaffected. `_load_pose_pipe` uses `enable_model_cpu_offload` rather than a plain `.to(device)`
for the same reason `_load` already does for `_base_pipe` under `use_identity_adapter` - a real
run hit CUDA OOM with both pipes fully resident on a 16GB card before this was added.

`pose_controlnet_scale` also trades headcount/position reliability against how much the shared
text prompt gets to drive each figure's actual appearance - a separate axis from the reliability
question above. 0.65 locked position more tightly in one real comparison but rendered both
figures as visually similar girls, ignoring that the prompt named one male and one female
character; 0.5 differentiated the two by the text prompt correctly. That's still a real, open
limitation on top of the headcount one: this only constrains *how many* figures and roughly
*where* - not *who's who*. Nothing here assigns a specific character's identity/LoRA to a
specific skeleton position, so a multi-character panel still can't get the same strong
per-character likeness lock a single-character panel gets from IP-Adapter/Character LoRA above.

## Camera hint prompt expansion

`panel.camera_hint` (Stage A's content-aware shot framing) reaches the real SDXL prompt via
`_build_prompt` in `manga_pipeline/backends.py`, but real generation comparisons found the
production `sdxl-turbo` checkpoint mostly ignored the terse 2-3 word form of four of the seven
hints: "wide two-shot" and "wide establishing shot" both rendered as an ordinary close/medium
two-shot, "bird's-eye view" produced no overhead angle at all, and "over-the-shoulder" rendered
as a flat frontal two-shot. Neither more inference steps nor switching to full SDXL base at
standard settings reliably fixed this alone (the checkpoint swap helped over-the-shoulder
somewhat, at the same ~7-8x per-panel cost noted above, but left the other three largely
unchanged). What did work: spelling out what the shot actually contains - room extent, camera
position, what's blurred vs. sharp - rather than the short label alone.
`_CAMERA_HINT_PROMPT_EXPANSIONS` in `backends.py` expands three of the four hints into this
longer phrasing at prompt-build time only (the short form stays what `panel.camera_hint`
actually holds, and what the trained captioner is trained to predict); this fixed
wide-two-shot, wide-establishing-shot, and bird's-eye-view outright on the unmodified
production checkpoint. Over-the-shoulder only partially improved (both faces turn to profile,
but no real foreground shoulder/head occlusion), and pushing the phrasing further overcorrected
into a single-subject blurred close-up that lost the second character and drifted off the manga
line-art style - so it's left at the milder, imperfect phrasing rather than chasing a further
fix.

A related, separate fix: page layout selection (`pack_into_pages` in `story_adapt.py`) used to
pick a page's panel layout by panel count alone, blind to camera framing - a real page-
generation test found a "wide two-shot" panel landed in an `H3` layout's narrow vertical strip
and rendered as a single figure standing at a window instead of the two people the prompt asked
for, even with the prompt expansion above working correctly. `is_wide_box` in `layouts.py` and
`_layout_fits`/`_WIDE_CAMERA_HINTS` in `story_adapt.py` now keep `pack_into_pages` from handing
a wide-shot panel a layout box that's physically too narrow to hold it, falling back to the old
any-template-of-this-count behavior only when no template of that panel count has a suitable
box at all.

Another real `pack_into_pages` bug, noticed from actually reading a generated multi-page story
rather than a single test page: it always produced 3-panel pages. `_SUPPORTED_COUNTS = [3, 4, 2,
9]` was tried in that fixed order on every page, and since `3 <= remaining` is true on nearly
every iteration for a story of any real length, 3 won essentially every time - `G22`/`H13`/`H31`
(4-panel) and `G33` (9-panel) layouts were effectively unreachable outside a trailing remainder.
`pack_into_pages` now rotates which count is tried first per page instead of always using the
same fixed order, while staying fully deterministic (the same story always paginates the same
way) - a 15-panel story that used to come out as five uniform 3-panel pages now varies
(`[3, 4, 2, 3, 3]` in one real run).

## Speech bubbles: SVG templates with face-anchored tails

Dialogue bubbles used to be plain PIL primitives (`rounded_rectangle`, `ellipse`) pinned to a
fixed offset from each panel's own top-left corner - no speech tail pointing at whoever's
talking, no thought-bubble trail, and every bubble in the same fixed corner regardless of where
the speaking character actually ended up in the generated art. `manga_pipeline/bubbles.py` now
renders bubbles as SVG templates (`cairosvg`) - a proper rounded-rect body with a triangular
tail for speech, an ellipse with a shrinking circle trail for thought, a plain box for
narration - composited onto the page, with the bubble's text still drawn via PIL on top (kept
separate deliberately: PIL's `wrap_to_width`/`textbbox` measurements are what the bubble's own
size is computed from, so keeping text rendering unchanged avoids any font-metric mismatch
between two renderers).

The tail points at an actual detected face, not a guess: `manga_pipeline/face_detect.py` reuses
the same OpenPose body detector already validated for pose-ControlNet headcount verification
(`controlnet_aux`'s `OpenposeDetector`) to find each panel's speaking character's face,
anchoring on the nose keypoint (the 18-point body format has no separate mouth/lip landmark, and
nose is close enough for "point the tail at this character" without a second facial-landmark
model). Real testing on actual generated panels found it precisely accurate whenever it detects
a face at all - every detected point landed right on a real face across multiple real test
images - but with the same recall gap already documented for the pose-ControlNet verification
use of this detector: it's trained on photos, not manga/anime line art, and missed several
real, clearly visible faces in one real test image. A panel where nothing is detected (or a
dialogue line with no anchor to cycle to) still renders correctly - just as a tail-less bubble
at its previous default position, exactly the old behavior, rather than guessing a direction
that might point at nothing. There's also no attempt to match a specific dialogue line's
speaker to a specific detected face - multiple anchors in one panel are just cycled through in
left-to-right order, the same "no real per-character identity match from pixels alone"
limitation already true of pose-ControlNet's multi-figure panels.

A fourth bubble style beyond the three `DialogueLine.kind`s (speech/thought/narration): a jagged
burst shape for shouted/exclaimed speech, `_shout_outline_points` in `bubbles.py` - a
deterministic zigzag polygon (alternating outer/inner radius around an ellipse), not hand-drawn
art. Nothing upstream currently tags dialogue with an intensity signal, so this is decided by a
cheap heuristic on the line's own text (`_is_shouted`): a trailing `!`, or the line being
substantially uppercase. Only ever applies to `speech`-kind lines - thought and narration keep
their own fixed shape regardless of text content.

## Per-panel generation vs. a shared page image

Each panel is generated independently (`DiffusersBackend.generate_panel`), sized to its own
layout box, rather than generating one shared "page" image and deriving each panel from it. An
earlier version of this project tried the shared-image approach and reverted it: panels on the
same page can have very different aspect ratios (a thin `V3` strip vs. a wide box from "Camera
hint prompt expansion" above), and warping one image into each via a plain resize visibly
stretched, squashed, and repeated the same content across panels. Cross-panel consistency
instead rests on identity conditioning (IP-Adapter, Character LoRA) plus shared prompt wording -
real, but not the same thing as literally sharing pixels between panels.

Outpainting was explored as a way to get shared-pixel consistency without the resize distortion:
generate one base image, then extend its canvas into each panel's own box shape instead of
stretching it. Two real attempts (`sdxl-turbo` and the dedicated
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` checkpoint, both at standard inpainting
settings) both failed the same way - the extended canvas came back essentially blank instead of
generating new content. Likely cause, not yet confirmed: filling the to-be-outpainted region
with plain white before masking/encoding, a known sensitivity of diffusers inpainting pipelines
that usually calls for a neutral gray or noise fill instead. Not pursued further since this
started as an exploratory question, not a firm feature request - independent per-panel
generation remains what's actually validated to work, and is the only approach that's been
shown to work at all for cross-panel coherence here, not just the one preferred by default.

## Project status

This is an active prototype, not a finished product. Known limitations are tracked
honestly rather than papered over:

- The Stage A caption LLM (3B by default) still sometimes hallucinates content not in the source text on passages with little concrete visual detail, though noticeably less than the 0.5B model it replaced. It also occasionally echoes its own prompt instructions back as if they were caption text (a known small-model failure mode) - a sanitization pass catches and strips the clearest cases (leaked instruction phrases, invented screenplay-style scene slugs) before the caption ever reaches image generation, but this is a guard against the worst outcomes, not a guarantee the model never misbehaves.
- Getting `panel.camera_hint` to actually change SDXL's output (not just reach the prompt
  string) needed real tuning, not just wiring - see "Camera hint prompt expansion" above for
  what was tried, what worked, and what's still imperfect (over-the-shoulder).
- A multi-character panel (2+ tagged characters) has no per-character identity conditioning at
  all - see "Pose ControlNet" above for the headcount/positioning fix and what it still doesn't
  solve (which figure looks like which character).
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
  panel captions. It trains on `data/caption_pairs_curated.jsonl` (see "Curating a clean caption
  dataset" below) on a joint `"CAPTION: ...\nCAMERA: ..."` target (`to_examples()` in
  `train_captioner.py`), so `Captioner.generate()` returns real, content-aware camera framing
  parsed straight out of the model's own output (`parse_caption_and_camera`) - not just the flat
  `guess_camera_hint()` keyword heuristic, though that's still the fallback when the model's
  output doesn't include a recognizable camera value (or for an older adapter trained on
  caption-only targets, whose output never will). Default base model is `google-t5/t5-base`,
  chosen over `t5-small` after a direct comparison at the same dataset size showed a clear
  quality gap (eval loss 1.72 vs. 2.17, and t5-small visibly confusing subjects in
  multi-character sentences that t5-base got right).

  Found via real generation output, not just eval loss: a fine-tuned T5-base captioner reliably
  learns the field order but not the literal newline between `CAPTION:` and `CAMERA:` - it
  collapses both onto one line. `parse_caption_and_camera` matches `camera:` anywhere in the
  response rather than only at the start of its own line, so this doesn't silently discard a
  correct camera prediction in favor of the heuristic.

  To use it: `python train_captioner.py --dataset data/caption_pairs_curated.jsonl` (writes
  to `models/captioner/adapter` by default), then check "Use trained captioner" on the studio's
  New Story form, or pass a `Captioner` instance to `adapt_story()` directly. Character
  descriptions still go through the bridge LLM either way - the captioner was never trained for
  that task. `data/caption_pairs.jsonl`, the older auto-harvested dataset that still grows from
  normal Stage A use, remains unreviewed and untrusted for training (its targets are the bridge
  LLM's own, sometimes-hallucinated output).

## Curating a clean caption dataset

The auto-harvested dataset above trains a LoRA to imitate the bridge LLM's mistakes along with
everything else, since its targets were never reviewed. `curate_dataset.py` generates review
candidates instead: it runs a deliberately *stronger* teacher model (`Qwen/Qwen2.5-7B-Instruct`
by default, loaded in 4-bit so it fits alongside everything else on a modest card) over one or
more story text files and writes `{input, characters, target, camera}` candidates to
`data/caption_candidates.jsonl` - the same shape `train_captioner.py` expects, but not yet
trusted.

```bash
pip install -r requirements-story-adapt.txt -r requirements-training.txt
python curate_dataset.py stories/*.txt
```

Review candidates in the studio's **Dataset** tab: each one shows the source passage, an
editable caption, and a camera dropdown, with Accept (optionally after editing either) or
Reject. Accepted examples go to `data/caption_pairs_curated.jsonl` - the clean dataset
`train_captioner.py` trains on by default. It currently holds 2,010 examples across the full
`stories/*.txt` corpus (337 short original stories, one skipped as too short to segment),
spanning a deliberately wide range of settings and character names - each with a real,
teacher-labeled camera hint rather than the flat keyword heuristic.

This full-corpus batch was curated by an automated accept gate (the same embedding-similarity
grounding filter `train_captioner.py` already applies at training time, `min_similarity=0.35`)
rather than the one-by-one manual review the first 2,000-example set got - reviewing 2,000+
examples by hand isn't practical to redo per dataset change. All 2,010 candidates passed the
0.35 threshold; the raw pre-filter harvest is kept at
`data/caption_candidates_camera_harvested.jsonl` as a queue for a future manual review pass, if
the automated gate's quality bar turns out not to be enough. The previous hand-reviewed
(caption-only, no camera field) dataset is recoverable from git history (the commit that added
it), and the prior captioner adapter is kept locally at `models/captioner_precamera_backup/`
(gitignored, not in version control) for a quick rollback.

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
