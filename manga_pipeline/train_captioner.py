from __future__ import annotations

import json
import re
from pathlib import Path

from .llm import get_embedder

_TASK_PREFIX = "caption: "

CAMERA_HINTS = [
    "extreme close-up",
    "close-up",
    "medium shot",
    "wide two-shot",
    "wide establishing shot",
    "over-the-shoulder",
    "bird's-eye view",
]


def guess_camera_hint(chunk: str, character_count: int) -> str:
    """Keyword-based fallback, used when neither the bridge LLM's own camera
    suggestion nor a trained captioner's CAMERA line (see
    parse_caption_and_camera) is present or recognized - a small model
    doesn't always follow a response format exactly, and this keeps Stage A
    from failing or defaulting to a flat 'medium shot' every time that
    happens."""
    lowered = chunk.lower()
    if "close" in lowered or "hand" in lowered or "eyes" in lowered:
        return "close-up"
    if character_count >= 2:
        return "wide two-shot"
    if "stood" in lowered or "stands" in lowered or "platform" in lowered or "room" in lowered:
        return "wide establishing shot"
    return "medium shot"


_CAPTION_LEAK_MARKERS = (
    "do not invent",
    "do not include dialogue",
    "do not add characters",
    "under 25 words",
    "describing only the setting",
)

_SCREENPLAY_SLUG_RE = re.compile(
    r"^(?:EXT\.|INT\.|EXT/INT\.|WIDE|MEDIUM|CLOSE[- ]UP)[A-Z0-9 .,'-]*:\s*", re.IGNORECASE
)


def _sanitize_caption(caption: str) -> str:
    """Guards against a known small-model failure mode: echoing its own
    prompt instructions back as if they were part of the caption (found on
    a real manuscript panel - the response literally contained "...Do not
    invent objects, locations, or events not in the passage."). Truncates
    at the first sign of leaked instruction text and backs up to the last
    complete sentence, rather than feeding that text straight through to
    the image generation prompt. Also strips a screenplay-slug-style prefix
    ("EXT. WIDE ESTABLISHING SHOT:") the model sometimes invents.
    """
    cleaned = _SCREENPLAY_SLUG_RE.sub("", caption).strip()

    lowered = cleaned.lower()
    cut = len(cleaned)
    for marker in _CAPTION_LEAK_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    if cut < len(cleaned):
        truncated = cleaned[:cut]
        # back up to the last complete sentence so we don't leave a
        # dangling fragment ("...following a soft pause. One sentence,")
        last_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        cleaned = truncated[: last_end + 1].strip() if last_end != -1 else truncated.strip()

    return cleaned or caption.strip()


_CAMERA_FIELD_RE = re.compile(r"camera:\s*([^\n]*)", re.IGNORECASE)


def parse_caption_and_camera(response: str, chunk: str, character_count: int) -> tuple[str, str]:
    """Parses a "CAPTION: ...\\nCAMERA: ..." response (see _CAPTION_PROMPT in
    story_adapt.py, and the same two-line format a trained captioner is
    fine-tuned to emit - see Captioner.generate) into (caption, camera_hint).
    Shared between the bridge-LLM path and the trained-captioner path so
    both get identical, content-aware camera framing from the same call that
    already produces the caption, instead of the flat keyword heuristic
    guess_camera_hint() alone.

    Defensive by design: a small model doesn't always follow the format
    exactly, so an unrecognized or missing camera value falls back to
    guess_camera_hint() rather than ever blocking Stage A or accepting a
    hallucinated camera term. Notably, a fine-tuned T5-base captioner was
    found (via real generation, not just eval loss) to reliably learn the
    "CAPTION: ... CAMERA: ..." field order but not the literal newline
    between them - it collapses both onto one line. Matching "camera:"
    anywhere in the response (not just at the start of its own line) so that
    real, correct model output isn't silently discarded in favor of the
    heuristic fallback.
    """
    match = _CAMERA_FIELD_RE.search(response)
    camera_hint: str | None = None
    remainder = response
    if match:
        candidate = match.group(1).strip().lower().rstrip(".")
        if candidate in CAMERA_HINTS:
            camera_hint = candidate
        remainder = response[: match.start()] + response[match.end() :]

    caption_parts: list[str] = []
    for line in remainder.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("caption:"):
            caption_parts.append(line.split(":", 1)[1].strip())
        else:
            caption_parts.append(line)

    caption = _sanitize_caption(" ".join(caption_parts).strip() or response.strip())
    if camera_hint is None:
        camera_hint = guess_camera_hint(chunk, character_count)
    return caption, camera_hint


def load_pairs(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def filter_grounded(records: list[dict], min_similarity: float = 0.35) -> list[dict]:
    """Drop pairs whose target caption has little embedding similarity to its
    source input - a cheap filter against the bridge LLM's hallucinated
    captions (observed in ~1/3 of harvested pairs), so the LoRA doesn't learn
    to imitate ungrounded output.
    """
    if not records:
        return []
    embedder = get_embedder()
    inputs = [r["input"] for r in records]
    targets = [r["target"] for r in records]
    input_emb = embedder.encode(inputs, normalize_embeddings=True)
    target_emb = embedder.encode(targets, normalize_embeddings=True)
    kept = []
    for record, in_emb, tgt_emb in zip(records, input_emb, target_emb):
        similarity = float(in_emb @ tgt_emb)
        if similarity >= min_similarity:
            kept.append(record)
    return kept


def to_examples(records: list[dict]) -> list[dict]:
    """Builds the (source, target) training pairs. target is the same
    two-line "CAPTION: ...\\nCAMERA: ..." format the bridge LLM prompt uses
    (see parse_caption_and_camera) - a record without a "camera" field
    (pre-camera-harvest data) falls back to guess_camera_hint() over its raw
    "input" text so every example still trains a well-formed two-line
    target, rather than teaching the model to sometimes emit a blank
    CAMERA: line.
    """
    examples = []
    for r in records:
        characters = ", ".join(r.get("characters", [])) or "none"
        source = f"{_TASK_PREFIX}characters: {characters}\n{r['input']}"
        camera = r.get("camera") or guess_camera_hint(r["input"], len(r.get("characters", [])))
        target = f"CAPTION: {r['target']}\nCAMERA: {camera}"
        examples.append({"source": source, "target": target})
    return examples


def train(
    dataset_path: str | Path,
    output_dir: str | Path,
    base_model: str = "google-t5/t5-base",
    epochs: int = 8,
    min_similarity: float = 0.35,
) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    records = load_pairs(dataset_path)
    kept = filter_grounded(records, min_similarity=min_similarity)
    examples = to_examples(kept)
    if len(examples) < 4:
        raise ValueError(f"only {len(examples)} grounded examples after filtering (of {len(records)}) - need more data")

    dataset = Dataset.from_list(examples).train_test_split(test_size=0.15, seed=0)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        model_inputs = tokenizer(batch["source"], max_length=192, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=96, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["source", "target"])

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-3,
        logging_steps=5,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    metrics = trainer.evaluate()

    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return {
        "total_pairs": len(records),
        "grounded_pairs": len(kept),
        "train_examples": len(tokenized["train"]),
        "eval_examples": len(tokenized["test"]),
        "eval_loss": metrics.get("eval_loss"),
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
    }
