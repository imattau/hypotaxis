from __future__ import annotations

import json
from pathlib import Path

from .llm import get_embedder

_TASK_PREFIX = "caption: "


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
    examples = []
    for r in records:
        characters = ", ".join(r.get("characters", [])) or "none"
        source = f"{_TASK_PREFIX}characters: {characters}\n{r['input']}"
        examples.append({"source": source, "target": r["target"]})
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
