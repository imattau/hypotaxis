from __future__ import annotations

from pathlib import Path

from .config import resolve_device
from .train_captioner import _TASK_PREFIX


def build_captioner_source(text: str, characters: list[str]) -> str:
    """The exact input-string format train_captioner.py's to_examples()
    builds from a curated {input, characters, target} record. Factored out
    as a pure function (no model involved) so Captioner.generate() and the
    test suite can both use it without drifting apart - a captioner fed
    text in a different shape than it was fine-tuned on will silently
    degrade rather than error, so this format must match exactly.
    """
    return f"{_TASK_PREFIX}characters: {', '.join(characters) or 'none'}\n{text}"


class Captioner:
    """Loads a LoRA-fine-tuned seq2seq model (see train_captioner.py) for
    Stage A's prose -> panel caption step, as a lighter-weight, much faster
    alternative to prompting the full instruct-tuned bridge LLM (SmallLLM)
    for every single panel.

    Deliberately narrow: it only replaces the caption call. Camera framing
    still falls back to story_adapt.py's guess_camera_hint() keyword
    heuristic, since the captioner was fine-tuned purely on caption text
    (see train_captioner.py's to_examples()) and never saw the two-line
    CAPTION/CAMERA format the bridge LLM prompt asks for. Character
    descriptions (_DESCRIPTION_PROMPT) also still go through the bridge
    LLM - the captioner was never trained for that task either.
    """

    def __init__(
        self,
        adapter_dir: str | Path,
        base_model: str = "google-t5/t5-base",
        device: str = "auto",
    ):
        self.adapter_dir = Path(adapter_dir)
        self.base_model = base_model
        self.device = resolve_device(device)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        model = PeftModel.from_pretrained(model, str(self.adapter_dir))
        model.to(self.device)
        model.eval()
        self._model = model

    def generate(self, text: str, characters: list[str]) -> str:
        """text: the chunk (optionally with a trailing "Known appearances:
        ..." note - see story_adapt.py's _caption_input()), matching the
        curated dataset's raw "input" field exactly."""
        self._load()
        import torch

        source = build_captioner_source(text, characters)
        inputs = self._tokenizer(source, return_tensors="pt", truncation=True, max_length=192)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=48, num_beams=4)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
