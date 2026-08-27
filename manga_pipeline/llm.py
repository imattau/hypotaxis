from __future__ import annotations

from functools import lru_cache

from .config import resolve_device


class SmallLLM:
    """Thin wrapper around a small local instruct model, used only for the
    one inherently generative step in Stage A (prose -> panel caption /
    character description). Kept as small as quality allows, per the
    project's constraint to minimize LLM use for small-hardware users.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto"):
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            dtype=dtype,
            device_map=self.device if self.device.startswith("cuda") else None,
        )

    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        self._load()
        messages = [{"role": "user", "content": prompt}]
        result = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        reply = result[0]["generated_text"][-1]["content"]
        return reply.strip().strip('"').strip()


@lru_cache(maxsize=None)
def get_embedder(model_id: str = "BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)
