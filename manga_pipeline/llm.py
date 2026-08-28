from __future__ import annotations

from functools import lru_cache

from .config import resolve_device


class SmallLLM:
    """Thin wrapper around a small local instruct model, used only for the
    one inherently generative step in Stage A (prose -> panel caption /
    character description). Kept as small as quality allows, per the
    project's constraint to minimize LLM use for small-hardware users.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto", quantize: bool = False):
        self.model_id = model_id
        self.device = resolve_device(device)
        # 4-bit loading - off by default (the small production bridge model
        # doesn't need it), but a larger teacher model used for dataset
        # curation (e.g. Qwen2.5-7B-Instruct, ~14GB in fp16) may not leave
        # enough headroom on a modest card once other processes hold some
        # VRAM too. 4-bit cuts that to ~4-5GB.
        self.quantize = quantize
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_kwargs = {}
        if self.quantize and self.device.startswith("cuda"):
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            dtype=dtype,
            device_map=self.device if self.device.startswith("cuda") else None,
            model_kwargs=model_kwargs,
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
