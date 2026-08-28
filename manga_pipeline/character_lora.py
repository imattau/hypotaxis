from __future__ import annotations

import re
from pathlib import Path


def sanitize_adapter_name(name: str) -> str:
    """A character name as a filesystem/diffusers-adapter-name-safe token.
    diffusers' load_lora_weights(adapter_name=...) and set_adapters() key
    adapters by this string, and it's also used to build the on-disk
    training-image/output directory name - so it needs to survive both
    without collisions between two characters whose names differ only in
    punctuation ("Jules" vs "Jules'").
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "character"


# Short, varied framing/expression phrases for the bootstrap training set
# (see DiffusersBackend.generate_character_lora_images). A LoRA trained on
# near-identical images of one static pose tends to bake that pose in
# rather than learning the character's identity independent of it - the
# standard DreamBooth mitigation is instance-image variety, which this
# project has to synthesize itself (bootstrapping from the single existing
# IP-Adapter reference-image prompt) since there's no photo set to draw on
# for a character that only exists in prose.
TRAINING_VIEW_PROMPTS = [
    "front-facing portrait, plain background",
    "three-quarter view portrait, plain background",
    "profile view, plain background",
    "close-up on face, neutral expression, plain background",
    "medium shot, standing, plain background",
    "slight smile, plain background",
    "looking slightly to the side, plain background",
    "candid pose, plain background",
]


def build_training_caption(name: str, description: str, style_prompt: str, view_prompt: str) -> str:
    """Per-image training caption. Deliberately mirrors _reference_image()'s
    existing reference-portrait prompt shape (style + 'character reference
    portrait of {name}' + description) rather than inventing a separate
    trigger-token scheme: the character's own name is already the token
    that will appear in real inference prompts (Stage A's captions
    naturally mention characters by name), so training on that exact
    phrasing means no prompt rewriting is needed at generation time for the
    LoRA to activate.
    """
    parts = [p for p in (style_prompt, f"character reference portrait of {name}", description, view_prompt) if p]
    return ", ".join(parts)


def default_lora_output_dir(models_dir: str | Path, story_id: str, name: str) -> Path:
    return Path(models_dir) / "character_loras" / story_id / sanitize_adapter_name(name)


def build_character_training_metadata(
    *,
    story_id: str,
    character: str,
    checkpoint: str,
    rank: int,
    steps: int,
    resolution: int,
    examples: int,
) -> dict[str, object]:
    """Return reproducible metadata for a character-LoRA training run."""

    if not story_id.strip() or not character.strip() or not checkpoint.strip():
        raise ValueError("story_id, character, and checkpoint must be non-empty")
    if rank <= 0 or steps <= 0 or resolution <= 0 or examples <= 0:
        raise ValueError("rank, steps, resolution, and examples must be positive")
    return {
        "method": "character-lora",
        "story_id": story_id,
        "character": character,
        "base_model": checkpoint,
        "rank": rank,
        "steps": steps,
        "resolution": resolution,
        "examples": examples,
    }


def train_character_lora(
    image_paths: list[Path],
    captions: list[str],
    checkpoint: str,
    output_dir: str | Path,
    rank: int = 8,
    steps: int = 300,
    learning_rate: float = 1e-4,
    resolution: int = 768,
    device: str = "auto",
    seed: int = 0,
    on_progress=None,
) -> dict:
    """Minimal SDXL DreamBooth-style LoRA trainer for one character.

    Deliberately narrow in scope compared to a general-purpose DreamBooth
    script: UNet-only LoRA (no text-encoder LoRA), batch size 1, no
    validation loop, no learning-rate warmup/scheduling. This keeps the
    implementation small and its failure surface understandable, at some
    cost to final quality versus a full-featured trainer - reasonable for
    a first character-LoRA pass on a handful of bootstrapped images rather
    than a large curated dataset.

    image_paths/captions: parallel lists, one caption per training image
    (see build_training_caption) - a fixed per-image caption rather than
    per-step-random captioning, since each bootstrapped image was itself
    generated from one specific view_prompt and should be captioned to
    match what's actually in it.

    Precomputes text embeddings for every (image, caption) pair once up
    front rather than re-encoding every step - the text encoders are frozen
    throughout training, so their output for a fixed caption never changes,
    and skipping the redundant re-encode noticeably speeds up a run at this
    scale (hundreds of steps over a handful of captions).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionXLPipeline
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image

    from .config import resolve_device

    if len(image_paths) != len(captions):
        raise ValueError("image_paths and captions must be the same length")
    if not image_paths:
        raise ValueError("need at least one training image")

    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    resolved_device = resolve_device(device)
    use_fp16 = resolved_device.startswith("cuda")
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    report("loading base model")
    pipe = StableDiffusionXLPipeline.from_pretrained(checkpoint, torch_dtype=weight_dtype)
    pipe.to(resolved_device)

    vae, unet = pipe.vae, pipe.unet
    # SDXL's VAE is numerically unstable in fp16 (known upstream issue) -
    # keep it fp32 regardless of the rest of the model, and cast the
    # resulting latents back to weight_dtype afterward
    vae.to(dtype=torch.float32)
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    if use_fp16:
        # the frozen base UNet stays fp16, but the trainable LoRA params
        # must not - fp16 AdamW state on such a small number of params
        # blows up within one or two optimizer steps (confirmed empirically:
        # loss goes to NaN by step 2 without this), a known fp16-training
        # pitfall diffusers' own training scripts work around the same way
        cast_training_params(unet, dtype=torch.float32)
    unet.enable_gradient_checkpointing()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    generator = torch.Generator(device=resolved_device).manual_seed(seed)

    report("preprocessing training images")
    latents_per_image = []
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        pixel_values = tensor.unsqueeze(0).to(resolved_device, dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample(generator=generator)
            latents = latents * vae.config.scaling_factor
        latents_per_image.append(latents.to(dtype=weight_dtype))

    report("encoding captions")
    embeds_per_caption = []
    with torch.no_grad():
        for caption in captions:
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=caption,
                device=resolved_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            embeds_per_caption.append((prompt_embeds, pooled_prompt_embeds))

    add_time_ids = pipe._get_add_time_ids(
        original_size=(resolution, resolution),
        crops_coords_top_left=(0, 0),
        target_size=(resolution, resolution),
        dtype=weight_dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(resolved_device)

    scheduler = pipe.scheduler
    n = len(image_paths)

    unet.train()
    report(f"training ({steps} steps over {n} images)")
    losses: list[float] = []
    for step in range(steps):
        idx = step % n
        latents = latents_per_image[idx]
        prompt_embeds, pooled_prompt_embeds = embeds_per_caption[idx]

        noise = torch.randn(latents.shape, generator=generator, device=resolved_device, dtype=latents.dtype)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (1,), device=resolved_device, generator=generator
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
        ).sample

        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"unsupported scheduler prediction_type: {scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if (step + 1) % max(1, steps // 10) == 0:
            report(f"step {step + 1}/{steps}, loss {loss.item():.4f}")

    unet.eval()
    lora_state_dict = get_peft_model_state_dict(unet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionXLPipeline.save_lora_weights(str(output_dir), unet_lora_layers=lora_state_dict)

    report("done")
    return {
        "output_dir": str(output_dir),
        "steps": steps,
        "images": n,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
    }
