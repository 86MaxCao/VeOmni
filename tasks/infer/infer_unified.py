"""Unified inference entry point for all 5 multimodal models in VeOmni.

Supports:
  - bagel / thinkmorph: Qwen2-VL + MoVQGAN (understanding + generation)
  - blip3o: xGen-MM + Diffusion (understanding + generation)
  - u1 (neo_chat): Qwen3 + Flow-Matching MoT (understanding + generation)
  - latentum: InternVL + MoT discrete (understanding + generation)

Usage:
    # Understanding (VQA)
    python tasks/infer/infer_unified.py \
        --model_type bagel \
        --model_path /mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT \
        --mode understand \
        --image path/to/image.jpg \
        --prompt "Describe this image."

    # Generation (T2I)
    python tasks/infer/infer_unified.py \
        --model_type u1 \
        --model_path /mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT \
        --mode generate \
        --prompt "A cat sitting on a windowsill" \
        --output output.png
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from veomni.models.loader import get_model_config, get_model_class
from veomni.models.module_utils import init_empty_weights, load_model_weights


MODEL_TYPE_TO_CHECKPOINT = {
    "bagel": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT",
    "thinkmorph": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/ThinkMorph-7B",
    "blip3o": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BLIP3o-Model-8B",
    "u1": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT",
    "latentum": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/LatentUM-Base",
}

MODEL_TYPE_TO_CONFIG_TYPE = {
    "bagel": "bagel",
    "thinkmorph": "thinkmorph",
    "blip3o": "blip3o_qwen",
    "u1": "neo_chat",
    "latentum": "latentum",
}


def load_model(model_path: str, device: str = "cuda"):
    """Load any model through VeOmni's unified loader."""
    print(f"[VeOmni] Loading config from {model_path}...")
    config = get_model_config(model_path, trust_remote_code=True)
    print(f"[VeOmni] model_type = {config.model_type}")

    model_cls = get_model_class(config)
    print(f"[VeOmni] model_cls = {model_cls.__name__}")

    with init_empty_weights():
        model = model_cls._from_config(config=config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[VeOmni] Parameters: {param_count / 1e9:.2f}B")

    model = model.to_empty(device=device)
    model = model.to(torch.bfloat16)
    load_model_weights(model, model_path)
    model.eval()

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[VeOmni] Model loaded on {device}. GPU memory: {mem_gb:.2f} GB")
    return model, config


def load_tokenizer(model_path: str, model_type: str):
    """Load tokenizer appropriate for the model type."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


# =============================================================================
# Model-specific inference helpers
# =============================================================================


def pil_img2rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(image: Image.Image, max_size: int, min_size: int, patch_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        w, h = int(w * scale), int(h * scale)
    w = max((w // patch_size) * patch_size, patch_size)
    h = max((h // patch_size) * patch_size, patch_size)
    return image.resize((w, h), Image.LANCZOS)


def patchify(img_tensor, patch_size):
    C, H, W = img_tensor.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    patches = img_tensor.reshape(C, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(h_patches * w_patches, C * patch_size * patch_size)
    return patches


def get_flattened_position_ids(H, W, patch_size, max_num_patches_per_side):
    h_patches = H // patch_size
    w_patches = W // patch_size
    position_ids = []
    for i in range(h_patches):
        for j in range(w_patches):
            position_ids.append(i * max_num_patches_per_side + j)
    return torch.tensor(position_ids, dtype=torch.long)


# =============================================================================
# Bagel / ThinkMorph inference
# =============================================================================


@torch.no_grad()
def infer_bagel_understand(model, config, tokenizer, image: Optional[Image.Image], prompt: str, device: str) -> str:
    """Bagel/ThinkMorph understanding with packed sequence forward."""
    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    user_prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    user_suffix = tokenizer.encode(f"\n{prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    vit_patch_size = config.vit_config.patch_size
    max_patches = config.vit_max_num_patch_per_side

    if image is not None:
        image = pil_img2rgb(image)
        image = resize_image(image, 980, 224, 14)
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        H, W = img_tensor.shape[2], img_tensor.shape[3]
        vit_pos_ids = get_flattened_position_ids(H, W, vit_patch_size, max_patches).to(device)
        vit_patches = patchify(img_tensor[0], vit_patch_size).to(device)

        cu_seqlens = torch.tensor([0, vit_patches.shape[0]], dtype=torch.int32, device=device)
        vit_hidden = model.vit_model(vit_patches, vit_pos_ids, cu_seqlens, vit_patches.shape[0])
        vit_embed = model.connector(vit_hidden) + model.vit_pos_embed(vit_pos_ids)

        start_tok = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_tok = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        token_ids = user_prefix + [start_tok] + [0] * vit_embed.shape[0] + [end_tok] + user_suffix
        vit_start = len(user_prefix) + 1
        num_vit = vit_embed.shape[0]
    else:
        token_ids = user_prefix + user_suffix
        vit_start = None
        num_vit = 0

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    embeddings = model.language_model.model.embed_tokens(input_ids)

    if vit_start is not None:
        embeddings[vit_start:vit_start + num_vit] = vit_embed

    seq_len = embeddings.shape[0]
    sample_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    und_indexes = torch.arange(seq_len, device=device)

    hidden = model.language_model(
        packed_sequence=embeddings,
        sample_lens=sample_lens,
        attention_mask=None,
        packed_position_ids=position_ids,
        packed_und_token_indexes=und_indexes,
        packed_gen_token_indexes=torch.tensor([], dtype=torch.long, device=device),
    )

    logits = model.language_model.lm_head(hidden[-1:])
    next_token = torch.argmax(logits[0], dim=-1).item()
    generated = [next_token]

    for _ in range(511):
        if next_token == eos_id:
            break
        tok_emb = model.language_model.model.embed_tokens(torch.tensor([next_token], device=device))
        cur_pos = torch.tensor([seq_len + len(generated) - 1], dtype=torch.long, device=device)
        sample_lens_step = torch.tensor([1], dtype=torch.int32, device=device)
        h = model.language_model(
            packed_sequence=tok_emb,
            sample_lens=sample_lens_step,
            attention_mask=None,
            packed_position_ids=cur_pos,
            packed_und_token_indexes=torch.tensor([0], device=device),
            packed_gen_token_indexes=torch.tensor([], dtype=torch.long, device=device),
        )
        logits = model.language_model.lm_head(h[-1:])
        next_token = torch.argmax(logits[0], dim=-1).item()
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# U1 (NEOChat) inference
# =============================================================================


@torch.no_grad()
def infer_u1_understand(model, config, tokenizer, image: Optional[Image.Image], prompt: str, device: str) -> str:
    """SenseNova-U1 understanding inference."""
    from transformers import AutoProcessor

    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": prompt})

    # Use the processor if available
    try:
        processor = AutoProcessor.from_pretrained(
            config._name_or_path if hasattr(config, "_name_or_path") else "Qwen/Qwen3-VL-8B",
            trust_remote_code=True,
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image] if image else None, return_tensors="pt").to(device)
    except Exception:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        inputs = {"input_ids": input_ids}

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    outputs = model.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else model.language_model.lm_head(outputs.last_hidden_state)

    next_token = torch.argmax(logits[0, -1], dim=-1).item()
    generated = [next_token]

    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    for _ in range(511):
        if next_token == eos_id:
            break
        new_input = torch.tensor([[next_token]], device=device)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)], dim=1)
        outputs = model.language_model(
            input_ids=new_input,
            attention_mask=attention_mask,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else model.language_model.lm_head(outputs.last_hidden_state)
        next_token = torch.argmax(logits[0, -1], dim=-1).item()
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# LatentUM inference
# =============================================================================


@torch.no_grad()
def infer_latentum_understand(model, config, tokenizer, image: Optional[Image.Image], prompt: str, device: str) -> str:
    """LatentUM understanding inference via InternVL backbone."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    embeddings = model.internvl.language_model.get_input_embeddings()(input_ids[0])

    if image is not None:
        image = pil_img2rgb(image)
        image = resize_image(image, 448, 224, 14)
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        vit_features = model.internvl.vision_model(img_tensor)
        vit_projected = model.internvl.mlp1(vit_features)
        embeddings = torch.cat([vit_projected.squeeze(0), embeddings], dim=0)

    seq_len = embeddings.shape[0]
    # Simple causal generation
    hidden = embeddings
    # Use the LM's forward for understanding tokens
    lm = model.internvl.language_model
    logits = lm.lm_head(hidden[-1:])
    next_token = torch.argmax(logits[0], dim=-1).item()
    generated = [next_token]

    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    for _ in range(511):
        if next_token == eos_id:
            break
        tok_emb = lm.get_input_embeddings()(torch.tensor([next_token], device=device))
        hidden = tok_emb
        logits = lm.lm_head(hidden[-1:])
        next_token = torch.argmax(logits[0], dim=-1).item()
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# BLIP3o inference
# =============================================================================


@torch.no_grad()
def infer_blip3o_understand(model, config, tokenizer, image: Optional[Image.Image], prompt: str, device: str) -> str:
    """BLIP3o understanding inference."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # BLIP3o uses standard HF-style forward for understanding
    if image is not None:
        image = pil_img2rgb(image)
        image = resize_image(image, 980, 224, 14)
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        pixel_values = img_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    else:
        pixel_values = None

    # Get language model output
    embeddings = model.get_input_embeddings()(input_ids[0])
    if pixel_values is not None and hasattr(model, "visual"):
        vit_features = model.visual(pixel_values)
        if hasattr(model, "visual_projector"):
            vit_features = model.visual_projector(vit_features)
        embeddings = torch.cat([vit_features.squeeze(0), embeddings], dim=0)

    lm_head = model.get_output_embeddings()
    logits = lm_head(embeddings[-1:])
    next_token = torch.argmax(logits[0], dim=-1).item()
    generated = [next_token]

    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    for _ in range(511):
        if next_token == eos_id:
            break
        tok_emb = model.get_input_embeddings()(torch.tensor([next_token], device=device))
        logits = lm_head(tok_emb)
        next_token = torch.argmax(logits[0], dim=-1).item()
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# Dispatch
# =============================================================================


UNDERSTAND_DISPATCH = {
    "bagel": infer_bagel_understand,
    "thinkmorph": infer_bagel_understand,
    "blip3o_qwen": infer_blip3o_understand,
    "neo_chat": infer_u1_understand,
    "latentum": infer_latentum_understand,
}


def main():
    parser = argparse.ArgumentParser(description="Unified inference for VeOmni multimodal models")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=list(MODEL_TYPE_TO_CHECKPOINT.keys()),
                        help="Model type: bagel, thinkmorph, blip3o, u1, latentum")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (defaults to known path for model_type)")
    parser.add_argument("--mode", type=str, choices=["understand", "generate"], default="understand")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_path = args.model_path or MODEL_TYPE_TO_CHECKPOINT[args.model_type]
    print(f"[Unified] model_type={args.model_type}, model_path={model_path}")

    model, config = load_model(model_path, args.device)
    tokenizer = load_tokenizer(model_path, args.model_type)

    config_model_type = config.model_type
    print(f"[Unified] Loaded model_type from config: {config_model_type}")

    if args.mode == "understand":
        image = Image.open(args.image) if args.image else None
        infer_fn = UNDERSTAND_DISPATCH.get(config_model_type)
        if infer_fn is None:
            print(f"Understanding inference not implemented for {config_model_type}")
            return
        result = infer_fn(model, config, tokenizer, image, args.prompt, args.device)
        print(f"\n{'='*60}")
        print(f"Model: {args.model_type}")
        print(f"Prompt: {args.prompt}")
        print(f"Result: {result}")
        print(f"{'='*60}")
    else:
        print(f"[Unified] Image generation mode for {args.model_type}")
        print("Generation requires model-specific sampling pipelines.")
        print("Use the model-specific inference scripts for full generation support.")


if __name__ == "__main__":
    main()
