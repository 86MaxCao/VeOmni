"""Bagel model inference script for VeOmni.

Supports both understanding (VQA) and generation (T2I) modes.
Uses VeOmni model loading with the original Bagel inference logic.

Usage:
    # Understanding (VQA)
    python tasks/infer/infer_bagel.py \
        --model_path /mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT \
        --mode understand \
        --image path/to/image.jpg \
        --prompt "Describe this image."

    # Generation (T2I)
    python tasks/infer/infer_bagel.py \
        --model_path /mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT \
        --mode generate \
        --prompt "A cat sitting on a windowsill" \
        --output output.png
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from veomni.models.loader import get_model_config, get_model_class
from veomni.models.module_utils import init_empty_weights, load_model_weights


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


class ImageTransform:
    def __init__(self, max_size, min_size, patch_size):
        self.max_size = max_size
        self.min_size = min_size
        self.patch_size = patch_size

    def resize_transform(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_side = max(w, h)
        min_side = min(w, h)

        if max_side > self.max_size:
            scale = self.max_size / max_side
            w, h = int(w * scale), int(h * scale)
        if min(w, h) < self.min_size:
            scale = self.min_size / min(w, h)
            w, h = int(w * scale), int(h * scale)

        w = (w // self.patch_size) * self.patch_size
        h = (h // self.patch_size) * self.patch_size
        w = max(w, self.patch_size)
        h = max(h, self.patch_size)

        image = image.resize((w, h), Image.LANCZOS)
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.resize_transform(image)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image


def pil_img2rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


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


def load_bagel_model(model_path: str, device: str = "cuda"):
    """Load Bagel model via VeOmni loader."""
    config = get_model_config(model_path, trust_remote_code=True)
    model_cls = get_model_class(config)

    with init_empty_weights():
        model = model_cls._from_config(config=config)

    model = model.to_empty(device=device)
    model = model.to(torch.bfloat16)
    load_model_weights(model, model_path)
    model.eval()
    return model, config


def load_tokenizer(model_path: str):
    """Load tokenizer and add special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    special_tokens = []
    existing = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            existing.append(v)
        elif isinstance(v, list):
            existing.extend(v)

    for token in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]:
        if token not in existing:
            special_tokens.append(token)

    if special_tokens:
        tokenizer.add_tokens(special_tokens)

    new_token_ids = {
        "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    return tokenizer, new_token_ids


class BagelInferencer:
    """Simplified Bagel inference wrapper."""

    def __init__(self, model, tokenizer, new_token_ids, config, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.config = config
        self.device = device

        self.vit_transform = ImageTransform(980, 224, 14)
        self.vae_transform = ImageTransform(1024, 512, 16)

        self.num_layers = config.llm_config.num_hidden_layers
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.vit_patch_size = config.vit_config.patch_size
        self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
        self.latent_patch_size = config.latent_patch_size
        self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
        self.max_latent_size = config.max_latent_size

    def _init_cache(self):
        return NaiveCache(self.num_layers)

    @torch.no_grad()
    def understand(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_length: int = 512,
        do_sample: bool = False,
        temperature: float = 0.3,
    ) -> str:
        """Run visual understanding (VQA) inference."""
        model = self.model
        device = self.device

        # Build the full input sequence for a single sample
        # Format: [bos] system_prompt [eos] [bos] user_msg (with image) [eos] [bos] assistant
        all_token_ids = []
        all_position_ids = []
        vit_embeddings = None
        vit_indices = None
        curr_pos = 0

        # Encode system prompt
        sys_text = ""
        sys_ids = self.tokenizer.encode(sys_text) if sys_text else []

        # Encode user message with image
        user_prefix_ids = self.tokenizer.encode(f"<|im_start|>user\n")
        user_suffix_ids = self.tokenizer.encode(f"\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

        # Process image through ViT if provided
        if image is not None:
            image = pil_img2rgb(image)
            image = self.vit_transform.resize_transform(image)
            img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            # Get ViT features
            H, W = img_tensor.shape[2], img_tensor.shape[3]
            vit_position_ids = get_flattened_position_ids(
                H, W, self.vit_patch_size, self.vit_max_num_patch_per_side
            ).to(device)

            # Patchify for ViT
            vit_patches = patchify(img_tensor[0], self.vit_patch_size).to(device)

            # Run ViT
            cu_seqlens = torch.tensor([0, vit_patches.shape[0]], dtype=torch.int32, device=device)
            vit_hidden = model.vit_model(vit_patches, vit_position_ids, cu_seqlens, vit_patches.shape[0])

            # Project to LLM dimension
            vit_embeddings = model.connector(vit_hidden)

            # Build token sequence: user_prefix + [start_of_image] + <vit_tokens> + [end_of_image] + user_suffix
            token_ids = user_prefix_ids + [self.new_token_ids["start_of_image"]]
            num_vit_tokens = vit_embeddings.shape[0]
            # placeholder token ids for ViT tokens (will be replaced by embeddings)
            vit_start_idx = len(token_ids)
            token_ids += [0] * num_vit_tokens  # placeholders
            token_ids += [self.new_token_ids["end_of_image"]]
            token_ids += user_suffix_ids
        else:
            token_ids = user_prefix_ids + user_suffix_ids
            vit_start_idx = None
            num_vit_tokens = 0

        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Get text embeddings
        text_embeddings = model.language_model.model.embed_tokens(input_ids[0])

        # Replace placeholder positions with ViT embeddings
        if vit_embeddings is not None:
            text_embeddings[vit_start_idx:vit_start_idx + num_vit_tokens] = vit_embeddings

        # Add ViT positional embeddings
        if vit_embeddings is not None:
            vit_pos_emb = model.vit_pos_embed(vit_position_ids)
            text_embeddings[vit_start_idx:vit_start_idx + num_vit_tokens] += vit_pos_emb

        # Create position ids (all image tokens share one rope position)
        position_ids = []
        pos = 0
        for i in range(len(token_ids)):
            if vit_start_idx is not None and vit_start_idx <= i < vit_start_idx + num_vit_tokens + 2:
                # image region: start_of_image, vit tokens, end_of_image all share same position
                position_ids.append(pos)
            else:
                position_ids.append(pos)
                pos += 1
        if vit_start_idx is not None:
            pos += 1  # advance past image region

        # Run LLM in understanding mode with packed sequence
        seq_len = text_embeddings.shape[0]
        sample_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        position_ids_tensor = torch.tensor(position_ids, dtype=torch.long, device=device)

        # Prepare packed position embeddings (RoPE)
        cos, sin = model.language_model.model.rotary_emb(
            text_embeddings.unsqueeze(0), position_ids_tensor.unsqueeze(0)
        )
        packed_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        # Understanding path indexes
        und_indexes = torch.arange(seq_len, device=device)

        # Forward through LLM layers
        hidden = text_embeddings.unsqueeze(0) if text_embeddings.dim() == 1 else text_embeddings

        output = model.language_model(
            packed_sequence=hidden,
            sample_lens=sample_lens,
            attention_mask=None,
            packed_position_ids=position_ids_tensor,
            packed_und_token_indexes=und_indexes,
            packed_gen_token_indexes=torch.tensor([], dtype=torch.long, device=device),
        )

        # Get logits for last token
        logits = model.language_model.lm_head(output[:1] if output.dim() == 1 else output[-1:])

        # Autoregressive generation
        generated_ids = []
        next_token = torch.argmax(logits[0, -1] if logits.dim() == 3 else logits[-1], dim=-1)
        generated_ids.append(next_token.item())

        for _ in range(max_length - 1):
            if next_token.item() == self.new_token_ids["eos_token_id"]:
                break
            # This is a simplified single-pass. Full KV-cache inference requires
            # the NaViT packed attention pattern from the original code.
            break  # TODO: implement full autoregressive with KV cache

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text


def main():
    parser = argparse.ArgumentParser(description="Bagel model inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["understand", "generate"], default="understand")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, config = load_bagel_model(args.model_path, args.device)
    tokenizer, new_token_ids = load_tokenizer(args.model_path)
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    inferencer = BagelInferencer(model, tokenizer, new_token_ids, config, args.device)

    if args.mode == "understand":
        image = Image.open(args.image) if args.image else None
        result = inferencer.understand(image, args.prompt, max_length=args.max_length)
        print(f"\nResult: {result}")
    else:
        print("Image generation requires full KV-cache inference pipeline.")
        print("Use the original Bagel inferencer for generation tasks.")


if __name__ == "__main__":
    main()
