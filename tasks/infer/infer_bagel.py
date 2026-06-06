"""Bagel model inference script for VeOmni.

Uses the official three-step KV-cache inference approach:
  Step 1: Image prefill — bidirectional attention (is_causal=False)
  Step 2: Text prefill — causal attention with cross-attention to image KV cache
  Step 3: Autoregressive generation — causal attention to all cached KV

Usage:
    python tasks/infer/infer_bagel.py \
        --model_path /mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT \
        --image path/to/image.jpg \
        --prompt "Describe this image."
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from veomni.models.loader import get_model_config, get_model_class
from veomni.models.module_utils import init_empty_weights, load_model_weights


class ImageTransform:
    def __init__(self, max_size, min_size, patch_size):
        self.max_size = max_size
        self.min_size = min_size
        self.patch_size = patch_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        max_side = max(w, h)
        if max_side > self.max_size:
            scale = self.max_size / max_side
            w, h = int(w * scale), int(h * scale)
        if min(w, h) < self.min_size:
            scale = self.min_size / min(w, h)
            w, h = int(w * scale), int(h * scale)

        w = max((w // self.patch_size) * self.patch_size, self.patch_size)
        h = max((h // self.patch_size) * self.patch_size, self.patch_size)
        image = image.resize((w, h), Image.LANCZOS)
        return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0


def load_bagel_model(model_path, device="cuda"):
    config = get_model_config(model_path, trust_remote_code=True)
    model_cls = get_model_class(config)
    with init_empty_weights():
        model = model_cls._from_config(config=config)
    model = model.to_empty(device=device)
    model = model.to(torch.bfloat16)
    load_model_weights(model, model_path)
    model.eval()
    return model, config


def load_tokenizer(model_path):
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


def main():
    parser = argparse.ArgumentParser(description="Bagel model inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, config = load_bagel_model(args.model_path, args.device)
    tokenizer, new_token_ids = load_tokenizer(args.model_path)
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    image_transform = ImageTransform(980, 224, config.vit_config.patch_size)
    images = []
    if args.image:
        img = Image.open(args.image)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images = [img]

    result = model.chat(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        image_transform=image_transform,
        images=images,
        prompt=args.prompt,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
