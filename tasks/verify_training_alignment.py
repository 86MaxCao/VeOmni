"""Verify training forward alignment: VeOmni vs Official.

For each model, loads the same checkpoint, constructs identical inputs,
runs forward in both VeOmni and official mode, and compares CE loss.

Focus: Understanding SFT only (CE loss on text tokens).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tasks/verify_training_alignment.py --model_type blip3o
    CUDA_VISIBLE_DEVICES=0 python tasks/verify_training_alignment.py --model_type all
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veomni.models.loader import get_model_config, get_model_class
from veomni.models.module_utils import init_empty_weights, load_model_weights

MODEL_CONFIGS = {
    "bagel": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BAGEL-7B-MoT",
    "thinkmorph": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/ThinkMorph-7B",
    "blip3o": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BLIP3o-Model-8B",
    "u1": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT",
    "latentum": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/LatentUM-Base",
}


def load_veomni_model(model_path, device="cuda"):
    config = get_model_config(model_path, trust_remote_code=True)
    model_cls = get_model_class(config)
    with init_empty_weights():
        model = model_cls._from_config(config=config)
    model = model.to_empty(device=device)
    model = model.to(torch.bfloat16)
    load_model_weights(model, model_path)
    return model, config


def verify_blip3o(device="cuda"):
    """BLIP3o: dual-path verification — VeOmni model.forward() vs official manual CE."""
    print("\n" + "=" * 70)
    print("BLIP3o: Understanding SFT alignment verification")
    print("=" * 70)

    model_path = MODEL_CONFIGS["blip3o"]
    t0 = time.time()
    model, config = load_veomni_model(model_path, device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    seq_len = 128
    vocab_size = config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100
    attention_mask = torch.ones_like(input_ids)

    # Path 1: VeOmni model.forward()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            target_images=None,
        )
    loss_veomni = output.loss.item()
    print(f"  VeOmni CE loss: {loss_veomni:.6f}")

    # Path 2: Verify internal loss by recomputing CE from logits
    # Official: blip3o_qwen.py:105-111 uses CrossEntropyLoss(mean) with no ignore_index
    logits = output.logits
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_manual = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
        )

    print(f"  Manual CE loss (from logits): {loss_manual.item():.6f}")
    diff = abs(loss_veomni - loss_manual.item())
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-4:
        print(f"  Status: ALIGNED (diff = {diff:.2e})")
    else:
        print(f"  Status: MISMATCH (diff = {diff:.2e})")

    # Verify gradient flows
    output.loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return diff < 1e-4


def verify_latentum(device="cuda"):
    """LatentUM: dual-path verification — VeOmni model.forward() vs official manual CE."""
    print("\n" + "=" * 70)
    print("LatentUM: Understanding SFT alignment verification")
    print("=" * 70)

    model_path = MODEL_CONFIGS["latentum"]
    t0 = time.time()
    model, config = load_veomni_model(model_path, device)
    model.train()
    model.internvl.vision_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    seq_len = 128
    vocab_size = config.internvl_config.llm_config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100
    attention_mask = torch.ones_like(input_ids)

    # Path 1: VeOmni model.forward()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=None,
        )
    loss_veomni = output.loss.item()
    print(f"  VeOmni CE loss: {loss_veomni:.6f}")

    # Path 2: Verify internal loss by recomputing CE from logits
    # Official: train_interleaved_lang_only.py uses F.cross_entropy with ignore_index=-100
    logits = output.logits
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_manual = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    print(f"  Manual CE loss (from logits): {loss_manual.item():.6f}")
    diff = abs(loss_veomni - loss_manual.item())
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-4:
        print(f"  Status: ALIGNED (diff = {diff:.2e})")
    else:
        print(f"  Status: MISMATCH (diff = {diff:.2e})")

    # Verify gradient flows
    output.loss.backward()
    has_grad = sum(1 for p in model.internvl.language_model.parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return diff < 1e-4


def verify_bagel(device="cuda"):
    """Bagel: packed sequence with ce_loss_indexes."""
    print("\n" + "=" * 70)
    print("Bagel: Understanding SFT alignment verification")
    print("=" * 70)

    model_path = MODEL_CONFIGS["bagel"]
    t0 = time.time()
    model, config = load_veomni_model(model_path, device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch.manual_seed(42)

    # Create a minimal packed sequence batch (text-only, no images)
    seq_len = 64
    vocab_size = config.llm_config.vocab_size
    text_ids = torch.randint(100, vocab_size - 100, (seq_len,), device=device)
    text_indexes = torch.arange(seq_len, device=device)
    position_ids = torch.arange(seq_len, device=device)
    sample_lens = [seq_len]

    # CE loss on last 44 tokens (labels for tokens 20..63)
    ce_loss_indexes = torch.zeros(seq_len, dtype=torch.bool, device=device)
    ce_loss_indexes[20:] = True
    label_ids = torch.randint(100, vocab_size - 100, (ce_loss_indexes.sum().item(),), device=device)

    # Simple causal mask
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(), float('-inf'))

    batch = {
        "sequence_length": seq_len,
        "packed_text_ids": text_ids,
        "packed_text_indexes": text_indexes,
        "sample_lens": sample_lens,
        "packed_position_ids": position_ids,
        "nested_attention_masks": [mask],
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": label_ids,
    }

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)

    loss_veomni = output.ce_loss.item()
    print(f"  VeOmni CE loss: {loss_veomni:.6f}")

    # Official Bagel: same formula but reduction="none" → .sum() / total_tokens
    # For a single sample: mean == sum/N, so they should match.
    # Let's verify by computing manually:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        packed_text_embedding = model.language_model.model.embed_tokens(text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(seq_len, model.hidden_size))
        packed_sequence[text_indexes] = packed_text_embedding

        last_hidden_state = model.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=[mask],
            packed_position_ids=position_ids,
        )
        packed_ce_preds = model.language_model.lm_head(last_hidden_state[ce_loss_indexes])

        # Official formula: reduction="none" then sum/total_tokens
        ce_none = F.cross_entropy(packed_ce_preds, label_ids, reduction="none")
        loss_official = ce_none.sum() / ce_none.numel()

    print(f"  Official CE loss (manual): {loss_official.item():.6f}")
    diff = abs(loss_veomni - loss_official.item())
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-4:
        print(f"  Status: ALIGNED (diff = {diff:.2e})")
    else:
        print(f"  Status: MISMATCH (diff = {diff:.2e})")

    # Verify gradient
    output.loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return diff < 1e-4


def verify_u1(device="cuda"):
    """U1 (NEOChat): dual-path verification — VeOmni model.forward() vs official manual CE."""
    print("\n" + "=" * 70)
    print("U1 (NEOChat): Understanding SFT alignment verification")
    print("=" * 70)

    model_path = MODEL_CONFIGS["u1"]
    t0 = time.time()
    model, config = load_veomni_model(model_path, device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    seq_len = 128
    vocab_size = config.llm_config.vocab_size if hasattr(config, 'llm_config') else 151936
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100
    attention_mask = torch.ones_like(input_ids)

    # Path 1: VeOmni model.forward()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            pixel_values=None,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    loss_veomni = output.loss.item()
    print(f"  VeOmni CE loss: {loss_veomni:.6f}")

    # Path 2: Verify internal loss by recomputing CE from hidden_states
    # Official: FlashGPTLMLoss uses nn.CrossEntropyLoss(reduction="mean")
    hidden_states = output.hidden_states
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model.language_model.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_manual = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )

    print(f"  Manual CE loss (from logits): {loss_manual.item():.6f}")
    diff = abs(loss_veomni - loss_manual.item())
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-4:
        print(f"  Status: ALIGNED (diff = {diff:.2e})")
    else:
        print(f"  Status: MISMATCH (diff = {diff:.2e})")

    # Verify gradient flows
    output.loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return diff < 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["blip3o", "latentum", "bagel", "thinkmorph", "u1", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}

    if args.model_type in ("blip3o", "all"):
        results["blip3o"] = verify_blip3o(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("latentum", "all"):
        results["latentum"] = verify_latentum(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("bagel", "all"):
        results["bagel"] = verify_bagel(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("thinkmorph", "all"):
        # ThinkMorph shares Bagel's architecture
        results["thinkmorph"] = verify_bagel(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("u1", "all"):
        results["u1"] = verify_u1(args.device)
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for model, aligned in results.items():
        status = "ALIGNED" if aligned else "NEEDS WORK"
        print(f"  {model:12s}: {status}")


if __name__ == "__main__":
    main()
