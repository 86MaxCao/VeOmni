"""Verify training forward pass for all 5 models.

This script loads each model, creates a dummy batch (input_ids + labels),
runs a forward pass, and verifies that loss is computed correctly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tasks/verify_training_forward.py --model_type bagel
    CUDA_VISIBLE_DEVICES=0 python tasks/verify_training_forward.py --model_type all
"""

import argparse
import os
import sys
import time

import torch

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


def verify_model(model_type: str, device: str = "cuda"):
    model_path = MODEL_CONFIGS[model_type]
    print(f"\n{'='*70}")
    print(f"Verifying training forward: {model_type}")
    print(f"Checkpoint: {model_path}")
    print(f"{'='*70}")

    # Load config and model
    t0 = time.time()
    config = get_model_config(model_path, trust_remote_code=True)
    model_cls = get_model_class(config)
    print(f"  model_type={config.model_type}, cls={model_cls.__name__}")

    with init_empty_weights():
        model = model_cls._from_config(config=config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.2f}B")

    model = model.to_empty(device=device)
    model = model.to(torch.bfloat16)
    load_model_weights(model, model_path)
    model.train()
    t1 = time.time()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded in {t1-t0:.1f}s, GPU memory: {mem_gb:.2f} GB")

    # Create dummy training batch
    if model_type in ("bagel", "thinkmorph"):
        batch = _create_bagel_batch(model, config, device)
    elif model_type == "blip3o":
        batch = _create_standard_batch(config, device, vocab_size=getattr(config, "vocab_size", 152064))
    elif model_type == "u1":
        batch = _create_u1_batch(config, device)
    elif model_type == "latentum":
        batch = _create_standard_batch(config, device, vocab_size=getattr(config, "internvl_config", {}).get("llm_config", {}).get("vocab_size", 151936) if isinstance(getattr(config, "internvl_config", None), dict) else 151936)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Forward pass
    t2 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
    t3 = time.time()

    loss = output.loss if hasattr(output, "loss") else output["loss"]
    print(f"  Forward time: {t3-t2:.3f}s")
    print(f"  Loss: {loss.item():.4f}")
    assert loss is not None and not torch.isnan(loss), f"Loss is invalid: {loss}"
    assert loss.requires_grad, "Loss doesn't have grad_fn"

    # Test backward
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"  Parameters with gradients: {len(grad_norms)}")
    if grad_norms:
        max_grad = max(grad_norms, key=lambda x: x[1])
        print(f"  Max grad norm: {max_grad[0]} = {max_grad[1]:.6f}")

    print(f"  ✅ {model_type} training forward PASSED")

    # Clean up
    del model, output
    torch.cuda.empty_cache()
    return True


def _create_bagel_batch(model, config, device):
    """Create a Bagel/ThinkMorph packed sequence batch."""
    seq_len = 128
    vocab_size = config.llm_config.vocab_size

    # Simple text-only batch (understanding mode)
    input_ids = torch.randint(0, vocab_size, (seq_len,), device=device)
    text_indexes = torch.arange(seq_len, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    sample_lens = [seq_len]

    # CE loss: boolean mask of size [seq_len], labels of matching size
    # Mask out the first token (no label for it), predict the rest
    ce_loss_indexes = torch.zeros(seq_len, dtype=torch.bool, device=device)
    ce_loss_indexes[1:] = True  # predict from position 1 onward
    num_loss_tokens = ce_loss_indexes.sum().item()
    label_ids = torch.randint(0, vocab_size, (num_loss_tokens,), device=device)

    return {
        "sequence_length": seq_len,
        "packed_text_ids": input_ids,
        "packed_text_indexes": text_indexes,
        "sample_lens": sample_lens,
        "packed_position_ids": position_ids,
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": label_ids,
    }


def _create_standard_batch(config, device, vocab_size=151936):
    """Create a standard [B, seq_len] batch for BLIP3o / LatentUM."""
    batch_size = 1
    seq_len = 128

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :10] = -100  # mask first 10 tokens as prompt

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _create_u1_batch(config, device):
    """Create a U1 (NEOChat) batch."""
    batch_size = 1
    seq_len = 128
    vocab_size = config.llm_config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :10] = -100

    # U1 expects 4D causal attention mask [B, 1, S, S] or None
    # Passing None lets the model use no mask (all-to-all attention)
    # For proper causal, create a lower-triangular mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["bagel", "thinkmorph", "blip3o", "u1", "latentum", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.model_type == "all":
        models = ["bagel", "thinkmorph", "blip3o", "u1", "latentum"]
    else:
        models = [args.model_type]

    results = {}
    for m in models:
        try:
            results[m] = verify_model(m, args.device)
        except Exception as e:
            print(f"  ❌ {m} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[m] = False
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for m, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {m:12s} {status}")


if __name__ == "__main__":
    main()
