"""Verify training forward alignment: Official models.

Runs official model code (under ablation_experiment/) to compute CE loss
on the same fixed-seed inputs used by verify_training_alignment.py.

Must run with amdpy10 environment (transformers 4.x) for BLIP3o/LatentUM.
U1 official forward raises NotImplementedError, so we use direct LM forward.

Usage:
    CUDA_VISIBLE_DEVICES=0 /mnt/nas-tbt/caoziqi/micromamba/envs/amdpy10/bin/python \
        tasks/verify_training_alignment_official.py --model_type blip3o
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

ABLATION_DIR = "/home/ximeng.czq/caoziqi/code/SpatialIntelligence/generative-spatial-drafts/ablation_experiment"

MODEL_CONFIGS = {
    "blip3o": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/BLIP3o-Model-8B",
    "u1": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT",
    "latentum": "/mnt/nas-tbt/tbt/checkpoint/hf_cache/LatentUM-Base",
}


def verify_blip3o_official(device="cuda"):
    """BLIP3o official: blip3oQwenForCausalLM.forward()"""
    print("\n" + "=" * 70)
    print("BLIP3o Official: Understanding SFT CE loss")
    print("=" * 70)

    model_path = MODEL_CONFIGS["blip3o"]

    # Add official BLIP3o code to path
    blip3o_dir = os.path.join(ABLATION_DIR, "BLIP3o")
    sys.path.insert(0, blip3o_dir)

    from blip3o.model import blip3oQwenForCausalLM

    t0 = time.time()
    model = blip3oQwenForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Same random input as VeOmni verification script
    torch.manual_seed(42)
    seq_len = 128
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100
    attention_mask = torch.ones_like(input_ids)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            target_images=None,
        )

    loss = output.loss.item()
    print(f"  Official CE loss: {loss:.6f}")

    output.loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return loss


def verify_u1_official(device="cuda"):
    """U1 official: direct LM forward via NEOChatModel."""
    print("\n" + "=" * 70)
    print("U1 Official: Understanding SFT CE loss")
    print("=" * 70)

    model_path = MODEL_CONFIGS["u1"]

    sys.path.insert(0, os.path.join(ABLATION_DIR, "SenseNova-U1", "src"))
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
    from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig

    t0 = time.time()
    config = NEOChatConfig.from_pretrained(model_path)
    model = NEOChatModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    seq_len = 128
    vocab_size = config.llm_config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100

    # Official: embed → LM forward with MoT indexes → shifted CE
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        t_indexes = torch.arange(seq_len, device=device)
        h_indexes = torch.zeros(seq_len, device=device, dtype=torch.long)
        w_indexes = torch.zeros(seq_len, device=device, dtype=torch.long)
        indexes = [t_indexes, h_indexes, w_indexes]
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            use_cache=False,
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else model.language_model.lm_head(outputs.last_hidden_state)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    print(f"  Official CE loss: {loss.item():.6f}")

    loss.backward()
    has_grad = sum(1 for p in model.language_model.parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    model.zero_grad()
    return loss.item()


def verify_latentum_official(device="cuda"):
    """LatentUM official: standalone LLM backbone (Qwen3) forward + shifted CE."""
    print("\n" + "=" * 70)
    print("LatentUM Official: Understanding SFT CE loss")
    print("=" * 70)

    model_path = MODEL_CONFIGS["latentum"]

    import json
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors.torch import load_file

    # Load LLM config from InternVL sub-config
    with open(os.path.join(model_path, "internvl", "config.json")) as f:
        internvl_config = json.load(f)
    llm_config = AutoConfig.for_model(**internvl_config["llm_config"])

    t0 = time.time()
    llm = AutoModelForCausalLM.from_config(llm_config, torch_dtype=torch.bfloat16, attn_implementation="eager")

    # Load weights, filtering for internvl.language_model prefix
    weights_path = os.path.join(model_path, "model.safetensors")
    all_weights = load_file(weights_path)
    llm_prefix = "internvl.language_model."
    llm_weights = {k[len(llm_prefix):]: v for k, v in all_weights.items() if k.startswith(llm_prefix)}
    llm.load_state_dict(llm_weights, strict=False)

    llm = llm.to(device)
    llm.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    seq_len = 128
    vocab_size = llm_config.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :20] = -100
    attention_mask = torch.ones_like(input_ids)

    # Official: train_interleaved_lang_only.py:203-215
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        input_embeds = llm.get_input_embeddings()(input_ids)
        outputs = llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    print(f"  Official CE loss: {loss.item():.6f}")

    loss.backward()
    has_grad = sum(1 for p in llm.parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradient: {has_grad}")
    llm.zero_grad()
    del all_weights
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["blip3o", "latentum", "u1", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}

    if args.model_type in ("blip3o", "all"):
        results["blip3o"] = verify_blip3o_official(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("latentum", "all"):
        results["latentum"] = verify_latentum_official(args.device)
        torch.cuda.empty_cache()

    if args.model_type in ("u1", "all"):
        results["u1"] = verify_u1_official(args.device)
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Official Loss Values (compare with VeOmni):")
    print("=" * 70)
    for model_name, loss_val in results.items():
        print(f"  {model_name:12s}: CE loss = {loss_val:.6f}")


if __name__ == "__main__":
    main()
