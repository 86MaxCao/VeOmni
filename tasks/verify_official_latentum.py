"""LatentUM official training forward verification.

Loads the LatentUM LLM backbone (Qwen3 MoT) directly from the checkpoint
and runs the same CE loss computation as train_interleaved_lang_only.py.

The official training script does:
  input_embeds = model.internvl.language_model.get_input_embeddings()(input_ids)
  outputs = model.internvl.language_model(inputs_embeds=..., attention_mask=...)
  logits = outputs.logits
  loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

We reproduce this by loading the LLM backbone as a standalone Qwen3 model.

Run with amdpy10 or amdpy11.
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
import json
import os
from safetensors.torch import load_file

model_path = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/LatentUM-Base"

# Load LLM config from the internvl sub-config
with open(os.path.join(model_path, "internvl", "config.json")) as f:
    internvl_config = json.load(f)
llm_config_dict = internvl_config["llm_config"]
llm_config = AutoConfig.for_model(**llm_config_dict)

print("Loading LatentUM LLM backbone (Qwen3 MoT)...")
llm = AutoModelForCausalLM.from_config(llm_config, torch_dtype=torch.bfloat16, attn_implementation="eager")

# Load weights, filtering for internvl.language_model prefix
weights_path = os.path.join(model_path, "model.safetensors")
all_weights = load_file(weights_path)
llm_prefix = "internvl.language_model."
llm_weights = {}
for k, v in all_weights.items():
    if k.startswith(llm_prefix):
        llm_weights[k[len(llm_prefix):]] = v

missing, unexpected = llm.load_state_dict(llm_weights, strict=False)
print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print(f"  Sample missing: {missing[:5]}")

llm = llm.cuda()
llm.train()
print(f"  Loaded. LLM type: {type(llm).__name__}")

# Same random input as VeOmni verification
torch.manual_seed(42)
seq_len = 128
vocab_size = llm_config.vocab_size
input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device="cuda")
labels = input_ids.clone()
labels[:, :20] = -100
attention_mask = torch.ones_like(input_ids)

# Official formula from train_interleaved_lang_only.py:203-215
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
