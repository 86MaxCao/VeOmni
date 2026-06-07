"""U1 official training forward verification.

U1's official forward() raises NotImplementedError. Training is handled by
InternEvo's pipeline which calls the model's forward internally.
The actual CE loss is: embed → LM forward → shifted CE (mean).

We verify by loading the official model and running the LM forward directly.

Run with amdpy10: CUDA_VISIBLE_DEVICES=0 /mnt/nas-tbt/caoziqi/micromamba/envs/amdpy10/bin/python tasks/verify_official_u1.py
"""
import sys
import os
import torch

ABLATION_DIR = "/home/ximeng.czq/caoziqi/code/SpatialIntelligence/generative-spatial-drafts/ablation_experiment"
sys.path.insert(0, os.path.join(ABLATION_DIR, "SenseNova-U1", "src"))

model_path = "/mnt/nas-tbt/tbt/checkpoint/hf_cache/SenseNova-U1-8B-MoT"

from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig

print("Loading U1 via official NEOChatModel...")
config = NEOChatConfig.from_pretrained(model_path)
model = NEOChatModel.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).cuda()
model.train()
print(f"  Loaded. LLM type: {type(model.language_model).__name__}")

# Same random input as VeOmni verification
torch.manual_seed(42)
seq_len = 128
vocab_size = config.llm_config.vocab_size if hasattr(config, 'llm_config') else 151936
input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device="cuda")
labels = input_ids.clone()
labels[:, :20] = -100

# Official: embed → LM forward → shifted CE (nn.CrossEntropyLoss mean)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    t_indexes = torch.arange(seq_len, device="cuda")
    h_indexes = torch.zeros(seq_len, device="cuda", dtype=torch.long)
    w_indexes = torch.zeros(seq_len, device="cuda", dtype=torch.long)
    indexes = [t_indexes, h_indexes, w_indexes]
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        indexes=indexes,
        use_cache=False,
    )
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = model.language_model.lm_head(outputs.last_hidden_state)
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
