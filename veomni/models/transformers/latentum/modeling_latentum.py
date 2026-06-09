"""
LatentUM Model: Multimodal understanding + generation via MoT discrete tokens.

Architecture:
- internvl: InternVLChatModel with MoT (Mixture of Transformers) modifications
  - vision_model: InternVisionModel (ViT encoder)
  - language_model: Qwen3ForCausalLM (with MoT dual-path attention and MLP)
  - mlp1: vision-to-language projector (pixel shuffle + MLP)
  - visual_projector: embedding_dim -> llm_hidden_size projector
  - ar_head: AutoregressiveHead for discrete token generation
- quantizer: VQ_MLP_MCQ (multi-codebook vector quantizer)

Weight key structure in checkpoint:
  internvl.vision_model.*
  internvl.language_model.*
  internvl.mlp1.*
  internvl.visual_projector.*
  internvl.ar_head.*
  quantizer.*
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen3ForCausalLM
from transformers.modeling_outputs import BaseModelOutput

from .configuration_latentum import InternVisionConfig, LatentUMConfig


# ==============================================================================
# Sampling utilities for AR generation
# ==============================================================================


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) filtering."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token indices from logits with temperature, top-k, and top-p.

    Args:
        logits: (B, V) logits
    Returns:
        idx: (B, 1) sampled indices
        probs: (B, V) probabilities
    """
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


# ==============================================================================
# InternVision components
# ==============================================================================


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed


class InternAttention(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)

        norm_cls = nn.LayerNorm if config.norm_type == "layer_norm" else nn.LayerNorm
        self.norm1 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class InternVisionEncoder(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class InternVisionModel(nn.Module):
    """InternVision ViT encoder."""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        return self.encoder(hidden_states).last_hidden_state


# ==============================================================================
# MoT (Mixture of Transformers) Qwen3 Attention and MLP
# ==============================================================================


class Qwen3MoTRMSNorm(nn.Module):
    """RMSNorm matching Qwen3's implementation (official LatentUM pattern)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MoTMLP(nn.Module):
    """Qwen3-style MLP (SwiGLU) for the vision MoT path."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoTAttention(nn.Module):
    """
    Qwen3 attention layer with MoT dual-path projections.

    Contains standard text projections (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm)
    plus vision-specific duplicates (*_vision) for the MoT vision path.
    Ref: LatentUM/model/latentum/internvl/mot.py — create_mot_attention_forward
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Text path
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3MoTRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3MoTRMSNorm(head_dim, eps=rms_norm_eps)

        # Vision path (MoT)
        self.q_proj_vision = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj_vision = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj_vision = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj_vision = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm_vision = Qwen3MoTRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm_vision = Qwen3MoTRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(self, hidden_states, rope_cos, rope_sin, vision_token_mask=None, attention_mask_4d=None):
        """Forward with MoT routing.

        Ref: LatentUM/model/latentum/internvl/mot.py — create_mot_attention_forward
        """
        B, N, D = hidden_states.shape
        has_mot = vision_token_mask is not None and vision_token_mask.any()

        if has_mot:
            vmask = vision_token_mask[:, None, :, None]  # [B, 1, S, 1]
            q_t = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k_t = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_t = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q_v = self.q_proj_vision(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k_v = self.k_proj_vision(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_v = self.v_proj_vision(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q_t = self.q_norm(q_t)
            k_t = self.k_norm(k_t)
            q_v = self.q_norm_vision(q_v)
            k_v = self.k_norm_vision(k_v)
            q = vmask * q_v + (1 - vmask) * q_t
            k = vmask * k_v + (1 - vmask) * k_t
            v = vmask * v_v + (1 - vmask) * v_t
        else:
            q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)

        head_dim = self.head_dim
        orig_q_dtype = q.dtype
        q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2:]
        q = q * rope_cos + torch.cat((-q2, q1), dim=-1) * rope_sin
        q = q.to(orig_q_dtype)
        k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2:]
        k = k * rope_cos + torch.cat((-k2, k1), dim=-1) * rope_sin
        k = k.to(orig_q_dtype)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if attention_mask_4d is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask_4d)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)

        if has_mot:
            vmask_s = vision_token_mask[:, :, None]  # [B, S, 1]
            o_t = self.o_proj(attn_out)
            o_v = self.o_proj_vision(attn_out)
            return vmask_s * o_v + (1 - vmask_s) * o_t
        else:
            return self.o_proj(attn_out)


class Qwen3MoTDecoderLayer(nn.Module):
    """
    Qwen3 decoder layer with MoT dual-path MLP.

    Contains standard components (input_layernorm, self_attn, post_attention_layernorm, mlp)
    plus a vision-specific mlp_vision for the MoT path.
    Ref: LatentUM/model/latentum/internvl/mot.py — create_mot_decoder_forward
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.input_layernorm = Qwen3MoTRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3MoTAttention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoTRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MoTMLP(hidden_size, intermediate_size)
        self.mlp_vision = Qwen3MoTMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, rope_cos, rope_sin, vision_token_mask=None, attention_mask_4d=None):
        """Forward with MoT routing.

        Ref: LatentUM/model/latentum/internvl/mot.py — create_mot_decoder_forward
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rope_cos, rope_sin, vision_token_mask, attention_mask_4d)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        has_mot = vision_token_mask is not None and vision_token_mask.any()
        if has_mot:
            vmask_s = vision_token_mask[:, :, None]
            mlp_t = self.mlp(hidden_states)
            mlp_v = self.mlp_vision(hidden_states)
            hidden_states = vmask_s * mlp_v + (1 - vmask_s) * mlp_t
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoTModel(nn.Module):
    """Qwen3 backbone model with MoT decoder layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, vocab_size: int, rms_norm_eps: float):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3MoTDecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.norm = Qwen3MoTRMSNorm(hidden_size, eps=rms_norm_eps)


class Qwen3MoTForCausalLM(nn.Module):
    """Qwen3ForCausalLM with MoT modifications."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, vocab_size: int, rms_norm_eps: float):
        super().__init__()
        self.model = Qwen3MoTModel(hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads, head_dim, vocab_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


# ==============================================================================
# Autoregressive Head
# ==============================================================================


def _precompute_freqs_cis_1d(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    pos = torch.arange(seq_len)
    freqs = torch.outer(pos, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class ARHeadRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ARHeadAttention(nn.Module):
    """Attention for the AR head with RMSNorm on Q/K."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.q_norm = ARHeadRMSNorm(self.head_dim)
        self.k_norm = ARHeadRMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, Hc]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)  # [B, H, N, Hc]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ARHeadFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        ffn_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, ffn_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ARHeadDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = ARHeadRMSNorm(hidden_size, eps=1e-6)
        self.attn = ARHeadAttention(hidden_size, num_heads)
        self.norm2 = ARHeadRMSNorm(hidden_size, eps=1e-6)
        self.mlp = ARHeadFeedForward(hidden_size, mlp_ratio)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class AutoregressiveHead(nn.Module):
    """
    Autoregressive head for discrete image token generation.

    Generates num_codebooks tokens autoregressively, each selecting from
    num_embeddings entries.
    """

    def __init__(self, num_codebooks: int, num_layers: int, hidden_size: int, num_embeddings: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings, hidden_size) for _ in range(num_codebooks)]
        )
        self.layers = nn.ModuleList(
            [ARHeadDecoderLayer(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = ARHeadRMSNorm(hidden_size, eps=1e-5)
        self.head = nn.Linear(hidden_size, num_embeddings)
        self._freqs_cache: dict[int, torch.Tensor] = {}

    def _code_to_embeddings(self, code: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices to embeddings.

        Args:
            code: (B, L, K) codebook indices where K = num_codebooks
        Returns:
            (B*L, K, D) embeddings
        """
        B, L, K = code.shape
        code = code.reshape(B * L, K)
        embs = []
        for i in range(K):
            embs.append(self.embeddings[i](code[:, i]))
        return torch.stack(embs, dim=1)

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._freqs_cache:
            head_dim = self.hidden_size // self.num_heads
            self._freqs_cache[seq_len] = _precompute_freqs_cis_1d(head_dim, seq_len)
        return self._freqs_cache[seq_len].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AR head decoder layers.

        Args:
            x: (B, N, D) input sequence
        Returns:
            (B, N, num_embeddings) logits
        """
        freqs_cis = self._get_freqs_cis(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)
        return self.head(x)

    def generate_from_base_token(
        self,
        base_token: torch.Tensor,
        cfg_scale: float,
        sampling_kwargs: dict,
    ) -> torch.Tensor:
        """Generate K codebook indices from a single LLM hidden state.

        Args:
            base_token: (B, 1, D) if cfg_scale <= 1,
                        (2*B, 1, D) if cfg_scale > 1 (first B=cond, last B=uncond)
            cfg_scale: classifier-free guidance scale
            sampling_kwargs: dict with temperature, top_k, top_p, sample_logits
        Returns:
            (B, K) generated codebook indices
        """
        generated_code = []
        if cfg_scale > 1:
            B = base_token.shape[0] // 2
            curr_state_cond = base_token[:B]
            curr_state_uncond = base_token[B:]
            for i in range(self.num_codebooks):
                logits_cond = self.forward(curr_state_cond)[:, -1, :]
                logits_uncond = self.forward(curr_state_uncond)[:, -1, :]
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                next_token, _ = _sample_from_logits(logits, **sampling_kwargs)
                generated_code.append(next_token)
                next_embeddings = self.embeddings[i](next_token)
                curr_state_cond = torch.cat([curr_state_cond, next_embeddings], dim=1)
                curr_state_uncond = torch.cat([curr_state_uncond, next_embeddings], dim=1)
            return torch.stack(generated_code, dim=1).squeeze(-1)
        else:
            curr_state = base_token
            for i in range(self.num_codebooks):
                logits = self.forward(curr_state)[:, -1, :]
                next_token, _ = _sample_from_logits(logits, **sampling_kwargs)
                generated_code.append(next_token)
                next_embeddings = self.embeddings[i](next_token)
                curr_state = torch.cat([curr_state, next_embeddings], dim=1)
            return torch.stack(generated_code, dim=1).squeeze(-1)


# ==============================================================================
# Vector Quantizer
# ==============================================================================


class VectorQuantizer(nn.Module):
    """Single-codebook vector quantizer."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.register_buffer("ema_usage", torch.zeros(num_embeddings))

    def quantize(self, z: torch.Tensor):
        """Find nearest codebook entries.

        Args:
            z: (..., D) continuous features
        Returns:
            z_q: (..., D) quantized features
            indices: (...) codebook indices
        """
        flat = z.reshape(-1, self.embedding_dim)
        dist = torch.cdist(flat.float(), self.codebook.weight.float())
        indices = dist.argmin(dim=-1)
        z_q = self.codebook(indices)
        return z_q.reshape(z.shape), indices.reshape(z.shape[:-1])


class MultiVectorQuantizer(nn.Module):
    """Multi-codebook (product) vector quantizer."""

    def __init__(self, num_embeddings: int, embedding_dim: int, num_codebooks: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embedding_dim = embedding_dim
        dim_per_codebook = embedding_dim // num_codebooks
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(num_embeddings, dim_per_codebook) for _ in range(num_codebooks)]
        )

    def quantize(self, z: torch.Tensor):
        """Product quantization across codebooks.

        Args:
            z: (..., D) features where D = num_codebooks * dim_per_codebook
        Returns:
            z_q: (..., D) quantized features
            indices: (..., num_codebooks) codebook indices
        """
        dim_per_cb = self.embedding_dim // self.num_codebooks
        z_splits = z.split(dim_per_cb, dim=-1)
        z_qs = []
        all_indices = []
        for i, (vq, z_i) in enumerate(zip(self.quantizers, z_splits)):
            z_q_i, idx_i = vq.quantize(z_i)
            z_qs.append(z_q_i)
            all_indices.append(idx_i)
        z_q = torch.cat(z_qs, dim=-1)
        indices = torch.stack(all_indices, dim=-1)
        return z_q, indices

    def indices_to_feature(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up codebook embeddings from indices.

        Args:
            indices: (..., num_codebooks) codebook indices
        Returns:
            z_q: (..., D) reconstructed quantized features
            indices: unchanged
        """
        z_qs = []
        for i, vq in enumerate(self.quantizers):
            z_q_i = vq.codebook(indices[..., i])
            z_qs.append(z_q_i)
        z_q = torch.cat(z_qs, dim=-1)
        return z_q, indices


class VQ_MLP_MCQ(nn.Module):
    """
    Multi-codebook VQ with MLP projection.

    Structure:
        down_proj: input_feature_dim -> embedding_dim
        quantizer: MultiVectorQuantizer
        up_proj: embedding_dim -> llm_hidden_size
    """

    def __init__(self, input_feature_dim: int, embedding_dim: int, llm_hidden_size: int, num_embeddings: int, num_codebooks: int):
        super().__init__()
        self.down_proj = nn.Sequential(
            nn.Linear(input_feature_dim, 4 * input_feature_dim),
            nn.GELU(),
            nn.Linear(4 * input_feature_dim, embedding_dim),
        )
        self.quantizer = MultiVectorQuantizer(num_embeddings, embedding_dim, num_codebooks)
        self.up_proj = nn.Sequential(
            nn.Linear(embedding_dim, 4 * llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * llm_hidden_size, llm_hidden_size),
        )

    def get_zq_indices(self, vit_features: torch.Tensor):
        """Encode ViT features to quantized latents and codebook indices.

        Args:
            vit_features: (B, L, D) ViT features (post pixel-shuffle, pre-mlp1)
        Returns:
            z_q: (B, L, embedding_dim) quantized latent vectors
            indices: (B, L, num_codebooks) codebook indices
        """
        z = self.down_proj(vit_features)
        z_q, indices = self.quantizer.quantize(z)
        return z_q, indices

    def indices_to_feature(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map codebook indices back to continuous quantized features.

        Args:
            indices: (..., num_codebooks) codebook indices
        Returns:
            z_q: (..., embedding_dim) quantized features
            indices: unchanged
        """
        return self.quantizer.indices_to_feature(indices)


# ==============================================================================
# InternVL Chat Model (with MoT modifications)
# ==============================================================================


class InternVLChatModel(nn.Module):
    """
    InternVL Chat Model with MoT dual-path modifications.

    Components and their weight key prefixes (under 'internvl.'):
        vision_model.* : InternVisionModel
        language_model.* : Qwen3MoTForCausalLM
        mlp1.* : vision-to-language projector (pixel shuffle + MLP)
        visual_projector.* : VQ embedding -> LLM hidden projector
        ar_head.* : AutoregressiveHead
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_model = InternVisionModel(config.vision_config)

        # Language model with MoT
        llm_cfg = config.llm_config
        self.language_model = Qwen3MoTForCausalLM(
            hidden_size=llm_cfg.hidden_size,
            intermediate_size=llm_cfg.intermediate_size,
            num_layers=llm_cfg.num_hidden_layers,
            num_heads=llm_cfg.num_attention_heads,
            num_kv_heads=llm_cfg.num_key_value_heads,
            head_dim=getattr(llm_cfg, "head_dim", llm_cfg.hidden_size // llm_cfg.num_attention_heads),
            vocab_size=llm_cfg.vocab_size,
            rms_norm_eps=llm_cfg.rms_norm_eps,
        )

        # Vision-to-language projector (pixel shuffle + MLP)
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = llm_cfg.hidden_size
        downsample_ratio = config.downsample_ratio
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Visual projector (embedding_dim -> llm_hidden_size)
        # This is set externally by modify_internvl_to_mixture
        self.visual_projector: nn.Module = None  # type: ignore[assignment]

        # AR head (set externally)
        self.ar_head: nn.Module = None  # type: ignore[assignment]

    def get_vit_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get ViT features with pixel shuffle but WITHOUT mlp1.

        Used for VQ encoding of target images (generation path).
        """
        vit_features = self.vision_model(pixel_values)
        if vit_features.dim() == 3:
            b, n, d = vit_features.shape
            h = w = int(n ** 0.5)
            # Remove CLS token (position 0) — official extract_feature does vit_embeds[:, 1:, :]
            if h * w < n:
                vit_features = vit_features[:, 1:h * w + 1, :]
                n = h * w
            ds = int(1 / self.config.downsample_ratio)
            if h % ds == 0 and w % ds == 0:
                vit_features = vit_features.reshape(b, h, w, d)
                vit_features = vit_features.reshape(b, h // ds, ds, w // ds, ds, d)
                vit_features = vit_features.permute(0, 1, 3, 2, 4, 5).reshape(b, (h // ds) * (w // ds), d * ds * ds)
        return vit_features


# ==============================================================================
# Top-level LatentUM Model
# ==============================================================================


class LatentUMModel(PreTrainedModel):
    """
    LatentUM: Multimodal understanding + generation via MoT with discrete tokens.

    This model wraps:
    - internvl: InternVLChatModel with MoT modifications, visual_projector, and ar_head
    - quantizer: VQ_MLP_MCQ multi-codebook vector quantizer

    The state_dict keys match the checkpoint exactly:
        internvl.vision_model.*
        internvl.language_model.*
        internvl.mlp1.*
        internvl.visual_projector.*
        internvl.ar_head.*
        quantizer.*
    """

    config_class = LatentUMConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoTDecoderLayer", "InternVisionEncoderLayer", "ARHeadDecoderLayer"]

    def __init__(self, config: LatentUMConfig):
        super().__init__(config)

        internvl_config = config.internvl_config
        head_cfg = config.head_config
        quant_cfg = config.quantizer_config

        # Build InternVL with MoT
        self.internvl = InternVLChatModel(internvl_config)

        # Attach visual_projector to internvl
        embedding_dim = config.embedding_dim
        llm_hidden_size = config.llm_hidden_size
        self.internvl.visual_projector = nn.Sequential(
            nn.Linear(embedding_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Attach AR head to internvl
        self.internvl.ar_head = AutoregressiveHead(
            num_codebooks=head_cfg["num_codebooks"],
            num_layers=head_cfg["num_layers"],
            hidden_size=head_cfg["hidden_size"],
            num_embeddings=head_cfg["num_embeddings"],
            num_heads=head_cfg["num_heads"],
            mlp_ratio=head_cfg["mlp_ratio"],
        )

        # Build quantizer
        self.quantizer = VQ_MLP_MCQ(
            input_feature_dim=quant_cfg["input_feature_dim"],
            embedding_dim=quant_cfg["embedding_dim"],
            llm_hidden_size=quant_cfg["llm_hidden_size"],
            num_embeddings=quant_cfg["num_embeddings"],
            num_codebooks=quant_cfg["num_codebooks"],
        )

    def get_input_embeddings(self):
        return self.internvl.language_model.model.embed_tokens

    def get_output_embeddings(self):
        return self.internvl.language_model.lm_head

    def _llm_forward(self, inputs_embeds, vision_token_mask=None, attention_mask_4d=None):
        """Forward through LLM layers with MoT routing.

        Args:
            inputs_embeds: [B, S, D] input embeddings
            vision_token_mask: [B, S] float mask (1.0=vision/gen, 0.0=text)
            attention_mask_4d: [B, 1, S, S] additive attention mask (optional)
        Returns:
            hidden_states: [B, S, D]
        """
        device = inputs_embeds.device
        B, S, D = inputs_embeds.shape
        llm_cfg = self.config.internvl_config.llm_config
        head_dim = llm_cfg.head_dim
        rope_theta = llm_cfg.rope_parameters.get("rope_theta", llm_cfg.default_theta)
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        ))
        positions = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_cos = emb.cos()[None, None, :, :]
        rope_sin = emb.sin()[None, None, :, :]

        hidden_states = inputs_embeds
        for layer in self.internvl.language_model.model.layers:
            hidden_states = layer(hidden_states, rope_cos, rope_sin, vision_token_mask, attention_mask_4d)

        hidden_states = self.internvl.language_model.model.norm(hidden_states)
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_flags=None,
        target_pixel_values=None,
        vision_token_mask=None,
        gen_token_starts=None,
        num_image_tokens=None,
        attention_mask_4d=None,
        **kwargs,
    ):
        """Training forward pass for LatentUM.

        Supports understanding SFT and generation training:
        - Understanding: standard causal LM with ViT features injected
        - Generation: VQ encode targets → dual-sequence with MoT → AR head → CE loss

        Args:
            input_ids: [B, seq_len] token IDs
            attention_mask: [B, seq_len]
            labels: [B, seq_len] with -100 for non-loss positions
            pixel_values: [B*N, C, H, W] source images for understanding
            image_flags: [B] number of images per sample
            target_pixel_values: [B*K, C, H, W] target images for generation
            vision_token_mask: [B, seq_len] float (1=vision/gen, 0=text) for MoT
            gen_token_starts: list of (batch_idx, start_pos, num_tokens) for gen blocks
            num_image_tokens: int, number of visual tokens per image (default 256)
            attention_mask_4d: [B, 1, S, S] custom 4D attention mask
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        inputs_embeds = self.internvl.language_model.model.embed_tokens(input_ids)

        # Process source images through ViT + mlp1 (understanding path)
        if pixel_values is not None and pixel_values.numel() > 0:
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
            with torch.no_grad():
                vit_features = self.internvl.vision_model(pixel_values)
                if vit_features.dim() == 3:
                    b, n, d = vit_features.shape
                    h = w = int(n ** 0.5)
                    # Remove CLS token (position 0) — official extract_feature does vit_embeds[:, 1:, :]
                    if h * w < n:
                        vit_features = vit_features[:, 1:h * w + 1, :]
                        n = h * w
                    ds = int(1 / self.config.internvl_config.downsample_ratio)
                    if h % ds == 0 and w % ds == 0:
                        vit_features = vit_features.reshape(b, h, w, d)
                        vit_features = vit_features.reshape(b, h // ds, ds, w // ds, ds, d)
                        vit_features = vit_features.permute(0, 1, 3, 2, 4, 5).reshape(b, (h // ds) * (w // ds), d * ds * ds)
            vit_projected = self.internvl.mlp1(vit_features)

        # VQ encode target images for generation
        code_tgt = None
        z_q = None
        if target_pixel_values is not None and target_pixel_values.numel() > 0:
            with torch.no_grad():
                vit_feat_tgt = self.internvl.get_vit_feature(target_pixel_values)
                z_q, code_tgt = self.quantizer.get_zq_indices(vit_feat_tgt)

        # Forward through LLM with MoT routing
        hidden_states = self._llm_forward(
            inputs_embeds,
            vision_token_mask=vision_token_mask,
            attention_mask_4d=attention_mask_4d,
        )

        # CE loss for understanding tokens
        logits = self.internvl.language_model.lm_head(hidden_states)
        ce_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = logits.shape[-1]
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # AR generation loss
        ar_loss = None
        if code_tgt is not None and gen_token_starts is not None:
            _num_img_tokens = num_image_tokens or 256
            num_codebooks = self.internvl.ar_head.num_codebooks
            num_embeddings = self.internvl.ar_head.head.out_features

            total_ar_loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
            valid_count = 0
            for batch_idx, start_pos, n_tokens in gen_token_starts:
                gen_hidden = hidden_states[batch_idx, start_pos:start_pos + _num_img_tokens, :]
                img_idx = valid_count
                code_n = code_tgt[img_idx]  # [L, K]

                BL = gen_hidden.shape[0]
                prefix = gen_hidden.unsqueeze(1)  # [L, 1, D]
                code_emb = self.internvl.ar_head._code_to_embeddings(
                    code_n.unsqueeze(0)
                )  # [L, K, D]
                h = torch.cat((prefix, code_emb), dim=1)  # [L, 1+K, D]
                ar_logits = self.internvl.ar_head(h[:, :-1, :])  # [L, K, V]

                loss_n = F.cross_entropy(
                    ar_logits.reshape(-1, num_embeddings),
                    code_n.reshape(-1),
                )
                total_ar_loss = total_ar_loss + loss_n
                valid_count += 1

            if valid_count > 0:
                ar_loss = total_ar_loss / valid_count

        # Combine losses
        loss = None
        if ce_loss is not None or ar_loss is not None:
            loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
            if ce_loss is not None:
                loss = loss + ce_loss
            if ar_loss is not None:
                loss = loss + ar_loss

        return SimpleNamespace(loss=loss, logits=logits, ar_loss=ar_loss)

    # ==================================================================
    # Inference: cached LLM forward + AR generation
    # ==================================================================

    def _get_rope_theta(self) -> float:
        """Extract rope_theta from the LLM config, handling different config layouts."""
        llm_cfg = self.config.internvl_config.llm_config
        # Try the structured rope_parameters dict first (transformers v5+)
        rp = getattr(llm_cfg, "rope_parameters", None)
        if isinstance(rp, dict):
            return float(rp.get("rope_theta", getattr(llm_cfg, "rope_theta", 1_000_000.0)))
        # Fall back to direct attribute
        return float(getattr(llm_cfg, "rope_theta", 1_000_000.0))

    def _llm_forward_cached(
        self,
        inputs_embeds: torch.Tensor,
        vision_token_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward through LLM layers with MoT routing and KV cache.

        Args:
            inputs_embeds: [B, S_new, D]
            vision_token_mask: [B, S_new] float (1.0=vision, 0.0=text)
            past_key_values: list of (k_cache, v_cache) per layer, or None.
                Each cache tensor has shape [B, num_kv_heads, S_past, head_dim].
            attention_mask: [B, S_total] (1=attend, 0=pad). S_total = S_past + S_new.
        Returns:
            hidden_states: [B, S_new, D]
            new_past_key_values: updated cache
        """
        device = inputs_embeds.device
        B, S_new, D = inputs_embeds.shape
        llm_cfg = self.config.internvl_config.llm_config
        head_dim = llm_cfg.head_dim

        past_seq_len = past_key_values[0][0].shape[2] if past_key_values else 0
        S_total = past_seq_len + S_new

        # RoPE for new positions only (cast to input dtype to avoid promotion)
        rope_theta = self._get_rope_theta()
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        ))
        positions = torch.arange(past_seq_len, S_total, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_cos = emb.cos()[None, None, :, :].to(inputs_embeds.dtype)
        rope_sin = emb.sin()[None, None, :, :].to(inputs_embeds.dtype)

        has_mot = vision_token_mask is not None and vision_token_mask.any()

        # Build attention mask (same dtype as inputs to avoid promotion)
        input_dtype = inputs_embeds.dtype
        attn_mask_4d: torch.Tensor | None = None
        if S_new > 1:
            causal = torch.triu(
                torch.ones(S_new, S_total, device=device, dtype=torch.bool),
                diagonal=past_seq_len + 1,
            )
            attn_mask_4d = torch.where(
                causal,
                torch.tensor(-65504.0, dtype=input_dtype, device=device),
                torch.tensor(0.0, dtype=input_dtype, device=device),
            )[None, None, :, :]
        if attention_mask is not None:
            pad_mask = ((1 - attention_mask[:, None, None, :].to(input_dtype)) * -65504.0)
            attn_mask_4d = pad_mask if attn_mask_4d is None else attn_mask_4d + pad_mask

        new_past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.internvl.language_model.model.layers):
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            _B, N, _D = hidden_states_norm.shape

            # Q, K, V projection with optional MoT blending
            if has_mot:
                vmask = vision_token_mask[:, None, :, None]  # [B, 1, S_new, 1]
                q_t = attn.q_proj(hidden_states_norm).view(_B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
                k_t = attn.k_proj(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                v_t = attn.v_proj(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                q_v = attn.q_proj_vision(hidden_states_norm).view(_B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
                k_v = attn.k_proj_vision(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                v_v = attn.v_proj_vision(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                q_t = attn.q_norm(q_t)
                k_t = attn.k_norm(k_t)
                q_v = attn.q_norm_vision(q_v)
                k_v = attn.k_norm_vision(k_v)
                q = vmask * q_v + (1 - vmask) * q_t
                k_new = vmask * k_v + (1 - vmask) * k_t
                v_new = vmask * v_v + (1 - vmask) * v_t
            else:
                q = attn.q_proj(hidden_states_norm).view(_B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
                k_new = attn.k_proj(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                v_new = attn.v_proj(hidden_states_norm).view(_B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                q = attn.q_norm(q)
                k_new = attn.k_norm(k_new)

            # RoPE on new positions
            q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2:]
            q = q * rope_cos + torch.cat((-q2, q1), dim=-1) * rope_sin
            k1, k2 = k_new[..., : head_dim // 2], k_new[..., head_dim // 2:]
            k_new = k_new * rope_cos + torch.cat((-k2, k1), dim=-1) * rope_sin

            # Concatenate with cached KV
            if past_key_values is not None:
                k_past, v_past = past_key_values[layer_idx]
                k_full = torch.cat([k_past, k_new], dim=2)
                v_full = torch.cat([v_past, v_new], dim=2)
            else:
                k_full = k_new
                v_full = v_new
            new_past_key_values.append((k_full, v_full))

            # GQA expansion
            if attn.num_kv_heads < attn.num_heads:
                rep = attn.num_heads // attn.num_kv_heads
                k_exp = k_full.repeat_interleave(rep, dim=1)
                v_exp = v_full.repeat_interleave(rep, dim=1)
            else:
                k_exp = k_full
                v_exp = v_full

            # Attention
            if attn_mask_4d is not None:
                attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask_4d)
            elif S_new > 1 and past_seq_len == 0:
                attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp)
            attn_out = attn_out.transpose(1, 2).reshape(_B, N, -1)

            # Output projection with MoT
            if has_mot:
                o_t = attn.o_proj(attn_out)
                o_v = attn.o_proj_vision(attn_out)
                vmask_s = vision_token_mask[:, :, None]
                hidden_states = residual + vmask_s * o_v + (1 - vmask_s) * o_t
            else:
                hidden_states = residual + attn.o_proj(attn_out)

            # MLP with MoT
            residual = hidden_states
            post_norm = layer.post_attention_layernorm(hidden_states)
            if has_mot:
                mlp_t = layer.mlp(post_norm)
                mlp_v = layer.mlp_vision(post_norm)
                vmask_s = vision_token_mask[:, :, None]
                hidden_states = residual + vmask_s * mlp_v + (1 - vmask_s) * mlp_t
            else:
                hidden_states = residual + layer.mlp(post_norm)

        hidden_states = self.internvl.language_model.model.norm(hidden_states)
        return hidden_states, new_past_key_values

    @torch.inference_mode()
    def generate_latents(
        self,
        tokenizer,
        prompts: str | list[str],
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 3.0,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        sample_logits: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate 256-position discrete latent codes from text prompts.

        Args:
            tokenizer: HF tokenizer for the model.
            prompts: single string or list of prompt strings.
            num_images_per_prompt: number of images to generate per prompt.
            cfg_scale: classifier-free guidance scale (>1 enables CFG).
            temperature, top_k, top_p: sampling parameters.
            seed: random seed for reproducibility.
            sample_logits: if False, use argmax instead of sampling.
            verbose: show progress bar.
        Returns:
            (B, 256, K) tensor of codebook indices.
        """
        from tqdm import trange

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        img_start_token = "<img>"

        if seed is not None:
            torch.manual_seed(seed)

        sampling_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_logits": sample_logits,
        }

        # Prepare prompts
        if isinstance(prompts, str):
            batch_prompts = [prompts + img_start_token] * num_images_per_prompt
        else:
            batch_prompts = [p + img_start_token for p in prompts for _ in range(num_images_per_prompt)]

        tokenizer_output = tokenizer(
            batch_prompts,
            padding=True,
            padding_side="left",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenizer_output["input_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        text_embedding = self.get_input_embeddings()(input_ids)

        # CFG: build unconditional embeddings
        if cfg_scale > 1:
            uncond_input_ids = input_ids.clone()
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            img_token_id = tokenizer.convert_tokens_to_ids(img_start_token)
            for b in range(uncond_input_ids.shape[0]):
                for t in range(uncond_input_ids.shape[1]):
                    if uncond_input_ids[b, t] == img_token_id:
                        break
                    uncond_input_ids[b, t] = pad_token_id
            uncond_text_embedding = self.get_input_embeddings()(uncond_input_ids)
            text_embedding_cfg = torch.cat([text_embedding, uncond_text_embedding], dim=0)
            attention_mask_cfg = torch.cat([attention_mask, attention_mask.clone()], dim=0)
        else:
            text_embedding_cfg = text_embedding
            attention_mask_cfg = attention_mask

        # 256-step AR generation loop
        past_key_values = None
        generated_codes: list[torch.Tensor] = []
        accumulated_attention_mask = attention_mask_cfg.clone()
        iterator = trange(256, desc="Generating latents") if verbose else range(256)

        for i in iterator:
            if i == 0:
                current_input = text_embedding_cfg
                current_attention_mask = accumulated_attention_mask
                vision_token_mask = torch.zeros(
                    current_input.shape[0], current_input.shape[1],
                    device=device, dtype=dtype,
                )
                vision_token_mask[:, -1] = 1.0  # <img> start token is vision
            else:
                if cfg_scale > 1:
                    current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
                else:
                    current_input = img_embeds_current  # noqa: F821
                accumulated_attention_mask = torch.cat([
                    accumulated_attention_mask,
                    torch.ones(
                        accumulated_attention_mask.shape[0], 1,
                        device=device, dtype=accumulated_attention_mask.dtype,
                    ),
                ], dim=1)
                current_attention_mask = accumulated_attention_mask
                vision_token_mask = torch.ones(
                    current_input.shape[0], current_input.shape[1],
                    device=device, dtype=dtype,
                )

            hidden_states, past_key_values = self._llm_forward_cached(
                current_input,
                vision_token_mask=vision_token_mask,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
            )
            base_token = hidden_states[:, -1:, :]

            generated_code = self.internvl.ar_head.generate_from_base_token(
                base_token, cfg_scale, sampling_kwargs,
            )
            z_q_current, _ = self.quantizer.indices_to_feature(generated_code.unsqueeze(1))
            img_embeds_current = self.internvl.visual_projector(z_q_current)
            generated_codes.append(generated_code)

        # generated_code from AR head is (B, K) when no CFG, or (B_half, K) with CFG
        # Stack along position dim
        return torch.stack(generated_codes, dim=1)  # (B, 256, K)

    @torch.inference_mode()
    def generate_latents_with_images(
        self,
        tokenizer,
        prompt_text: str,
        pixel_values: torch.Tensor,
        *,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate discrete latent codes conditioned on source images.

        Aligned with official world_model.py _generate_next_frame_codes().
        No CFG (cfg_scale=1.0), consistent with official it2i generation.

        Args:
            tokenizer: HF tokenizer.
            prompt_text: prompt containing <img><IMG_CONTEXT>*N</img> blocks
                for each source image. Must NOT end with the generation
                trigger <img> — that is appended automatically.
            pixel_values: [N_images, C, H, W] ImageNet-normalized source images.
            temperature, top_k, top_p: sampling parameters.
            seed: random seed.
            verbose: show progress bar.
        Returns:
            (1, num_image_tokens, K) tensor of codebook indices.
        """
        from tqdm import trange

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        num_image_tokens = self.config.num_image_tokens

        if seed is not None:
            torch.manual_seed(seed)

        sampling_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_logits": True,
        }

        # Step 1: Encode context images — ViT → pixel_shuffle → mlp1
        # (same path as forward() lines 974-987)
        vit_features = self.internvl.vision_model(pixel_values)
        if vit_features.dim() == 3:
            b, n, d = vit_features.shape
            h = w = int(n ** 0.5)
            # Remove CLS token (position 0) — official extract_feature does vit_embeds[:, 1:, :]
            if h * w < n:
                vit_features = vit_features[:, 1:h * w + 1, :]
                n = h * w
            ds = int(1 / self.config.internvl_config.downsample_ratio)
            if h % ds == 0 and w % ds == 0:
                vit_features = vit_features.reshape(b, h, w, d)
                vit_features = vit_features.reshape(b, h // ds, ds, w // ds, ds, d)
                vit_features = vit_features.permute(0, 1, 3, 2, 4, 5).reshape(
                    b, (h // ds) * (w // ds), d * ds * ds
                )
        visual_emb = self.internvl.mlp1(vit_features)  # [N_images, num_image_tokens, D]

        # Step 2: Tokenize and inject visual embeddings at <IMG_CONTEXT> positions
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")

        tokenizer_output = tokenizer(
            [prompt_text],
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        )
        input_ids = tokenizer_output["input_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        input_embeds = self.get_input_embeddings()(input_ids).clone()

        img_positions = (input_ids[0] == img_context_token_id).nonzero(as_tuple=True)[0]
        n_images = pixel_values.shape[0]
        for src_idx in range(n_images):
            start = src_idx * num_image_tokens
            end = start + num_image_tokens
            input_embeds[0, img_positions[start:end]] = visual_emb[src_idx]

        # Step 3: Prefill — vision_token_mask=0 (all text for prompt context)
        _, past_key_values = self._llm_forward_cached(
            input_embeds,
            vision_token_mask=torch.zeros(1, input_embeds.shape[1], device=device, dtype=dtype),
            past_key_values=None,
            attention_mask=attention_mask,
        )

        # Step 4: <img> start token — vision_token_mask=1 (generation trigger)
        img_embed = self.get_input_embeddings()(
            torch.tensor([[img_start_token_id]], device=device)
        )
        hidden_states, past_key_values = self._llm_forward_cached(
            img_embed,
            vision_token_mask=torch.ones(1, 1, device=device, dtype=dtype),
            past_key_values=past_key_values,
        )

        # Step 5: 256-step AR generation loop (no CFG, cfg_scale=1.0)
        generated_codes = []
        code = self.internvl.ar_head.generate_from_base_token(
            hidden_states,
            cfg_scale=1.0,
            sampling_kwargs=sampling_kwargs,
        )
        generated_codes.append(code)

        iterator = trange(num_image_tokens - 1, desc="Generating it2i latents") if verbose else range(num_image_tokens - 1)
        for _ in iterator:
            z_q, _ = self.quantizer.indices_to_feature(code.unsqueeze(1))
            current_input = self.internvl.visual_projector(z_q)
            hidden_states, past_key_values = self._llm_forward_cached(
                current_input,
                vision_token_mask=torch.ones(1, 1, device=device, dtype=dtype),
                past_key_values=past_key_values,
            )
            code = self.internvl.ar_head.generate_from_base_token(
                hidden_states,
                cfg_scale=1.0,
                sampling_kwargs=sampling_kwargs,
            )
            generated_codes.append(code)

        return torch.stack(generated_codes, dim=1)  # (1, num_image_tokens, K)

    @torch.inference_mode()
    def generate_images(
        self,
        tokenizer,
        prompts: str | list[str],
        *,
        decoder=None,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 3.0,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        show_progress: bool = False,
    ):
        """Generate images from text prompts (latent generation + decoder).

        Args:
            tokenizer: HF tokenizer.
            prompts: text prompts.
            decoder: LatentUMDecoderModel for decoding latents to pixels.
            Other args: see generate_latents and decoder.decode.
        Returns:
            list of PIL Images.
        """
        if decoder is None:
            raise ValueError("A decoder is required for generate_images().")
        latents = self.generate_latents(
            tokenizer, prompts,
            num_images_per_prompt=num_images_per_prompt,
            cfg_scale=cfg_scale, temperature=temperature,
            top_k=top_k, top_p=top_p, seed=seed,
            verbose=show_progress,
        )
        device = next(self.parameters()).device
        z_q, _ = self.quantizer.indices_to_feature(latents.to(device))
        image_size = self.config.image_size
        return decoder.decode(
            z_q, seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=image_size, width=image_size,
            show_progress=show_progress,
        )
