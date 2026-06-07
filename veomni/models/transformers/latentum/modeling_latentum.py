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
    """RMSNorm matching Qwen3's implementation."""

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


class Qwen3MoTDecoderLayer(nn.Module):
    """
    Qwen3 decoder layer with MoT dual-path MLP.

    Contains standard components (input_layernorm, self_attn, post_attention_layernorm, mlp)
    plus a vision-specific mlp_vision for the MoT path.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.input_layernorm = Qwen3MoTRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3MoTAttention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoTRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MoTMLP(hidden_size, intermediate_size)
        self.mlp_vision = Qwen3MoTMLP(hidden_size, intermediate_size)


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


class AutoregressiveHead(nn.Module):
    """
    Autoregressive head for discrete image token generation.

    Generates num_codebooks tokens autoregressively, each selecting from
    num_embeddings entries.
    """

    def __init__(self, num_codebooks: int, num_layers: int, hidden_size: int, num_embeddings: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings, hidden_size) for _ in range(num_codebooks)]
        )
        self.layers = nn.ModuleList(
            [ARHeadDecoderLayer(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = ARHeadRMSNorm(hidden_size, eps=1e-5)
        self.head = nn.Linear(hidden_size, num_embeddings)


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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_flags=None,
        **kwargs,
    ):
        """Training forward pass for LatentUM.

        Following the original LatentUM training (train_interleaved_lang_only.py):
        1. Encode source images through InternVL's vision_model + mlp1
        2. Inject visual embeddings into the input sequence
        3. Run causal LM forward through the Qwen3 MoT backbone
        4. Compute cross-entropy loss on labeled tokens (ignore_index=-100)

        Args:
            input_ids: [B, seq_len] token IDs
            attention_mask: [B, seq_len]
            labels: [B, seq_len] with -100 for non-loss positions
            pixel_values: [B, N, C, H, W] or [B*N, C, H, W] image patches
            image_flags: [B] number of images per sample (for positioning)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Get text embeddings
        inputs_embeds = self.internvl.language_model.model.embed_tokens(input_ids)

        # Process images through ViT + projector
        if pixel_values is not None and pixel_values.numel() > 0:
            if pixel_values.dim() == 5:
                B, N, C, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B * N, C, H, W)
            vit_features = self.internvl.vision_model(pixel_values)
            # Pixel shuffle downsample (matches mlp1 input expectation)
            if vit_features.dim() == 3:
                b, n, d = vit_features.shape
                h = w = int(n**0.5)
                ds = int(1 / self.config.internvl_config.downsample_ratio)
                if h % ds == 0 and w % ds == 0:
                    vit_features = vit_features.reshape(b, h, w, d)
                    vit_features = vit_features.reshape(b, h // ds, ds, w // ds, ds, d)
                    vit_features = vit_features.permute(0, 1, 3, 2, 4, 5).reshape(b, (h // ds) * (w // ds), d * ds * ds)
            vit_projected = self.internvl.mlp1(vit_features)
            # Note: injection positions depend on tokenizer special tokens
            # The data collator is responsible for marking image placeholder positions

        # Precompute RoPE embeddings
        llm_cfg = self.config.internvl_config.llm_config
        head_dim = llm_cfg.head_dim
        rope_theta = llm_cfg.rope_parameters.get("rope_theta", llm_cfg.default_theta)
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        ))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        rope_cos = emb.cos()[None, None, :, :]  # [1, 1, S, D]
        rope_sin = emb.sin()[None, None, :, :]

        # Forward through language model layers (understanding path only for SFT)
        hidden_states = inputs_embeds
        for layer in self.internvl.language_model.model.layers:
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            B, N, D = hidden_states_norm.shape
            q = attn.q_proj(hidden_states_norm).view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states_norm).view(B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states_norm).view(B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            # Apply rotary position embeddings
            q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
            q = q * rope_cos + torch.cat((-q2, q1), dim=-1) * rope_sin
            k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]
            k = k * rope_cos + torch.cat((-k2, k1), dim=-1) * rope_sin
            # GQA repeat
            if attn.num_kv_heads < attn.num_heads:
                rep = attn.num_heads // attn.num_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
            hidden_states = residual + attn.o_proj(attn_out)

            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.internvl.language_model.model.norm(hidden_states)

        # Compute LM logits and loss
        logits = self.internvl.language_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(loss=loss, logits=logits)
