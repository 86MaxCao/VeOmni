"""
BLIP3o model implementation for VeOmni framework.

This model defines the full architecture that matches the BLIP3o checkpoint weight keys:
- visual.*  (Qwen2.5-VL style vision encoder with merger)
- model.embed_tokens, model.layers.*, model.norm  (Qwen2 LLM backbone)
- model.dit.model.*  (Diffusion Transformer for image generation)
- model.vae.*  (VAE encoder/decoder)
- model.gen_vision_tower.*  (EVA-CLIP vision tower for generation)
- model.latent_queries  (learnable latent queries)
- lm_head  (language model head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_blip3o import BLIP3oConfig


# =============================================================================
# Qwen2.5-VL Vision Encoder (understanding pathway)
# Produces keys: visual.patch_embed.*, visual.blocks.N.*, visual.merger.*
# =============================================================================


class VisionPatchEmbed(nn.Module):
    """3D patch embedding for video/image input."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, temporal_patch_size: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )


class VisionAttention(nn.Module):
    """Attention module for vision blocks with fused QKV."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = 128
        self.num_heads = hidden_size // self.head_dim
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


class VisionMLP(nn.Module):
    """SiLU-gated MLP for vision blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nn.Module):
    """Single vision transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, bias=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, bias=False)
        self.attn = VisionAttention(hidden_size)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionMerger(nn.Module):
    """Merges vision tokens into LLM hidden space."""

    def __init__(self, hidden_size: int, out_hidden_size: int, spatial_merge_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(hidden_size, elementwise_affine=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(merge_dim, merge_dim, bias=True),
            nn.GELU(),
            nn.Linear(merge_dim, out_hidden_size, bias=True),
        )

    def forward(self, hidden_states, grid_hw=None):
        # hidden_states: [B, N, C] where N = h_patches * w_patches
        hidden_states = self.ln_q(hidden_states)
        B, N, C = hidden_states.shape
        s = self.spatial_merge_size
        # Determine spatial dimensions
        if grid_hw is not None:
            h, w = grid_hw
        else:
            h = w = int(N ** 0.5)
            if h * w != N:
                # Find h, w such that h*w == N and both divisible by s
                for hh in range(int(N**0.5), 0, -1):
                    if N % hh == 0:
                        ww = N // hh
                        if hh % s == 0 and ww % s == 0:
                            h, w = hh, ww
                            break
        # Merge spatial_merge_size x spatial_merge_size patches
        if h % s == 0 and w % s == 0 and h * w == N:
            hidden_states = hidden_states.reshape(B, h, w, C)
            hidden_states = hidden_states.reshape(B, h // s, s, w // s, s, C)
            hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(B, (h // s) * (w // s), C * s * s)
        else:
            # Fallback: linear grouping
            n_merged = N // (s * s)
            hidden_states = hidden_states[:, :n_merged * s * s, :].reshape(B, n_merged, C * s * s)
        return self.mlp(hidden_states)


class VisionEncoder(nn.Module):
    """
    Qwen2.5-VL style vision encoder.
    Produces state_dict keys: patch_embed.proj.weight, blocks.N.*, merger.*
    """

    def __init__(self, config):
        super().__init__()
        vc = config.vision_config
        self.hidden_size = vc.hidden_size
        self.patch_embed = VisionPatchEmbed(
            in_channels=vc.in_channels,
            hidden_size=vc.hidden_size,
            patch_size=vc.spatial_patch_size,
            temporal_patch_size=vc.temporal_patch_size,
        )
        self.blocks = nn.ModuleList([
            VisionBlock(vc.hidden_size, vc.intermediate_size)
            for _ in range(vc.depth)
        ])
        self.merger = VisionMerger(vc.hidden_size, vc.out_hidden_size, vc.spatial_merge_size)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, C, T, H, W] or [B, C, H, W] image tensor
        Returns:
            merged vision embeddings [B, num_merged_tokens, out_hidden_size]
        """
        if pixel_values.dim() == 4:
            # [B, C, H, W] -> [B, C, 2, H, W] (duplicate for temporal_patch_size=2)
            pixel_values = pixel_values.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        # Patch embed: Conv3D -> [B, hidden_size, t, h, w]
        x = self.patch_embed.proj(pixel_values)
        # Get spatial grid dims before flattening
        _, _, t, h_out, w_out = x.shape
        # Flatten spatial dims: [B, hidden_size, t*h*w] -> [B, t*h*w, hidden_size]
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Merger — pass grid dimensions for non-square handling
        grid_h = t * h_out
        grid_w = w_out
        x = self.merger(x, grid_hw=(grid_h, grid_w))
        return x


# =============================================================================
# Qwen2 LLM Backbone
# Produces keys: embed_tokens.*, layers.N.*, norm.*
# =============================================================================


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2Attention(nn.Module):
    """Qwen2 attention with separate Q/K/V projections (biases on Q/K/V, no bias on O)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, rms_norm_eps: float):
        super().__init__()
        self.self_attn = Qwen2Attention(hidden_size, num_heads, num_kv_heads)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, rms_norm_eps)


# =============================================================================
# DIT (Diffusion Transformer) for image generation
# Produces keys: model.dit.model.*
# =============================================================================


class DiTSelfAttention(nn.Module):
    """Self-attention (attn1) in DIT blocks - no output projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm_k = nn.LayerNorm(dim, elementwise_affine=True)


class DiTCrossAttention(nn.Module):
    """Cross-attention (attn2) in DIT blocks - has output projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm_k = nn.LayerNorm(dim, elementwise_affine=True)


class DiTFeedForward(nn.Module):
    """GLU-style feed forward for DIT."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, ffn_dim, bias=False)
        self.linear_2 = nn.Linear(ffn_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, ffn_dim, bias=False)


class DiTAdaLayerNorm(nn.Module):
    """Adaptive layer norm with time conditioning."""

    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        # norm1.norm.weight -> LayerNorm weight
        self.norm = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        # norm1.linear.{weight, bias} -> projects time embedding to scale/shift
        # Output dim is 4*dim for (scale1, shift1, scale2, shift2) = but checkpoint shows 7168 = 4*1792
        # Actually 7168/1024 = 7, so it's from time_embed_dim to some multiple
        # Looking at checkpoint: norm1.linear.weight: [7168, 1024], norm1.linear.bias: [7168]
        # 7168 = 4 * 1792 (scale_self, shift_self, gate_self, gate_cross... or similar)
        self.linear = nn.Linear(time_embed_dim, dim * 4, bias=True)


class DiTLayer(nn.Module):
    """Single DIT transformer layer."""

    def __init__(self, dim: int, ffn_dim: int, time_embed_dim: int, num_heads: int):
        super().__init__()
        self.attn1 = DiTSelfAttention(dim)
        self.attn2 = DiTCrossAttention(dim)
        self.feed_forward = DiTFeedForward(dim, ffn_dim)
        self.norm1 = DiTAdaLayerNorm(dim, time_embed_dim)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        self.ffn_norm1 = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        self.ffn_norm2 = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        self.gate = nn.Parameter(torch.zeros(num_heads))


class DiTTimeCaptionEmbed(nn.Module):
    """Timestep and caption embedding module."""

    def __init__(self, dim: int, time_embed_dim: int, timestep_input_dim: int):
        super().__init__()
        # timestep_embedder projects sinusoidal timestep to time_embed_dim
        self.timestep_embedder = DiTTimestepEmbedder(time_embed_dim, timestep_input_dim)
        # caption_embedder: LayerNorm + Linear
        self.caption_embedder = nn.ModuleList([
            nn.LayerNorm(dim, elementwise_affine=True),
            nn.Linear(dim, time_embed_dim, bias=True),
        ])


class DiTTimestepEmbedder(nn.Module):
    def __init__(self, time_embed_dim: int, timestep_input_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(timestep_input_dim, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)


class DiTCaptionProjection(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)


class DiTNormOut(nn.Module):
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(time_embed_dim, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)


class DiTPatchEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # patch_embedder projects latent_channels (after patchify) to dim
        # From checkpoint: [1792, 1792] meaning input_dim == output_dim
        self.proj = nn.Linear(dim, dim, bias=True)


class DiTModel(nn.Module):
    """
    The inner DIT model.
    Produces keys at: caption_projection.*, layers.N.*, norm_out.*,
                      patch_embedder.*, time_caption_embed.*
    """

    def __init__(self, config):
        super().__init__()
        dim = config.dit_hidden_size
        time_embed_dim = config.dit_time_embed_dim
        timestep_input_dim = config.dit_timestep_input_dim
        ffn_dim = config.dit_ffn_hidden_size
        num_layers = config.dit_num_layers
        num_heads = config.dit_num_heads

        self.patch_embedder = DiTPatchEmbedder(dim)
        self.time_caption_embed = DiTTimeCaptionEmbed(dim, time_embed_dim, timestep_input_dim)
        self.caption_projection = DiTCaptionProjection(config.hidden_size, dim)
        self.layers = nn.ModuleList([
            DiTLayer(dim, ffn_dim, time_embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm_out = DiTNormOut(dim, time_embed_dim)


class DiTWrapper(nn.Module):
    """
    Wrapper to produce keys at: model.*
    So full path from top-level is: model.dit.model.*
    """

    def __init__(self, config):
        super().__init__()
        self.model = DiTModel(config)


# =============================================================================
# VAE (Variational Autoencoder)
# Produces keys: model.vae.encoder.*, model.vae.decoder.*
# =============================================================================


class VAEResnetBlock(nn.Module):
    """Standard VAE resnet block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, has_shortcut: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if has_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)


class VAEAttentionBlock(nn.Module):
    """Self-attention block in VAE mid_block."""

    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True)])


class VAEMidBlock(nn.Module):
    """Mid block with attention and resnets."""

    def __init__(self, channels: int):
        super().__init__()
        self.attentions = nn.ModuleList([VAEAttentionBlock(channels)])
        self.resnets = nn.ModuleList([
            VAEResnetBlock(channels, channels),
            VAEResnetBlock(channels, channels),
        ])


class VAEDownBlock(nn.Module):
    """Encoder down block with resnets and optional downsampler."""

    def __init__(self, in_channels: int, out_channels: int, num_resnets: int = 2, has_downsampler: bool = True):
        super().__init__()
        resnets = []
        for i in range(num_resnets):
            ch_in = in_channels if i == 0 else out_channels
            has_shortcut = (ch_in != out_channels)
            resnets.append(VAEResnetBlock(ch_in, out_channels, has_shortcut=has_shortcut))
        self.resnets = nn.ModuleList(resnets)
        if has_downsampler:
            self.downsamplers = nn.ModuleList([VAEDownsampler(out_channels)])


class VAEDownsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)


class VAEUpBlock(nn.Module):
    """Decoder up block with resnets and optional upsampler."""

    def __init__(self, in_channels: int, out_channels: int, num_resnets: int = 3, has_upsampler: bool = True):
        super().__init__()
        resnets = []
        for i in range(num_resnets):
            ch_in = in_channels if i == 0 else out_channels
            has_shortcut = (ch_in != out_channels)
            resnets.append(VAEResnetBlock(ch_in, out_channels, has_shortcut=has_shortcut))
        self.resnets = nn.ModuleList(resnets)
        if has_upsampler:
            self.upsamplers = nn.ModuleList([VAEUpsampler(out_channels)])


class VAEUpsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)


class VAEEncoder(nn.Module):
    """VAE Encoder: 3 -> latent_channels*2 (mean + logvar)."""

    def __init__(self):
        super().__init__()
        # in_channels=3, block_out_channels=(128, 256, 512, 512)
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        self.down_blocks = nn.ModuleList([
            VAEDownBlock(128, 128, num_resnets=2, has_downsampler=True),
            VAEDownBlock(128, 256, num_resnets=2, has_downsampler=True),
            VAEDownBlock(256, 512, num_resnets=2, has_downsampler=True),
            VAEDownBlock(512, 512, num_resnets=2, has_downsampler=False),
        ])
        self.mid_block = VAEMidBlock(512)
        self.conv_norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 32, 3, padding=1)  # 32 = latent_channels * 2


class VAEDecoder(nn.Module):
    """VAE Decoder: latent_channels -> 3."""

    def __init__(self):
        super().__init__()
        # latent_channels=16, block_out_channels reversed = (512, 512, 256, 128)
        self.conv_in = nn.Conv2d(16, 512, 3, padding=1)
        self.up_blocks = nn.ModuleList([
            VAEUpBlock(512, 512, num_resnets=3, has_upsampler=True),
            VAEUpBlock(512, 512, num_resnets=3, has_upsampler=True),
            VAEUpBlock(512, 256, num_resnets=3, has_upsampler=True),
            VAEUpBlock(256, 128, num_resnets=3, has_upsampler=False),
        ])
        self.mid_block = VAEMidBlock(512)
        self.conv_norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)


class VAE(nn.Module):
    """Full VAE with encoder and decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()


# =============================================================================
# EVA-CLIP Vision Tower for Generation
# Produces keys: model.gen_vision_tower.vision_tower.model.*
# =============================================================================


class EVAAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)


class EVAMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=True)


class EVABlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = EVAAttention(dim)
        self.mlp = EVAMLP(dim, mlp_dim)


class EVAModel(nn.Module):
    """
    EVA-CLIP model for generation pathway.
    Produces keys at: cls_token, pos_embed, patch_embed.proj.*, blocks.N.*
    """

    def __init__(self, config):
        super().__init__()
        dim = config.gen_vit_hidden_size
        num_blocks = config.gen_vit_num_blocks
        mlp_dim = int(dim * config.gen_vit_mlp_ratio)
        patch_size = config.gen_vit_patch_size
        # pos_embed: 1025 = (image_size/patch_size)^2 + 1 (cls_token)
        num_patches = (config.gen_vit_image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.patch_embed = EVAPatchEmbed(dim, patch_size)
        self.blocks = nn.ModuleList([
            EVABlock(dim, mlp_dim) for _ in range(num_blocks)
        ])


class EVAPatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=True)


class EVAVisionTower(nn.Module):
    """Wrapper: vision_tower.model.* """

    def __init__(self, config):
        super().__init__()
        self.model = EVAModel(config)


class GenVisionTowerWrapper(nn.Module):
    """Wrapper: gen_vision_tower.vision_tower.* """

    def __init__(self, config):
        super().__init__()
        self.vision_tower = EVAVisionTower(config)


# =============================================================================
# BLIP3o Qwen Model (inner model combining LLM + generation modules)
# =============================================================================


class BLIP3oQwenModel(nn.Module):
    """
    Inner model that holds:
    - embed_tokens, layers, norm (LLM)
    - dit (diffusion transformer)
    - vae (variational autoencoder)
    - gen_vision_tower (EVA-CLIP for generation)
    - latent_queries (learnable queries)
    """

    def __init__(self, config):
        super().__init__()
        # LLM components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Generation components
        self.dit = DiTWrapper(config)
        self.vae = VAE()
        self.gen_vision_tower = GenVisionTowerWrapper(config)
        self.latent_queries = nn.Parameter(torch.zeros(1, config.n_query, config.hidden_size))


# =============================================================================
# BLIP3o Top-Level Model
# =============================================================================


class BLIP3oQwenForCausalLM(PreTrainedModel):
    """
    BLIP3o multimodal model for causal language modeling with image understanding
    and generation capabilities.

    State dict structure:
    - visual.* (Qwen2.5-VL vision encoder)
    - model.* (LLM + generation modules)
    - lm_head.* (output projection)
    """

    config_class = BLIP3oConfig

    def __init__(self, config: BLIP3oConfig):
        super().__init__(config)

        # Vision encoder for understanding
        self.visual = VisionEncoder(config)

        # Inner model (LLM + generation)
        self.model = BLIP3oQwenModel(config)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        """Training forward pass for BLIP3o.

        Standard causal LM SFT: embed tokens, forward through decoder layers,
        compute cross-entropy loss on labeled positions.

        For understanding mode, visual features from the Qwen2.5-VL ViT are
        injected at image placeholder positions in the input sequence (handled
        by data collator). For pure language SFT, no pixel_values needed.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            labels: [B, seq_len] with -100 for non-loss tokens
            pixel_values: optional [B, C, H, W] for understanding
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        inputs_embeds = self.model.embed_tokens(input_ids)

        # Forward through transformer layers with causal attention
        hidden_states = inputs_embeds
        for layer in self.model.layers:
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm(hidden_states)
            # Self-attention
            attn = layer.self_attn
            B, N, D = hidden_states_norm.shape
            q = attn.q_proj(hidden_states_norm).view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states_norm).view(B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states_norm).view(B, N, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            # GQA repeat
            if attn.num_kv_heads < attn.num_heads:
                rep = attn.num_heads // attn.num_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
            hidden_states = residual + attn.o_proj(attn_out)
            # MLP
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)
