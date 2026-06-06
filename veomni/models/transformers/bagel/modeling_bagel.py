import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_bagel import BagelConfig, BagelLLMConfig, BagelVitConfig, BagelVaeConfig


# =============================================================================
# Utility modules
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        pos_embed = get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


# =============================================================================
# VAE (AutoEncoder)
# =============================================================================

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = swish(self.norm1(x))
        h = self.conv1(h)
        h = swish(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = nn.ModuleList()
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = self.mid.block_1(hs[-1])
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = swish(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleList()
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = swish(self.norm_out(h))
        return self.conv_out(h)


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, config: BagelVaeConfig):
        super().__init__()
        self.encoder = Encoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        self.decoder = Decoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)


# =============================================================================
# SigLIP Vision Transformer (NaViT variant with flash attention)
# =============================================================================

class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)

        grid_h = torch.arange(0, max_h).to(inv_freq.dtype)[:, None].repeat(1, max_w)
        grid_w = torch.arange(0, max_w).to(inv_freq.dtype)[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_vit_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SiglipAttention(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h=None,
        sin_h=None,
        cos_w=None,
        sin_w=None,
    ) -> torch.Tensor:
        total_q_len = hidden_states.size(0)

        query_states = self.q_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, : self.head_dim // 2], query_states[:, :, self.head_dim // 2:]
            kh, kw = key_states[:, :, : self.head_dim // 2], key_states[:, :, self.head_dim // 2:]
            qh, kh = _apply_vit_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = _apply_vit_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        try:
            from flash_attn import flash_attn_varlen_func

            attn_output = flash_attn_varlen_func(
                query_states.to(torch.bfloat16),
                key_states.to(torch.bfloat16),
                value_states.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
        except ImportError:
            # Fallback: use SDPA with manual packing
            attn_output = self._sdpa_varlen(query_states, key_states, value_states, cu_seqlens, max_seqlen)

        return self.out_proj(attn_output.reshape(total_q_len, -1))

    def _sdpa_varlen(self, q, k, v, cu_seqlens, max_seqlen):
        batch_size = cu_seqlens.shape[0] - 1
        outputs = []
        for i in range(batch_size):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            qi = q[start:end].transpose(0, 1).unsqueeze(0)
            ki = k[start:end].transpose(0, 1).unsqueeze(0)
            vi = v[start:end].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qi, ki, vi)
            outputs.append(o.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)


class SiglipMLP(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, cos_h=None, sin_h=None, cos_w=None, sin_w=None):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens, max_seqlen,
            cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size**2, config.hidden_size, bias=True
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, config.hidden_size)
        self.config = config

    def forward(self, packed_pixel_values, packed_flattened_position_ids):
        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            patch_embeds = patch_embeds + self.position_embedding(packed_flattened_position_ids)
        return patch_embeds


class SiglipEncoder(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, cu_seqlens, max_seqlen, **extra):
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen, **extra)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        hidden_states = self.embeddings(packed_pixel_values, packed_flattened_position_ids)
        extra = {}
        if self.config.rope:
            extra = dict(
                cos_h=self.rope.cos_h[packed_flattened_position_ids],
                sin_h=self.rope.sin_h[packed_flattened_position_ids],
                cos_w=self.rope.cos_w[packed_flattened_position_ids],
                sin_w=self.rope.sin_w[packed_flattened_position_ids],
            )
        hidden_states = self.encoder(hidden_states, cu_seqlens, max_seqlen, **extra)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: BagelVitConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        return self.vision_model(packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen)


# =============================================================================
# Qwen2 LLM with MoT (Mixture of Transformers) support
# =============================================================================

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: BagelLLMConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    def __init__(self, config: BagelLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PackedAttention(nn.Module):
    def __init__(self, config: BagelLLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, packed_sequence, sample_lens, packed_position_embeddings):
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        cos, sin = packed_position_embeddings
        packed_query_states, packed_key_states = _apply_rotary_pos_emb(
            packed_query_states, packed_key_states, cos, sin
        )

        try:
            from flash_attn import flash_attn_varlen_func
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(torch.tensor(sample_lens, device=packed_sequence.device, dtype=torch.int32), dim=0),
                (1, 0),
            ).to(torch.int32)
            max_seqlen = max(sample_lens)

            attn_output = flash_attn_varlen_func(
                packed_query_states.to(torch.bfloat16),
                packed_key_states.to(torch.bfloat16),
                packed_value_states.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
        except ImportError:
            # Fallback: per-sample SDPA
            packed_key_states_expanded = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_expanded = packed_key_states_expanded.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states_expanded = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states_expanded = packed_value_states_expanded.reshape(-1, self.num_heads, self.head_dim)

            q_split = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            k_split = packed_key_states_expanded.transpose(0, 1).split(sample_lens, dim=1)
            v_split = packed_value_states_expanded.transpose(0, 1).split(sample_lens, dim=1)
            outputs = []
            for qi, ki, vi in zip(q_split, k_split, v_split):
                o = F.scaled_dot_product_attention(
                    qi.unsqueeze(0).to(torch.bfloat16),
                    ki.unsqueeze(0).to(torch.bfloat16),
                    vi.unsqueeze(0).to(torch.bfloat16),
                    is_causal=True,
                )
                outputs.append(o.squeeze(0))
            attn_output = torch.cat(outputs, dim=1).transpose(0, 1)

        attn_output = attn_output.reshape(-1, self.hidden_size)
        return self.o_proj(attn_output)


class PackedAttentionMoT(nn.Module):
    def __init__(self, config: BagelLLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

    def forward(
        self,
        packed_sequence,
        sample_lens,
        packed_position_embeddings,
        packed_und_token_indexes,
        packed_gen_token_indexes,
    ):
        seq_len = packed_sequence.shape[0]
        packed_query_states = packed_sequence.new_zeros((seq_len, self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((seq_len, self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((seq_len, self.num_key_value_heads * self.head_dim))

        seq_und = packed_sequence[packed_und_token_indexes]
        seq_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(seq_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(seq_gen)
        packed_key_states[packed_und_token_indexes] = self.k_proj(seq_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(seq_gen)
        packed_value_states[packed_und_token_indexes] = self.v_proj(seq_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(seq_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply separate QK norms
        q_normed = packed_query_states.new_zeros(packed_query_states.shape)
        k_normed = packed_key_states.new_zeros(packed_key_states.shape)
        q_normed[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        q_normed[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])
        k_normed[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        k_normed[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        cos, sin = packed_position_embeddings
        q_normed, k_normed = _apply_rotary_pos_emb(q_normed, k_normed, cos, sin)

        try:
            from flash_attn import flash_attn_varlen_func
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(torch.tensor(sample_lens, device=packed_sequence.device, dtype=torch.int32), dim=0),
                (1, 0),
            ).to(torch.int32)
            max_seqlen = max(sample_lens)
            attn_output = flash_attn_varlen_func(
                q_normed.to(torch.bfloat16),
                k_normed.to(torch.bfloat16),
                packed_value_states.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
        except ImportError:
            # Fallback SDPA
            packed_key_states_expanded = k_normed[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_expanded = packed_key_states_expanded.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states_expanded = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states_expanded = packed_value_states_expanded.reshape(-1, self.num_heads, self.head_dim)

            q_split = q_normed.transpose(0, 1).split(sample_lens, dim=1)
            k_split = packed_key_states_expanded.transpose(0, 1).split(sample_lens, dim=1)
            v_split = packed_value_states_expanded.transpose(0, 1).split(sample_lens, dim=1)
            outputs = []
            for qi, ki, vi in zip(q_split, k_split, v_split):
                o = F.scaled_dot_product_attention(
                    qi.unsqueeze(0).to(torch.bfloat16),
                    ki.unsqueeze(0).to(torch.bfloat16),
                    vi.unsqueeze(0).to(torch.bfloat16),
                    is_causal=True,
                )
                outputs.append(o.squeeze(0))
            attn_output = torch.cat(outputs, dim=1).transpose(0, 1)

        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)
        output = attn_output.new_zeros((seq_len, self.hidden_size))
        output[packed_und_token_indexes] = self.o_proj(attn_output[packed_und_token_indexes])
        output[packed_gen_token_indexes] = self.o_proj_moe_gen(attn_output[packed_gen_token_indexes])
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: BagelLLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PackedAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, packed_sequence, sample_lens, packed_position_embeddings, **kwargs):
        residual = packed_sequence
        packed_sequence = self.self_attn(
            self.input_layernorm(packed_sequence), sample_lens, packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence
        residual = packed_sequence
        packed_sequence = self.mlp(self.post_attention_layernorm(packed_sequence))
        return residual + packed_sequence


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(self, config: BagelLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionMoT(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        packed_sequence,
        sample_lens,
        packed_position_embeddings,
        packed_und_token_indexes,
        packed_gen_token_indexes,
    ):
        residual = packed_sequence
        normed = packed_sequence.new_zeros(packed_sequence.shape)
        normed[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        normed[packed_gen_token_indexes] = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])

        attn_out = self.self_attn(
            normed, sample_lens, packed_position_embeddings,
            packed_und_token_indexes, packed_gen_token_indexes,
        )
        packed_sequence = residual + attn_out

        residual = packed_sequence
        mlp_out = packed_sequence.new_zeros(packed_sequence.shape)
        mlp_out[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        mlp_out[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        return residual + mlp_out


_DECODER_LAYER_DICT = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoTDecoderLayer": Qwen2MoTDecoderLayer,
}


class Qwen2Model(nn.Module):
    def __init__(self, config: BagelLLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        layer_cls = _DECODER_LAYER_DICT[config.layer_module]
        self.layers = nn.ModuleList([layer_cls(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        packed_sequence,
        sample_lens,
        attention_mask,
        packed_position_ids,
        packed_und_token_indexes=None,
        packed_gen_token_indexes=None,
    ):
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra = {}
        if self.use_moe:
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_position_ids.new_ones(size=[0])
            extra = dict(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        for layer in self.layers:
            packed_sequence = layer(packed_sequence, sample_lens, packed_position_embeddings, **extra)

        if self.use_moe:
            out = torch.zeros_like(packed_sequence)
            out[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            out[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            return out
        return self.norm(packed_sequence)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: BagelLLMConfig):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, packed_sequence, sample_lens, attention_mask, packed_position_ids, **kwargs):
        return self.model(packed_sequence, sample_lens, attention_mask, packed_position_ids, **kwargs)


# =============================================================================
# BagelForConditionalGeneration — Main model class
# =============================================================================

@dataclass
class BagelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None


class BagelForConditionalGeneration(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = "bagel"

    def __init__(self, config: BagelConfig):
        super().__init__(config)
        llm_config = config.llm_config
        vit_config = config.vit_config
        vae_config = config.vae_config

        self.language_model = Qwen2ForCausalLM(llm_config)
        self.hidden_size = llm_config.hidden_size
        self.use_moe = "Mo" in llm_config.layer_module
        self.num_heads = llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = vae_config.z_channels
            self.patch_latent_dim = config.latent_patch_size**2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(config.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = SiglipVisionModel(vit_config)
            self.vit_patch_size = vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = _get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = _get_flattened_position_ids_extrapolate

        self._init_gen_weights()

    def _init_gen_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: Optional[List[torch.Tensor]] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> BagelOutput:
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        attention_mask = nested_attention_masks

        if self.config.visual_und and packed_vit_tokens is not None:
            cu_seqlens = F.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen and padded_latent is not None:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, : h * p, : w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = (
                self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            )
            packed_latent_noisy = (
                (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            )
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent_noisy = self.vae2llm(packed_latent_noisy) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent_noisy

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse_loss = None
        if self.config.visual_gen and mse_loss_indexes is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean
            has_mse = packed_timesteps > 0
            mse_loss = (packed_mse_preds - target[has_mse]).pow(2).mean()

        ce_loss = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce_loss = F.cross_entropy(packed_ce_preds, packed_label_ids)

        loss = None
        if mse_loss is not None or ce_loss is not None:
            loss = torch.tensor(0.0, device=packed_text_ids.device)
            if mse_loss is not None:
                loss = loss + mse_loss
            if ce_loss is not None:
                loss = loss + ce_loss

        return BagelOutput(loss=loss, mse_loss=mse_loss, ce_loss=ce_loss)


# =============================================================================
# Position ID helpers
# =============================================================================

def _get_flattened_position_ids_extrapolate(H, W, patch_size, max_num_patches_per_side):
    h = H // patch_size
    w = W // patch_size
    h_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    w_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    position_ids = h_ids * max_num_patches_per_side + w_ids
    return position_ids.flatten()


def _get_flattened_position_ids_interpolate(H, W, patch_size, max_num_patches_per_side):
    h = H // patch_size
    w = W // patch_size
    h_ids = torch.linspace(0, max_num_patches_per_side - 1, h).long().unsqueeze(1).expand(-1, w)
    w_ids = torch.linspace(0, max_num_patches_per_side - 1, w).long().unsqueeze(0).expand(h, -1)
    position_ids = h_ids * max_num_patches_per_side + w_ids
    return position_ids.flatten()
