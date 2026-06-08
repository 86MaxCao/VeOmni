"""
NEOChatModel (SenseNova-U1) implementation for VeOmni.

Architecture:
  - Vision encoder: NEOVisionModel (lightweight ViT with 2D-RoPE and downsampling)
  - LLM backbone: Qwen3 dense 8B with MoT (Mixture of Transformers)
    - Each decoder layer has dual paths: understanding (und) + generation (gen)
    - Gen path uses *_mot_gen suffixed weights for attention/MLP/norms
  - Flow-Matching head: timestep embedder + generation ViT + FM head (nn.Sequential)

The model uses a custom 3D positional encoding (t, h, w) for the LLM backbone,
enabling both temporal (causal text) and spatial (image patch) position awareness.
"""

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_neo_chat import NEOChatConfig, NEOLLMConfig, NEOVisionConfig

# ---------------------------------------------------------------------------
# Constants for generation inference
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# Attention backend utilities
# ---------------------------------------------------------------------------


def _sdpa_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """SDPA fallback for flash_attn_func. Input layout: [B, S, H, D]."""
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)

    h_q = q_bhsd.shape[1]
    h_kv = k_bhsd.shape[1]
    if h_q != h_kv:
        n_rep = h_q // h_kv
        k_bhsd = k_bhsd.repeat_interleave(n_rep, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(n_rep, dim=1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
    )
    return out.transpose(1, 2).contiguous()


def _flash_or_sdpa(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """Dispatch between flash-attn and SDPA based on availability."""
    if _HAS_FLASH_ATTN and q.device.type == "cuda":
        return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    return _sdpa_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)


# ---------------------------------------------------------------------------
# Vision Encoder (NEOVisionModel)
# ---------------------------------------------------------------------------


def _precompute_rope_freqs_sincos(dim, max_position, base=10000.0, device=None):
    """Precompute 1D RoPE cos/sin values."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _build_abs_positions_from_grid_hw(grid_hw, device=None):
    """Compute per-patch (x, y) coordinates from a grid_hw tensor of shape (B, 2)."""
    device = grid_hw.device
    B = grid_hw.shape[0]
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch
    return abs_x, abs_y


# ---------------------------------------------------------------------------
# Generation inference helpers
# ---------------------------------------------------------------------------


def create_block_causal_mask(index: torch.Tensor):
    """Create block-wise causal attention mask from temporal index tensor.

    Args:
        index: (L,) temporal index for each token.
    Returns:
        (1, 1, L, L) mask: 0 where attention allowed, -inf where blocked.
    """
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)
    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))
    return torch.where(mask[None, None, :, :] > 0, torch.tensor(0.0, device=index.device), torch.tensor(float("-inf"), device=index.device))


def prepare_flash_kv_cache(past_key_values, current_len: int, batch_size: int):
    """Convert prefix KV cache to flash-attn layout and preallocate buffers.

    Stores flash caches as ``_flash_k_caches`` / ``_flash_v_caches`` on the
    DynamicCache object.  Called once before the denoising loop.
    """
    if past_key_values is None:
        return
    num_layers = len(past_key_values.layers)
    flash_k = []
    flash_v = []
    flash_prefix = []
    for li in range(num_layers):
        past_k = past_key_values.layers[li].keys
        past_v = past_key_values.layers[li].values
        if past_k.numel() == 0:
            flash_k.append(None)
            flash_v.append(None)
            flash_prefix.append(0)
            continue
        pk = past_k.transpose(1, 2).contiguous()  # [B, S, H, D]
        pv = past_v.transpose(1, 2).contiguous()
        prefix_len = pk.shape[1]
        total_len = prefix_len + current_len
        kc = torch.empty((batch_size, total_len, pk.shape[2], pk.shape[3]), device=pk.device, dtype=pk.dtype)
        vc = torch.empty((batch_size, total_len, pv.shape[2], pv.shape[3]), device=pv.device, dtype=pv.dtype)
        kc[:, :prefix_len].copy_(pk)
        vc[:, :prefix_len].copy_(pv)
        flash_k.append(kc)
        flash_v.append(vc)
        flash_prefix.append(prefix_len)
    past_key_values._flash_k_caches = flash_k
    past_key_values._flash_v_caches = flash_v
    past_key_values._flash_prefix_lens = flash_prefix


def clear_flash_kv_cache(past_key_values):
    """Remove preallocated flash KV buffers after denoising."""
    if past_key_values is None:
        return
    for attr in ("_flash_k_caches", "_flash_v_caches", "_flash_prefix_lens"):
        if hasattr(past_key_values, attr):
            delattr(past_key_values, attr)


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {max(height, width) / min(height, width)}")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def load_image_native(
    image,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    """Load and preprocess image: RGB convert → smart-resize → normalize → patchify."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    image = image.convert("RGB")
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    size_factor = int(patch_size // downsample_ratio)
    w, h = image.size
    rh, rw = _smart_resize(h, w, factor=size_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((rw, rh))

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    pixel_values = transform(image).to(torch.float32)

    c, ph, pw = pixel_values.shape
    grid_h = ph // patch_size
    grid_w = pw // patch_size
    flatten_pv = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size ** 2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]])
    return flatten_pv, grid_hw


def _apply_rotary_emb_1d(x, cos_cached, sin_cached, positions):
    """Apply 1D RoPE to input tensor."""
    cos = cos_cached[positions]
    sin = sin_cached[positions]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def _apply_2d_rotary_pos_emb(x, cos_x, sin_x, cos_y, sin_y, abs_x, abs_y):
    """Apply 2D RoPE (x-axis + y-axis) to input tensor."""
    dim_half = x.shape[-1] // 2
    x_part1 = x[..., :dim_half]
    x_part2 = x[..., dim_half:]
    rotated1 = _apply_rotary_emb_1d(x_part1, cos_x, sin_x, abs_x)
    rotated2 = _apply_rotary_emb_1d(x_part2, cos_y, sin_y, abs_y)
    return torch.cat((rotated1, rotated2), dim=-1)


class NEOVisionEmbeddings(nn.Module):
    """Patch embedding + 2D-RoPE + downsampling for NEO Vision.

    The RoPE cos/sin buffers are recomputed on first forward if they are found
    to be all-zero (which happens when the model is materialised from
    meta-device via ``init_empty_weights`` + ``to_empty``).
    """

    def __init__(self, config: NEOVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size[0]
        self.downsample_factor = int(1 / config.downsample_ratio[0])
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

        self.rope_dim_part = self.embed_dim // 2
        cos_x, sin_x = _precompute_rope_freqs_sincos(
            self.rope_dim_part, config.max_position_embeddings_vision, base=config.rope_theta_vision
        )
        cos_y, sin_y = _precompute_rope_freqs_sincos(
            self.rope_dim_part, config.max_position_embeddings_vision, base=config.rope_theta_vision
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    def _ensure_rope_buffers(self, device: torch.device) -> None:
        """Reinitialise RoPE cos/sin buffers if they were zeroed or corrupted by meta-device loading."""
        if (
            self.cos_cached_x is not None
            and self.cos_cached_x.abs().sum() > 0
            and self.cos_cached_x.abs().max() <= 1.0
        ):
            return
        cos_x, sin_x = _precompute_rope_freqs_sincos(
            self.rope_dim_part,
            self.config.max_position_embeddings_vision,
            base=self.config.rope_theta_vision,
            device=device,
        )
        cos_y, sin_y = _precompute_rope_freqs_sincos(
            self.rope_dim_part,
            self.config.max_position_embeddings_vision,
            base=self.config.rope_theta_vision,
            device=device,
        )
        self.cos_cached_x = cos_x
        self.sin_cached_x = sin_x
        self.cos_cached_y = cos_y
        self.sin_cached_y = sin_y

    def forward(self, pixel_values: torch.FloatTensor, grid_hw=None) -> torch.Tensor:
        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)

        # Ensure RoPE buffers are initialised (may be zeroed after meta-device loading)
        self._ensure_rope_buffers(patch_embeds.device)

        # Apply 2D RoPE
        abs_pos_x, abs_pos_y = _build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        patch_embeds = _apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),
            self.cos_cached_x.to(patch_embeds.device),
            self.sin_cached_x.to(patch_embeds.device),
            self.cos_cached_y.to(patch_embeds.device),
            self.sin_cached_y.to(patch_embeds.device),
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)

        # Downsample via dense_embedding conv
        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.reshape(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)
        return embeddings


class NEOVisionModel(PreTrainedModel):
    """NEO Vision encoder model."""

    config_class = NEOVisionConfig
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: NEOVisionConfig):
        super().__init__(config)
        self.embeddings = NEOVisionEmbeddings(config)

    def forward(self, pixel_values=None, output_hidden_states=None, return_dict=None, grid_hw=None):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided")
        hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)
        return type("Output", (), {"last_hidden_state": hidden_states})()


# ---------------------------------------------------------------------------
# Flow-Matching Modules
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000.0):
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
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


# ---------------------------------------------------------------------------
# LLM Backbone: Qwen3 with MoT (Mixture of Transformers)
# ---------------------------------------------------------------------------


class Qwen3RMSNorm(nn.Module):
    """RMSNorm as used in Qwen3."""

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


class Qwen3MLP(nn.Module):
    """Gated MLP as used in Qwen3."""

    def __init__(self, config: NEOLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k. cos/sin shape: [1, 1, S, D]."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_rope_parameters(config, device=None):
    """Compute default RoPE inverse frequencies."""
    base = config.rope_theta
    head_dim = config.head_dim
    dim = int(head_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary position embedding for the LLM backbone.

    Uses a special frequency computation that effectively doubles the head_dim
    before selecting every other frequency, matching the original U1 implementation.

    The ``inv_freq`` buffer is recomputed on first forward if it is found to be
    all-zero (which happens when the model is materialised from meta-device via
    ``init_empty_weights`` + ``to_empty``).
    """

    def __init__(self, config: NEOLLMConfig, rope_theta=None, max_position_embeddings=None, head_dim_override=None):
        super().__init__()
        self.config = config
        self._effective_head_dim = head_dim_override if head_dim_override is not None else config.head_dim
        self._effective_theta = rope_theta if rope_theta is not None else config.rope_theta
        effective_max_pos = max_position_embeddings if max_position_embeddings is not None else config.max_position_embeddings

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = effective_max_pos

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies using the keep-freq-range pattern."""
        doubled_dim = self._effective_head_dim * 2
        full_inv_freq = 1.0 / (
            self._effective_theta ** (torch.arange(0, doubled_dim, 2, dtype=torch.float32) / doubled_dim)
        )
        return full_inv_freq[::2]

    def _ensure_inv_freq(self, device: torch.device) -> None:
        """Reinitialise ``inv_freq`` if it was zeroed by meta-device loading."""
        if self.inv_freq is not None and self.inv_freq.numel() > 0 and self.inv_freq.abs().sum() > 0:
            return
        self.inv_freq = self._compute_inv_freq().to(device=device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: tensor for dtype/device reference
            position_ids: [B, S] or [1, S] position indices
        Returns:
            cos, sin: [B, 1, S, D] tensors
        """
        self._ensure_inv_freq(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.unsqueeze(1).to(dtype=x.dtype), sin.unsqueeze(1).to(dtype=x.dtype)


class Qwen3Attention(nn.Module):
    """Multi-head attention with MoT dual-path (understanding + generation).

    Each path has its own q/k/v/o projections and q/k norms.
    Both paths share the same RoPE embeddings.
    """

    def __init__(self, config: NEOLLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Understanding path projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Generation path projections (MoT)
        self.q_proj_mot_gen = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_mot_gen = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_mot_gen = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_mot_gen = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # QK norms (applied to half of head_dim for temporal part)
        half_head = self.head_dim // 2
        self.q_norm = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.q_norm_mot_gen = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)

        # QK norms for spatial (h, w) part
        self.q_norm_hw = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)
        self.k_norm_hw_mot_gen = Qwen3RMSNorm(half_head, eps=config.rms_norm_eps)

        # Temporal RoPE (half head_dim)
        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb = Qwen3RotaryEmbedding(config, head_dim_override=config.head_dim // 2)

        # Spatial RoPE (quarter head_dim, different theta)
        self.rotary_emb_hw = Qwen3RotaryEmbedding(
            config,
            rope_theta=config.rope_theta_hw,
            max_position_embeddings=config.max_position_embeddings_hw,
            head_dim_override=config.head_dim // 4,
        )

    def forward(self, hidden_states, indexes, attention_mask, past_key_values=None, **kwargs):
        """Understanding-path forward (default)."""
        return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)

    def forward_und(self, hidden_states, indexes, attention_mask, past_key_values=None, **kwargs):
        """Forward pass for understanding path."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = self.q_norm(query_states_t).transpose(1, 2)
        query_states_hw = self.q_norm_hw(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = self.k_norm(key_states_t).transpose(1, 2)
        key_states_hw = self.k_norm_hw(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply temporal RoPE
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = _apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

        # Apply spatial RoPE (h)
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = _apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

        # Apply spatial RoPE (w)
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = _apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        # Concatenate all parts
        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Eager attention
        key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states_expanded.shape[-2]]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_expanded)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def forward_gen(self, hidden_states, indexes, attention_mask, past_key_values=None, **kwargs):
        """Forward pass for generation path (uses _mot_gen weights)."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = self.q_norm_mot_gen(query_states_t).transpose(1, 2)
        query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)
        key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply temporal RoPE
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = _apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

        # Apply spatial RoPE (h)
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = _apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

        # Apply spatial RoPE (w)
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = _apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        update_cache = kwargs.get("update_cache", True)

        # Flash attention path for generation (bidirectional within image block)
        if attention_mask is None:
            q = query_states.transpose(1, 2).contiguous()
            k_cur = key_states.transpose(1, 2).contiguous()
            v_cur = value_states.transpose(1, 2).contiguous()

            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()
                else:
                    if (
                        hasattr(past_key_values, "_flash_k_caches")
                        and past_key_values._flash_k_caches[self.layer_idx] is not None
                    ):
                        prefix_len = past_key_values._flash_prefix_lens[self.layer_idx]
                        cur_len = k_cur.shape[1]
                        past_key_values._flash_k_caches[self.layer_idx][:, prefix_len : prefix_len + cur_len].copy_(k_cur)
                        past_key_values._flash_v_caches[self.layer_idx][:, prefix_len : prefix_len + cur_len].copy_(v_cur)
                        k = past_key_values._flash_k_caches[self.layer_idx][:, : prefix_len + cur_len]
                        v = past_key_values._flash_v_caches[self.layer_idx][:, : prefix_len + cur_len]
                    else:
                        layer_cache = past_key_values.layers[self.layer_idx]
                        past_k = layer_cache.keys
                        past_v = layer_cache.values
                        if past_k.numel() > 0:
                            pk = past_k.transpose(1, 2).contiguous()
                            pv = past_v.transpose(1, 2).contiguous()
                            k = torch.cat([pk, k_cur], dim=1)
                            v = torch.cat([pv, v_cur], dim=1)
                        else:
                            k = k_cur
                            v = v_cur
            else:
                k = k_cur
                v = v_cur

            attn_output = _flash_or_sdpa(q, k, v, dropout_p=0.0, softmax_scale=self.scaling, causal=False)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            # Eager fallback with mask
            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                else:
                    layer_cache = past_key_values.layers[self.layer_idx]
                    past_k = layer_cache.keys
                    past_v = layer_cache.values
                    if past_k.numel() > 0:
                        key_states = torch.cat([past_k, key_states], dim=2)
                        value_states = torch.cat([past_v, value_states], dim=2)

            key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * self.scaling
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states_expanded.shape[-2]]
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_expanded)
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj_mot_gen(attn_output)
        return attn_output, None


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer with MoT (dual understanding + generation paths)."""

    def __init__(self, config: NEOLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        # Understanding path MLP + norms
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Generation path MLP + norms (MoT)
        self.mlp_mot_gen = Qwen3MLP(config)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states,
        image_gen_indicators=None,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        """Dispatch to understanding or generation forward based on indicators."""
        if image_gen_indicators is None or not image_gen_indicators.any():
            return self._forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        elif image_gen_indicators.all():
            return self._forward_gen(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        else:
            return self._forward_mixed(hidden_states, image_gen_indicators, indexes, attention_mask, past_key_values, **kwargs)

    def _forward_und(self, hidden_states, indexes, attention_mask, past_key_values, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn.forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward_gen(self, hidden_states, indexes, attention_mask, past_key_values, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states, _ = self.self_attn.forward_gen(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward_mixed(self, hidden_states, image_gen_indicators, indexes, attention_mask, past_key_values, **kwargs):
        """Mixed path: some tokens are understanding, some are generation."""
        residual = hidden_states
        _hidden = hidden_states.new_zeros(hidden_states.shape)
        _hidden[~image_gen_indicators] = self.input_layernorm(hidden_states[~image_gen_indicators])
        _hidden[image_gen_indicators] = self.input_layernorm_mot_gen(hidden_states[image_gen_indicators])
        hidden_states = _hidden

        # For mixed path, use understanding attention (shared KV)
        hidden_states, _ = self.self_attn.forward_und(hidden_states, indexes, attention_mask, past_key_values, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        _hidden = hidden_states.new_zeros(hidden_states.shape)
        _hidden[~image_gen_indicators] = self.mlp(self.post_attention_layernorm(hidden_states[~image_gen_indicators]))
        _hidden[image_gen_indicators] = self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(hidden_states[image_gen_indicators])
        )
        hidden_states = residual + _hidden
        return hidden_states


class Qwen3Model(nn.Module):
    """Qwen3 transformer model with MoT support."""

    def __init__(self, config: NEOLLMConfig):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        image_gen_indicators=None,
        indexes=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        exist_gen = image_gen_indicators is not None and image_gen_indicators.any()
        exist_und = image_gen_indicators is None or (~image_gen_indicators).any()

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        # Apply final norm based on token type
        if not exist_gen:
            hidden_states = self.norm(hidden_states)
        elif not exist_und:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            _hidden = hidden_states.new_zeros(hidden_states.shape)
            _hidden[~image_gen_indicators] = self.norm(hidden_states[~image_gen_indicators])
            _hidden[image_gen_indicators] = self.norm_mot_gen(hidden_states[image_gen_indicators])
            hidden_states = _hidden

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 causal LM with MoT backbone."""

    def __init__(self, config: NEOLLMConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        image_gen_indicators=None,
        indexes=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            image_gen_indicators=image_gen_indicators,
            indexes=indexes,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )


# ---------------------------------------------------------------------------
# Top-level Model: NEOChatModel
# ---------------------------------------------------------------------------


class NEOChatModel(PreTrainedModel):
    """NEOChatModel (SenseNova-U1): unified understanding + generation via Flow-Matching.

    Architecture:
      - vision_model: NEOVisionModel (understanding ViT)
      - language_model: Qwen3ForCausalLM with MoT (dual-path decoder layers)
      - fm_modules: ModuleDict containing:
          - vision_model_mot_gen: NEOVisionModel (generation ViT)
          - timestep_embedder: TimestepEmbedder
          - noise_scale_embedder: TimestepEmbedder (optional)
          - fm_head: nn.Sequential (Flow-Matching prediction head)
    """

    config_class = NEOChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["NEOVisionModel", "Qwen3DecoderLayer"]

    def __init__(self, config: NEOChatConfig):
        super().__init__(config)

        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        # Vision encoder (understanding)
        self.vision_model = NEOVisionModel(config.vision_config)

        # Language model (Qwen3 + MoT)
        self.language_model = Qwen3ForCausalLM(config.llm_config)

        # Flow-Matching modules
        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3 * (patch_size * merge_size) ** 2
        llm_hidden_size = config.llm_config.hidden_size

        # Generation ViT
        vision_model_mot_gen = NEOVisionModel(config.vision_config)

        # Timestep embedder
        timestep_embedder = TimestepEmbedder(llm_hidden_size)

        # FM head: simple 2-layer MLP (when fm_head_layers <= 2)
        self.use_deep_fm_head = config.fm_head_layers > 2
        if self.use_deep_fm_head:
            # Deep FM head path (not used in this checkpoint)
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, config.fm_head_dim, bias=True),
                nn.GELU(),
                nn.Linear(config.fm_head_dim, output_dim, bias=True),
            )
        else:
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, output_dim, bias=True),
            )

        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": vision_model_mot_gen,
                "timestep_embedder": timestep_embedder,
                "fm_head": fm_head,
            }
        )

        # Noise scale embedder (if enabled)
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules["noise_scale_embedder"] = noise_scale_embedder

        # Store FM config params
        self.use_pixel_head = config.use_pixel_head
        self.concat_time_token_num = config.concat_time_token_num
        self.noise_scale = config.noise_scale
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len
        self.noise_scale_max_value = config.noise_scale_max_value
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len
        self.t_eps = config.t_eps
        self.P_mean = config.P_mean
        self.P_std = config.P_std

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        """Extract visual features using understanding or generation ViT."""
        if gen_model:
            return self.fm_modules["vision_model_mot_gen"](pixel_values=pixel_values, grid_hw=grid_hw).last_hidden_state
        else:
            return self.vision_model(pixel_values=pixel_values, grid_hw=grid_hw).last_hidden_state

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_gen_indicators=None,
        indexes=None,
        gen_pixel_values=None,
        timesteps=None,
        gen_grid_hw=None,
        grid_hw=None,
        **kwargs,
    ):
        """Training forward pass for SenseNova-U1.

        Supports three modes depending on inputs:
        1. Understanding only: input_ids + labels (+ optional pixel_values for VQA)
        2. Generation only: input_ids + gen_pixel_values + timesteps
        3. Mixed: both understanding and generation tokens in same batch

        Returns ModelOutput with .loss for VeOmni trainer compatibility.
        """
        device = input_ids.device if input_ids is not None else next(self.parameters()).device
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 0

        # Build 3D position IDs (t, h, w) if not provided
        # indexes shape: [3, S] where row 0=temporal, 1=h, 2=w
        # For text-only: t = sequential positions, h = 0, w = 0
        if indexes is None:
            t_ids = torch.arange(seq_len, device=device)
            h_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
            w_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
            indexes = torch.stack([t_ids, h_ids, w_ids], dim=0)  # [3, S]

        # Embed input tokens
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Process understanding images (ViT features injected into sequence)
        if pixel_values is not None:
            vit_features = self.extract_feature(pixel_values, gen_model=False, grid_hw=grid_hw)

        # Process generation images (noisy target for flow-matching loss)
        # Official (SenseNova-U1 training/sensenovavl):
        #   1. Create noise in pixel space, interpolate: z = t*x + (1-t)*noise
        #   2. Merge patches: [h,w] -> [h/m, w/m] with m=merge_size
        #   3. Pass noisy pixels through gen ViT -> features for LLM
        #   4. fm_head predicts in merged pixel space
        #   5. v = (x - z)/(1-t), pred_v = (pred_x - z_merged)/(1-t)
        #   6. loss = MSE(pred_v, v)
        fm_loss = None
        image_gen_x = None
        image_gen_z = None
        image_gen_t = None
        if gen_pixel_values is not None and timesteps is not None and image_gen_indicators is not None:
            merge_size = int(1 / self.downsample_ratio)
            patch_size = self.patch_size
            ps_merged = patch_size * merge_size

            noise = torch.randn_like(gen_pixel_values)
            if self.noise_scale != 1.0:
                noise = noise * self.noise_scale

            t = timesteps.view(-1, 1) if timesteps.dim() == 1 else timesteps
            noisy_pixels = t * gen_pixel_values + (1 - t) * noise

            # Pass noisy pixels through gen ViT to get LLM-space features
            gen_features = self.extract_feature(noisy_pixels, gen_model=True, grid_hw=gen_grid_hw)

            gen_mask = image_gen_indicators.bool()
            if gen_mask.any():
                flat_gen = gen_features.reshape(-1, gen_features.shape[-1])
                num_gen_tokens = gen_mask.sum().item()
                if flat_gen.shape[0] >= num_gen_tokens:
                    inputs_embeds[gen_mask] = flat_gen[:num_gen_tokens]

            # Merge pixel patches for FM target (pixel space, post-merge)
            # gen_pixel_values: [N_patches, 3*p*p] -> merge into [N_merged, ps_merged^2*3]
            def _merge_patches(pv, grid_hw_t):
                merged = []
                cur = 0
                for i in range(grid_hw_t.shape[0]):
                    h, w = int(grid_hw_t[i, 0]), int(grid_hw_t[i, 1])
                    n = h * w
                    img = pv[cur:cur + n].view(h, w, 3, patch_size, patch_size)
                    img = img.view(h // merge_size, merge_size, w // merge_size, merge_size, 3, patch_size, patch_size)
                    img = torch.einsum("h a w b c i j -> h w a i b j c", img).contiguous()
                    img = img.view(-1, ps_merged ** 2 * 3)
                    merged.append(img)
                    cur += n
                return torch.cat(merged, dim=0)

            image_gen_x = _merge_patches(gen_pixel_values, gen_grid_hw)
            image_gen_z = _merge_patches(noisy_pixels, gen_grid_hw)
            # Per-merged-token timesteps
            num_merged = image_gen_x.shape[0]
            image_gen_t = timesteps.expand(num_merged)

        # Add timestep + noise_scale embedding to generation token positions
        if image_gen_t is not None and image_gen_indicators is not None:
            gen_mask = image_gen_indicators.bool()
            if gen_mask.any():
                t_emb = self.fm_modules["timestep_embedder"](image_gen_t)
                if self.add_noise_scale_embedding and "noise_scale_embedder" in self.fm_modules:
                    ns_val = torch.full_like(image_gen_t, self.noise_scale / self.noise_scale_max_value)
                    ns_emb = self.fm_modules["noise_scale_embedder"](ns_val)
                    t_emb = t_emb + ns_emb
                num_gen_tokens = gen_mask.sum().item()
                if t_emb.shape[0] >= num_gen_tokens:
                    inputs_embeds[gen_mask] = inputs_embeds[gen_mask] + t_emb[:num_gen_tokens]

        # Build 4D causal attention mask from 2D padding mask
        # attention code expects [B, 1, S, S] additive mask (0 = attend, -inf = mask)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=inputs_embeds.dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if attention_mask is not None and attention_mask.dim() == 2:
            # Combine with padding mask: masked positions (0) get -inf
            pad_mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
            pad_mask = torch.where(pad_mask == 0, torch.tensor(float("-inf"), device=device, dtype=inputs_embeds.dtype), torch.zeros_like(pad_mask))
            causal_mask = causal_mask + pad_mask

        # Forward through language model (returns CausalLMOutputWithPast with .hidden_states)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            image_gen_indicators=image_gen_indicators,
            indexes=indexes,
            attention_mask=causal_mask,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states

        # CE loss for understanding tokens
        ce_loss = None
        if labels is not None:
            logits = self.language_model.lm_head(hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.language_model.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Flow-matching loss for generation tokens
        # Official: pred_x = fm_head(hidden), pred_v = (pred_x - z)/(1-t),
        #           target_v = (x - z)/(1-t), loss = MSE(pred_v, target_v)
        if image_gen_x is not None and image_gen_z is not None and image_gen_t is not None:
            gen_mask = image_gen_indicators.bool()
            if gen_mask.any():
                gen_hidden = hidden_states[gen_mask]
                fm_pred_x = self.fm_modules["fm_head"](gen_hidden)
                num_gen = min(fm_pred_x.shape[0], image_gen_x.shape[0])
                t_col = image_gen_t[:num_gen].view(-1, 1)
                denom = (1 - t_col).clamp_min(self.t_eps)
                pred_v = (fm_pred_x[:num_gen] - image_gen_z[:num_gen]) / denom
                target_v = (image_gen_x[:num_gen] - image_gen_z[:num_gen]) / denom
                fm_loss = torch.nn.functional.mse_loss(pred_v, target_v)

        # Combine losses
        loss = None
        if ce_loss is not None or fm_loss is not None:
            loss = torch.tensor(0.0, device=device)
            if ce_loss is not None:
                loss = loss + ce_loss
            if fm_loss is not None:
                loss = loss + fm_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=self.language_model.lm_head(hidden_states) if labels is None else None,
            hidden_states=hidden_states,
        )

    # ------------------------------------------------------------------
    # Generation inference helpers
    # ------------------------------------------------------------------

    def patchify(self, images, patch_size, channel_first=False):
        """images: (N, 3, H, W) → (N, L, patch_size²*3)"""
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
        if channel_first:
            x = torch.einsum("nchpwq->nhwcpq", x)
        else:
            x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(images.shape[0], h * w, patch_size ** 2 * 3)

    def unpatchify(self, x, patch_size, h=None, w=None):
        """x: (N, L, patch_size²*3) → (N, 3, H, W)"""
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h = h // patch_size
            w = w // patch_size
        x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int, timestep_shift: float) -> torch.Tensor:
        sigma = 1 - t
        shift = timestep_shift
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return 1 - sigma

    def _build_t2i_query(self, prompt_text, system_message=None, append_text=None):
        sys_msg = system_message if system_message is not None else ""
        parts = []
        if sys_msg:
            parts.append(f"<|im_start|>system\n{sys_msg}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{prompt_text}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        result = "".join(parts)
        if append_text:
            result += append_text
        return result

    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        n = token_h * token_w
        t_image = torch.full((n,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(n, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def get_thw_indexes(self, input_ids, grid_hw=None):
        img_start_shift = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=input_ids.device), (input_ids == self.img_start_token_id).long()],
            dim=0,
        )[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
        h_indexes = torch.zeros_like(t_indexes)
        w_indexes = torch.zeros_like(t_indexes)

        if grid_hw is not None:
            selected = input_ids == self.img_context_token_id
            if selected.long().sum() > 0:
                abs_pos_w, abs_pos_h = _build_abs_positions_from_grid_hw(
                    grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device
                )
                h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
                w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    def _build_it2i_inputs(self, tokenizer, query, pixel_values=None, grid_hw=None):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        indexes = self.get_thw_indexes(input_ids[0], grid_hw)
        attention_mask = create_block_causal_mask(indexes[0])

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = input_ids_flat == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)

        return input_embeds, indexes, attention_mask

    def _it2i_prefix_forward(self, input_embeds, indexes, attention_mask):
        past_key_values = DynamicCache()
        out = self.language_model.model(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return out.past_key_values, out.last_hidden_state

    def _t2i_predict_v(self, input_embeds, indexes_image, past_key_values, t, z, image_token_num, image_size=None):
        B, L = z.shape[0], z.shape[1]
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones(
                (input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.bool, device=input_embeds.device
            ),
            indexes=indexes_image,
            attention_mask=None,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )

        if self.use_deep_fm_head:
            x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B * L, -1), t.repeat(B * L)
            ).view(B, L, -1)
        else:
            x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)
            ).view(B, L, -1)

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def it2i_generate(
        self,
        tokenizer,
        prompt,
        images,
        cfg_scale=1,
        img_cfg_scale=1,
        cfg_norm="none",
        enable_timestep_shift=True,
        timestep_shift=1,
        image_size=(256, 256),
        num_steps=30,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        method="euler",
        cfg_interval=(0, 1),
        batch_size=1,
        t_eps=0.02,
        think_mode=False,
        seed=0,
    ):
        assert cfg_norm in ("none", "global", "channel")

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.t_eps = t_eps

        image_token_count = prompt.count("<image>")
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            if image_token_count == 0 and len(images) > 1:
                prompt = "".join(f"Image-{i + 1}:<image>\n" for i in range(len(images))) + prompt
            else:
                prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values = []
        grid_hw = []
        for image in images:
            cur_pv, cur_ghw = load_image_native(
                image,
                self.patch_size,
                self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=min(2048 * 2048, (4096 * 4096) // len(images)),
                upscale=False,
            )
            pixel_values.append(cur_pv.to(self.device).to(torch.bfloat16))
            grid_hw.append(cur_ghw.to(self.device))
        pixel_values = torch.cat(pixel_values)
        grid_hw = torch.cat(grid_hw)

        merge_size = int(1 / self.downsample_ratio)
        needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
        needs_img_condition = needs_cfg and (img_cfg_scale == 1 or cfg_scale != img_cfg_scale)
        needs_uncondition = needs_cfg and img_cfg_scale != 1

        think_content = "<think>\n" if think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
        query_condition = self._build_t2i_query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_img_condition = (
            self._build_t2i_query("<image>" * len(images), append_text=IMG_START_TOKEN)
            if needs_img_condition
            else None
        )
        query_uncondition = self._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_uncondition else None

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio ** 2)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            query_condition = query_condition.replace("<image>", image_tokens, 1)
            if query_img_condition is not None:
                query_img_condition = query_img_condition.replace("<image>", image_tokens, 1)

        input_embeds_condition, indexes_condition, attn_mask_condition = self._build_it2i_inputs(
            tokenizer, query_condition, pixel_values, grid_hw
        )
        if query_img_condition is not None:
            input_embeds_img_cond, indexes_img_cond, attn_mask_img_cond = self._build_it2i_inputs(
                tokenizer, query_img_condition, pixel_values, grid_hw
            )
        else:
            input_embeds_img_cond = indexes_img_cond = attn_mask_img_cond = None
        if query_uncondition is not None:
            input_embeds_uncond, indexes_uncond, attn_mask_uncond = self._build_it2i_inputs(
                tokenizer, query_uncondition
            )
        else:
            input_embeds_uncond = indexes_uncond = attn_mask_uncond = None

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_cond = self._build_t2i_image_indexes(
            token_h, token_w, indexes_condition[0].max() + 1, device=self.device
        )
        indexes_image_img_cond = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_img_cond[0].max() + 1, device=self.device
            )
            if indexes_img_cond is not None
            else None
        )
        indexes_image_uncond = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_uncond[0].max() + 1, device=self.device
            )
            if indexes_uncond is not None
            else None
        )

        # Prefill KV caches
        past_kv_cond, hidden_cond = self._it2i_prefix_forward(
            input_embeds_condition, indexes_condition, attn_mask_condition
        )
        past_kv_img_cond = None
        if input_embeds_img_cond is not None:
            past_kv_img_cond, _ = self._it2i_prefix_forward(
                input_embeds_img_cond, indexes_img_cond, attn_mask_img_cond
            )
        past_kv_uncond = None
        if input_embeds_uncond is not None:
            past_kv_uncond, _ = self._it2i_prefix_forward(
                input_embeds_uncond, indexes_uncond, attn_mask_uncond
            )

        device = hidden_cond.device
        dtype = hidden_cond.dtype

        del pixel_values, grid_hw
        del input_embeds_condition, indexes_condition, attn_mask_condition
        if input_embeds_img_cond is not None:
            del input_embeds_img_cond, indexes_img_cond, attn_mask_img_cond
        if input_embeds_uncond is not None:
            del input_embeds_uncond, indexes_uncond, attn_mask_uncond
        del hidden_cond

        # Expand prefix caches to batch_size
        for li in range(len(past_kv_cond.layers)):
            lc = past_kv_cond.layers[li]
            lc.keys = lc.keys.expand(batch_size, *lc.keys.shape[1:])
            lc.values = lc.values.expand(batch_size, *lc.values.shape[1:])
            if past_kv_img_cond is not None:
                lc2 = past_kv_img_cond.layers[li]
                lc2.keys = lc2.keys.expand(batch_size, *lc2.keys.shape[1:])
                lc2.values = lc2.values.expand(batch_size, *lc2.values.shape[1:])
            if past_kv_uncond is not None:
                lc3 = past_kv_uncond.layers[li]
                lc3.keys = lc3.keys.expand(batch_size, *lc3.keys.shape[1:])
                lc3.values = lc3.values.expand(batch_size, *lc3.values.shape[1:])

        # Prepare flash KV caches
        img_tokens = token_h * token_w
        prepare_flash_kv_cache(past_kv_cond, current_len=img_tokens, batch_size=batch_size)
        if past_kv_img_cond is not None:
            prepare_flash_kv_cache(past_kv_img_cond, current_len=img_tokens, batch_size=batch_size)
        if past_kv_uncond is not None:
            prepare_flash_kv_cache(past_kv_uncond, current_len=img_tokens, batch_size=batch_size)

        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        gen_grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size ** 2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator
        )

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, img_tokens, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self.extract_feature(
                image_input.view(batch_size * grid_h * grid_w, -1), gen_model=True, grid_hw=gen_grid_hw
            ).view(batch_size, img_tokens, -1)
            t_expanded = t.expand(batch_size * img_tokens)
            timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(batch_size, img_tokens, -1)
            if self.add_noise_scale_embedding:
                ns_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
                noise_emb = self.fm_modules["noise_scale_embedder"](ns_tensor).view(batch_size, img_tokens, -1)
                timestep_embeddings = timestep_embeddings + noise_emb
            image_embeds = image_embeds + timestep_embeddings

            out_cond = self._t2i_predict_v(
                image_embeds, indexes_image_cond, past_kv_cond, t, z, image_token_num=img_tokens, image_size=image_size
            )

            if not use_cfg:
                v_pred = out_cond
            elif cfg_scale == 1 and img_cfg_scale == 1:
                v_pred = out_cond
            elif img_cfg_scale == 1:
                out_img_cond = self._t2i_predict_v(
                    image_embeds, indexes_image_img_cond, past_kv_img_cond, t, z,
                    image_token_num=img_tokens, image_size=image_size,
                )
                v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
            elif cfg_scale == img_cfg_scale:
                out_uncond = self._t2i_predict_v(
                    image_embeds, indexes_image_uncond, past_kv_uncond, t, z,
                    image_token_num=img_tokens, image_size=image_size,
                )
                v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                out_img_cond = self._t2i_predict_v(
                    image_embeds, indexes_image_img_cond, past_kv_img_cond, t, z,
                    image_token_num=img_tokens, image_size=image_size,
                )
                out_uncond = self._t2i_predict_v(
                    image_embeds, indexes_image_uncond, past_kv_uncond, t, z,
                    image_token_num=img_tokens, image_size=image_size,
                )
                v_pred = (
                    out_uncond
                    + cfg_scale * (out_cond - out_img_cond)
                    + img_cfg_scale * (out_img_cond - out_uncond)
                )
            if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
                if cfg_norm == "global":
                    norm_v_cond = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale
                elif cfg_norm == "channel":
                    norm_v_cond = torch.norm(out_cond, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale

            z = z + (t_next - t) * v_pred
            image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_kv_cond)
        if past_kv_img_cond is not None:
            clear_flash_kv_cache(past_kv_img_cond)
        if past_kv_uncond is not None:
            clear_flash_kv_cache(past_kv_uncond)

        return image_prediction
