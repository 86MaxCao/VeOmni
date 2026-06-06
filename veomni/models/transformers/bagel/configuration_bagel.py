from dataclasses import dataclass
from typing import Optional

from transformers.configuration_utils import PretrainedConfig


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159


class BagelLLMConfig(PretrainedConfig):
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und
        self.head_dim = hidden_size // num_attention_heads
        self.pad_token_id = kwargs.get("pad_token_id", None)


class BagelVitConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=26,
        num_attention_heads=16,
        num_channels=3,
        image_size=980,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope = rope


class BagelVaeConfig(PretrainedConfig):

    def __init__(
        self,
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=None,
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.in_channels = in_channels
        self.downsample = downsample
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult or [1, 2, 4, 4]
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor


class BagelConfig(PretrainedConfig):
    model_type = "bagel"

    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.latent_patch_size = latent_patch_size
        # BAGEL-7B-MoT checkpoint was trained with max_latent_size=64 but
        # config.json incorrectly says 32; always use 64 to match weights.
        self.max_latent_size = max(max_latent_size, 64)
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

        if isinstance(llm_config, dict):
            self.llm_config = BagelLLMConfig(**llm_config)
        elif llm_config is None:
            self.llm_config = BagelLLMConfig()
        else:
            self.llm_config = llm_config

        if isinstance(vit_config, dict):
            # Bagel uses select_layer=-2 so the last ViT layer is unused/unsaved
            if vit_config.get("num_hidden_layers", 27) == 27:
                vit_config = dict(vit_config)
                vit_config["num_hidden_layers"] = 26
            self.vit_config = BagelVitConfig(**vit_config)
        elif vit_config is None:
            self.vit_config = BagelVitConfig()
        else:
            self.vit_config = vit_config

        if isinstance(vae_config, dict):
            self.vae_config = BagelVaeConfig(**vae_config)
        elif vae_config is None:
            self.vae_config = BagelVaeConfig()
        else:
            self.vae_config = vae_config
