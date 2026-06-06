"""Configuration classes for the NEO-Chat (SenseNova-U1) model."""

import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NEOVisionConfig(PretrainedConfig):
    """Configuration for the NEO Vision encoder (lightweight ViT with 2D-RoPE)."""

    model_type = "neo_vision"

    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        hidden_size=1024,
        llm_hidden_size=2048,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
        min_pixels=65536,
        max_pixels=4194304,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        # NOTE: trailing comma intentionally creates a tuple - the ViT embedding
        # code indexes these as config.llm_hidden_size[0] / config.downsample_ratio[0]
        self.llm_hidden_size = (llm_hidden_size,)
        self.downsample_ratio = (downsample_ratio,)
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)


class NEOLLMConfig(PretrainedConfig):
    """Configuration for the dense Qwen3 backbone with MoT (Mixture of Transformers).

    Extends a standard Qwen3 config with spatial RoPE parameters.
    """

    model_type = "qwen3"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=42,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=5000000.0,
        rope_scaling=None,
        sliding_window=None,
        use_sliding_window=False,
        max_window_layers=42,
        tie_word_embeddings=False,
        use_cache=False,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=10000,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151645,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers
        self.use_cache = use_cache
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw

        # Build layer_types for attention type dispatch
        use_swa = bool(use_sliding_window) and sliding_window is not None
        self.layer_types = [
            "sliding_attention" if (use_swa and i >= max_window_layers) else "full_attention"
            for i in range(num_hidden_layers)
        ]


class NEOChatConfig(PretrainedConfig):
    """Top-level configuration for NEOChatModel (SenseNova-U1).

    Composes:
      - vision_config: NEOVisionConfig (understanding ViT)
      - llm_config: NEOLLMConfig (Qwen3 + MoT backbone)
      - Flow-Matching generation parameters
    """

    model_type = "neo_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        downsample_ratio=0.5,
        template=None,
        # Flow-Matching parameters
        timestep_shift=1.0,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        add_noise_scale_embedding=True,
        noise_scale_max_value=8.0,
        noise_scale=1.0,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        fm_head_dim=1536,
        fm_head_layers=2,
        fm_head_mlp_ratio=1,
        extra_num_layers_post=0,
        concat_time_token_num=0,
        use_pixel_head=False,
        use_adaLN=False,
        patch_size=16,
        min_pixels=65536,
        max_pixels=16777216,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["NEOVisionModel"]}
            logger.info("vision_config is None. Initializing the NEOVisionConfig with default values.")

        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info("llm_config is None. Initializing the LLM config with default values.")

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if isinstance(llm_config, dict):
            self.llm_config = NEOLLMConfig(**llm_config)
        else:
            self.llm_config = llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        # Flow-Matching parameters
        self.timestep_shift = timestep_shift
        self.time_schedule = time_schedule
        self.time_shift_type = time_shift_type
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.noise_scale_max_value = noise_scale_max_value
        self.noise_scale = noise_scale
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.extra_num_layers_post = extra_num_layers_post
        self.concat_time_token_num = concat_time_token_num
        self.use_pixel_head = use_pixel_head
        self.use_adaLN = use_adaLN
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        return output
