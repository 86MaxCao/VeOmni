"""Configuration for the LatentUM model (understanding + generation via MoT discrete tokens)."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3Config


class InternVisionConfig(PretrainedConfig):
    """Configuration for the InternVision encoder used in LatentUM."""

    model_type = "intern_vit_6b"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 14,
        image_size: int = 448,
        qkv_bias: bool = False,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        qk_normalization: bool = False,
        num_hidden_layers: int = 24,
        use_flash_attn: bool = True,
        use_fa3: bool = False,
        hidden_act: str = "gelu",
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.qkv_bias = qkv_bias
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qk_normalization = qk_normalization
        self.num_hidden_layers = num_hidden_layers
        self.use_flash_attn = use_flash_attn
        self.use_fa3 = use_fa3
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class InternVLChatConfig(PretrainedConfig):
    """Configuration for the InternVL Chat model (vision + LLM) used within LatentUM."""

    model_type = "internvl_chat"
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        select_layer: int = -1,
        force_image_size: Optional[int] = None,
        downsample_ratio: float = 0.5,
        template: Optional[str] = None,
        dynamic_image_size: bool = False,
        use_thumbnail: bool = False,
        ps_version: str = "v1",
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 6,
        pad2square: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}

        if isinstance(vision_config, dict):
            self.vision_config = InternVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if isinstance(llm_config, dict):
            self.llm_config = Qwen3Config(**llm_config)
        else:
            self.llm_config = llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.pad2square = pad2square
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LatentUMConfig(PretrainedConfig):
    """
    Configuration for LatentUM: a multimodal model combining InternVL (vision + LLM)
    with MoT (Mixture of Transformers) and discrete image token generation via VQ.

    The checkpoint structure is:
        internvl.* -> InternVLChatModel (with MoT modifications + ar_head + visual_projector)
        quantizer.* -> VQ_MLP_MCQ (multi-codebook vector quantizer)
    """

    model_type = "latentum"
    # Needed so the loader can recognize the architecture
    architectures = ["LatentUMModel"]

    def __init__(
        self,
        # Top-level LatentUM params
        base_model_name_or_path: Optional[str] = None,
        quantizer_ckpt_path: Optional[str] = None,
        llm_hidden_size: int = 2560,
        mixture_mode: str = "mot",
        embedding_dim: int = 256,
        image_size: int = 448,
        num_image_tokens: int = 256,
        max_num_patches: int = 12,
        image_token: str = "<image>",
        # Nested configs (stored as dicts in JSON, parsed here)
        internvl_config: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        quantizer: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        # Legacy
        legacy_checkpoint_path: Optional[str] = None,
        legacy_state_key: str = "module",
        **kwargs,
    ):
        # Remove kwargs that conflict with PretrainedConfig
        kwargs.pop("architectures", None)

        super().__init__(**kwargs)

        self.base_model_name_or_path = base_model_name_or_path
        self.quantizer_ckpt_path = quantizer_ckpt_path
        self.llm_hidden_size = llm_hidden_size
        self.mixture_mode = mixture_mode
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.max_num_patches = max_num_patches
        self.image_token = image_token

        # Parse nested InternVL config
        if internvl_config is not None:
            self.internvl_config = InternVLChatConfig(**internvl_config)
        else:
            self.internvl_config = InternVLChatConfig()

        # Model sub-config (contains mixture_mode, embedding_dim, quantizer, head info)
        self.model_config = model if model is not None else {}

        # Quantizer config
        self.quantizer_config = quantizer if quantizer is not None else {
            "vq_type": "multi_vq",
            "type": "MLP",
            "input_feature_dim": 4096,
            "embedding_dim": 256,
            "llm_hidden_size": 2560,
            "num_embeddings": 2048,
            "num_codebooks": 8,
        }

        # AR head config
        self.head_config = head if head is not None else {
            "num_codebooks": 8,
            "num_layers": 3,
            "hidden_size": 2560,
            "num_embeddings": 2048,
            "num_heads": 32,
            "mlp_ratio": 4.0,
        }

        self.legacy_checkpoint_path = legacy_checkpoint_path
        self.legacy_state_key = legacy_state_key

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if isinstance(self.internvl_config, InternVLChatConfig):
            output["internvl_config"] = self.internvl_config.to_dict()
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # The LatentUM config.json does not have a top-level model_type field.
        # We inject it here so PretrainedConfig machinery works.
        if "model_type" not in config_dict:
            config_dict["model_type"] = "latentum"
        if "architectures" not in config_dict:
            config_dict["architectures"] = ["LatentUMModel"]
        return cls.from_dict(config_dict, **kwargs)
