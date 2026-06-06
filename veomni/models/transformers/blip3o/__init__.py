from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("blip3o_qwen")
def register_blip3o_model_config():
    from .configuration_blip3o import BLIP3oConfig

    return BLIP3oConfig


@MODELING_REGISTRY.register("blip3o_qwen")
def register_blip3o_modeling(architecture: str):
    from .modeling_blip3o import BLIP3oQwenForCausalLM

    return BLIP3oQwenForCausalLM
