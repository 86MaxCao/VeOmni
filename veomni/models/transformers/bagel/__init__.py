from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("bagel")
def register_bagel_model_config():
    from .configuration_bagel import BagelConfig

    return BagelConfig


@MODELING_REGISTRY.register("bagel")
def register_bagel_modeling(architecture: str):
    from .modeling_bagel import BagelForConditionalGeneration

    return BagelForConditionalGeneration
