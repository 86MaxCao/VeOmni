from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("thinkmorph")
def register_thinkmorph_model_config():
    from .configuration_thinkmorph import ThinkMorphConfig

    return ThinkMorphConfig


@MODELING_REGISTRY.register("thinkmorph")
def register_thinkmorph_modeling(architecture: str):
    from ..bagel.modeling_bagel import BagelForConditionalGeneration

    return BagelForConditionalGeneration
