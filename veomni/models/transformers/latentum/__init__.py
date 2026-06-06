from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("latentum")
def register_latentum_model_config():
    from .configuration_latentum import LatentUMConfig

    return LatentUMConfig


@MODELING_REGISTRY.register("latentum")
def register_latentum_modeling(architecture: str):
    from .modeling_latentum import LatentUMModel

    return LatentUMModel
