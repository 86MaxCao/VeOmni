from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("neo_chat")
def register_neo_chat_config():
    from .configuration_neo_chat import NEOChatConfig

    return NEOChatConfig


@MODELING_REGISTRY.register("neo_chat")
def register_neo_chat_modeling(architecture: str):
    from .modeling_neo_chat import NEOChatModel

    return NEOChatModel
