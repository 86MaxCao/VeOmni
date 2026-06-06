"""Configuration for BLIP3o model."""

from transformers import PretrainedConfig


class BLIP3oVisionConfig(PretrainedConfig):
    """Configuration for the Qwen2.5-VL style vision encoder used in BLIP3o."""

    model_type = "blip3o_vision"

    def __init__(
        self,
        depth: int = 32,
        hidden_size: int = 1280,
        num_heads: int = 16,
        intermediate_size: int = 3420,
        patch_size: int = 14,
        spatial_patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        in_chans: int = 3,
        out_hidden_size: int = 3584,
        hidden_act: str = "silu",
        window_size: int = 112,
        fullatt_block_indexes: list = None,
        tokens_per_second: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.in_channels = in_channels
        self.in_chans = in_chans
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.tokens_per_second = tokens_per_second


class BLIP3oConfig(PretrainedConfig):
    """Configuration for BLIP3o multimodal model (understanding + generation)."""

    model_type = "blip3o_qwen"

    def __init__(
        self,
        # LLM config
        vocab_size: int = 151668,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
        attention_dropout: float = 0.0,
        sliding_window: int = 32768,
        max_window_layers: int = 28,
        use_sliding_window: bool = False,
        # Vision config
        vision_config: dict = None,
        # Generation config
        gen_hidden_size: int = 1792,
        gen_pooling: str = "early_pool2d_4",
        gen_vision_tower: str = "eva-clip-E-14-plus",
        # Projector config
        mm_projector_type: str = "mlp2x_gelu",
        use_mm_proj: bool = True,
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = "patch",
        mm_patch_merge_type: str = "flat",
        mm_use_im_start_end: bool = False,
        mm_use_im_patch_token: bool = False,
        # Token IDs
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        vision_token_id: int = 151654,
        # Generation-specific
        n_query: int = 64,
        # DIT config (derived from checkpoint)
        dit_hidden_size: int = 1792,
        dit_num_layers: int = 24,
        dit_num_heads: int = 28,
        dit_ffn_hidden_size: int = 4864,
        dit_time_embed_dim: int = 1024,
        dit_timestep_input_dim: int = 256,
        # Gen vision tower config (EVA-CLIP)
        gen_vit_hidden_size: int = 1792,
        gen_vit_num_blocks: int = 64,
        gen_vit_num_heads: int = 16,
        gen_vit_mlp_ratio: float = 8.571428571428571,  # 15360/1792
        gen_vit_patch_size: int = 14,
        gen_vit_image_size: int = 448,  # 32*14=448 -> 1024 patches + 1 cls
        # VAE config
        vae_latent_channels: int = 16,
        # Other
        tokenizer_model_max_length: int = 512,
        tokenizer_padding_side: str = "right",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # LLM
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
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window

        # Vision
        if vision_config is None:
            self.vision_config = BLIP3oVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = BLIP3oVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Generation
        self.gen_hidden_size = gen_hidden_size
        self.gen_pooling = gen_pooling
        self.gen_vision_tower = gen_vision_tower

        # Projector
        self.mm_projector_type = mm_projector_type
        self.use_mm_proj = use_mm_proj
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token

        # Token IDs
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_token_id = vision_token_id

        # Queries
        self.n_query = n_query

        # DIT
        self.dit_hidden_size = dit_hidden_size
        self.dit_num_layers = dit_num_layers
        self.dit_num_heads = dit_num_heads
        self.dit_ffn_hidden_size = dit_ffn_hidden_size
        self.dit_time_embed_dim = dit_time_embed_dim
        self.dit_timestep_input_dim = dit_timestep_input_dim

        # Gen vision tower
        self.gen_vit_hidden_size = gen_vit_hidden_size
        self.gen_vit_num_blocks = gen_vit_num_blocks
        self.gen_vit_num_heads = gen_vit_num_heads
        self.gen_vit_mlp_ratio = gen_vit_mlp_ratio
        self.gen_vit_patch_size = gen_vit_patch_size
        self.gen_vit_image_size = gen_vit_image_size

        # VAE
        self.vae_latent_channels = vae_latent_channels

        # Other
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
