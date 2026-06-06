"""ThinkMorph configuration.

ThinkMorph uses the exact same architecture as Bagel (BAGEL-7B-MoT), only
differing in trained weights and inference behavior (Chain-of-Thought).
Its checkpoint config.json is malformed, so we construct BagelConfig from
the separate llm_config.json and vit_config.json files.
"""

import json
import os

from ..bagel.configuration_bagel import BagelConfig, BagelLLMConfig, BagelVitConfig


class ThinkMorphConfig(BagelConfig):
    model_type = "thinkmorph"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dir = pretrained_model_name_or_path

        llm_config_path = os.path.join(config_dir, "llm_config.json")
        vit_config_path = os.path.join(config_dir, "vit_config.json")

        with open(llm_config_path) as f:
            llm_dict = json.load(f)
        with open(vit_config_path) as f:
            vit_dict = json.load(f)

        llm_dict["qk_norm"] = True
        llm_dict["tie_word_embeddings"] = False
        llm_dict["layer_module"] = "Qwen2MoTDecoderLayer"

        config = cls.__new__(cls)
        BagelConfig.__init__(
            config,
            visual_gen=True,
            visual_und=True,
            llm_config=llm_dict,
            vit_config=vit_dict,
            latent_patch_size=2,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            timestep_shift=1.0,
        )
        return config
