"""Unified Multimodal Trainer for VeOmni.

Supports training all 5 unified multimodal models:
- bagel / thinkmorph: Packed NaViT with CE + MSE loss
- blip3o: Standard causal LM with diffusion generation
- neo_chat (U1): Qwen3 MoT with Flow-Matching
- latentum: InternVL MoT with discrete tokens

All models expose a forward(**batch) -> output.loss interface that VeOmni's
BaseTrainer.forward_backward_step() calls directly.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..arguments import TrainingArguments
from .vlm_trainer import VeOmniVLMArguments, VLMTrainer, VLMMDataArguments, VLMMModelArguments, VLMTrainingArguments


@dataclass
class UnifiedTrainingArguments(VLMTrainingArguments):
    """Extended training arguments for unified multimodal models."""

    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for MSE/flow-matching loss (Bagel/ThinkMorph/U1)."},
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for cross-entropy loss."},
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Whether to freeze VAE/quantizer parameters."},
    )
    freeze_gen_modules: bool = field(
        default=False,
        metadata={"help": "Whether to freeze generation-specific modules (DIT, FM head, etc.)."},
    )


@dataclass
class UnifiedVeOmniArguments(VeOmniVLMArguments):
    train: "UnifiedTrainingArguments" = field(default_factory=UnifiedTrainingArguments)


class UnifiedMultimodalTrainer(VLMTrainer):
    """Unified trainer that handles all 5 multimodal model types.

    Extends VLMTrainer with model-specific freeze logic and loss handling.
    The forward pass is delegated entirely to the model's own forward() method,
    which already computes the appropriate loss.
    """

    def __init__(self, args: UnifiedVeOmniArguments):
        super().__init__(args)

    def _freeze_model_module(self):
        """Extended freeze logic for unified multimodal models."""
        args: UnifiedVeOmniArguments = self.base.args
        model = self.base.model
        model_type = self.base.model_config.model_type

        if model_type in ("bagel", "thinkmorph"):
            self._freeze_bagel(model, args)
        elif model_type == "blip3o_qwen":
            self._freeze_blip3o(model, args)
        elif model_type == "neo_chat":
            self._freeze_neo_chat(model, args)
        elif model_type == "latentum":
            self._freeze_latentum(model, args)
        else:
            super()._freeze_model_module()
            return

        from ..utils.model_utils import pretty_print_trainable_parameters
        from ..utils import helper
        pretty_print_trainable_parameters(model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _freeze_bagel(self, model, args):
        """Freeze logic for Bagel/ThinkMorph: optionally freeze ViT and VAE."""
        if args.train.freeze_vit and hasattr(model, "vit_model"):
            model.vit_model.requires_grad_(False)
            if hasattr(model, "connector"):
                model.connector.requires_grad_(True)

        if args.train.freeze_vae:
            if hasattr(model, "vae_encoder"):
                model.vae_encoder.requires_grad_(False)
            if hasattr(model, "vae_decoder"):
                model.vae_decoder.requires_grad_(False)

    def _freeze_blip3o(self, model, args):
        """Freeze logic for BLIP3o: optionally freeze ViT, VAE, DIT."""
        if args.train.freeze_vit and hasattr(model, "visual"):
            model.visual.requires_grad_(False)

        if args.train.freeze_vae and hasattr(model.model, "vae"):
            model.model.vae.requires_grad_(False)

        if args.train.freeze_gen_modules:
            if hasattr(model.model, "dit"):
                model.model.dit.requires_grad_(False)
            if hasattr(model.model, "gen_vision_tower"):
                model.model.gen_vision_tower.requires_grad_(False)

    def _freeze_neo_chat(self, model, args):
        """Freeze logic for SenseNova-U1: optionally freeze ViT and FM modules."""
        if args.train.freeze_vit and hasattr(model, "vision_model"):
            model.vision_model.requires_grad_(False)

        if args.train.freeze_gen_modules and hasattr(model, "fm_modules"):
            model.fm_modules.requires_grad_(False)

    def _freeze_latentum(self, model, args):
        """Freeze logic for LatentUM: optionally freeze ViT and quantizer."""
        if args.train.freeze_vit and hasattr(model, "internvl"):
            if hasattr(model.internvl, "vision_model"):
                model.internvl.vision_model.requires_grad_(False)

        if args.train.freeze_vae and hasattr(model, "quantizer"):
            model.quantizer.requires_grad_(False)

        if args.train.freeze_gen_modules:
            if hasattr(model.internvl, "ar_head"):
                model.internvl.ar_head.requires_grad_(False)
            if hasattr(model.internvl, "visual_projector"):
                model.internvl.visual_projector.requires_grad_(False)
