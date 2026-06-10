"""Unified Multimodal Trainer for VeOmni.

Supports training all 5 unified multimodal models:
- bagel / thinkmorph: Packed NaViT with CE + MSE loss
- blip3o: Standard causal LM with diffusion generation
- neo_chat (U1): Qwen3 MoT with Flow-Matching
- latentum: InternVL MoT with discrete tokens

All models expose a forward(**batch) -> output.loss interface that VeOmni's
BaseTrainer.forward_backward_step() calls directly.

Freeze strategies are aligned with each model's original training code.
"""

import logging
import os
import random
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF, InterpolationMode

from ..arguments import TrainingArguments
from ..utils.constants import IGNORE_INDEX
from .vlm_trainer import VeOmniVLMArguments, VLMTrainer, VLMMDataArguments, VLMMModelArguments, VLMTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingArguments(VLMTrainingArguments):
    """Extended training arguments for unified multimodal models.

    Freeze flags are designed to cover all 5 models' original training configs.
    """

    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for MSE/flow-matching loss (Bagel/ThinkMorph/U1)."},
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for cross-entropy loss."},
    )
    # --- Common freeze flags ---
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Freeze VAE/quantizer parameters. (Bagel/BLIP3o/LatentUM)"},
    )
    freeze_gen_modules: bool = field(
        default=False,
        metadata={"help": "Freeze generation-specific modules (DiT, FM head, ar_head, etc.)."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze LLM backbone. (Bagel/ThinkMorph/U1)"},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={"help": "Freeze MLP projector. (U1: mlp1, Bagel: connector)"},
    )
    # --- Bagel/ThinkMorph specific ---
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze understanding path and detach in forward. (Bagel/ThinkMorph) "
                  "Sets config.freeze_und=True so the model detaches understanding hidden states."},
    )
    # --- U1 specific ---
    unfreeze_mot_gen: bool = field(
        default=False,
        metadata={"help": "Unfreeze MoT generation branch (*_mot_gen parameters). (U1)"},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={"help": "Unfreeze last N ViT encoder layers. 0 means no unfreeze. (U1)"},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Unfreeze lm_head even when LLM is frozen. (U1)"},
    )
    # --- LatentUM specific ---
    training_stage: str = field(
        default="full",
        metadata={"help": "Training stage for LatentUM: 'full', 'lang_only', 'vision_only'. "
                  "'lang_only' trains LLM text path + lm_head; "
                  "'vision_only' trains MoT vision path + ar_head."},
    )
    # --- BLIP3o specific ---
    mm_tunable_parts: str = field(
        default="",
        metadata={"help": "Comma-separated tunable parts for BLIP3o: "
                  "'mm_vision_tower,mm_language_model,mm_embedding'. "
                  "When set, model is first fully frozen then listed parts are unfrozen. "
                  "DiT (sana) is always frozen; caption projection is always trainable."},
    )


@dataclass
class UnifiedVeOmniArguments(VeOmniVLMArguments):
    train: "UnifiedTrainingArguments" = field(default_factory=UnifiedTrainingArguments)


class UnifiedMultimodalTrainer(VLMTrainer):
    """Unified trainer that handles all 5 multimodal model types.

    Extends VLMTrainer with model-specific freeze logic aligned with each
    model's original training code.
    """

    def __init__(self, args: UnifiedVeOmniArguments):
        super().__init__(args)
        self._setup_vae_model(args)
        self._patch_preforward()
        self._patch_postforward()

    def _setup_vae_model(self, args: UnifiedVeOmniArguments):
        """Load VAE model for Bagel/ThinkMorph when mse_weight > 0."""
        model_type = self.base.model_config.model_type
        if model_type not in ("bagel", "thinkmorph"):
            return
        if args.train.mse_weight <= 0:
            return

        model_path = args.model.model_path
        ae_path = os.path.join(model_path, "ae.safetensors")
        if not os.path.exists(ae_path):
            logger.warning(f"VAE weights not found at {ae_path}, MSE loss will not be available")
            return

        from safetensors.torch import load_file
        from ..models.transformers.bagel.modeling_bagel import AutoEncoder
        from ..models.transformers.bagel.configuration_bagel import BagelVaeConfig

        vae_config = self.base.model_config.vae_config
        if isinstance(vae_config, dict):
            vae_config = BagelVaeConfig(**vae_config)

        vae_model = AutoEncoder(vae_config)
        state_dict = load_file(ae_path)
        vae_model.load_state_dict(state_dict, strict=False)
        vae_model.eval()
        vae_model.requires_grad_(False)
        self._vae_model = vae_model
        logger.info(f"Loaded VAE model from {ae_path}")

    def _patch_postforward(self):
        """Patch base.postforward to normalize losses and apply weights (matching official Bagel)."""
        model_type = self.base.model_config.model_type
        if model_type not in ("bagel", "thinkmorph", "neo_chat"):
            return

        args = self.base.args
        ce_weight = args.train.ce_weight
        mse_weight = args.train.mse_weight

        def _postforward_unified(outputs, micro_batch):
            import torch.distributed as dist
            loss_dict = {}
            device = torch.device("cpu")
            for v in micro_batch.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            loss = torch.tensor(0.0, device=device)

            if hasattr(outputs, "ce_loss") and outputs.ce_loss is not None:
                ce_raw = outputs.ce_loss
                total_ce_tokens = torch.tensor(ce_raw.numel(), device=device)
                if dist.is_initialized():
                    dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
                ce = ce_raw.sum() * world_size / total_ce_tokens.clamp(min=1)
                loss_dict["ce_loss"] = ce.detach()
                loss = loss + ce * ce_weight

            if hasattr(outputs, "mse_loss") and outputs.mse_loss is not None:
                mse_raw = outputs.mse_loss
                total_mse_tokens = torch.tensor(mse_raw.shape[0], device=device)
                if dist.is_initialized():
                    dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse_raw.mean(dim=-1).sum() * world_size / total_mse_tokens.clamp(min=1)
                loss_dict["mse_loss"] = mse.detach()
                loss = loss + mse * mse_weight

            if hasattr(outputs, "fm_loss") and outputs.fm_loss is not None:
                loss_dict["fm_loss"] = outputs.fm_loss.detach()
                loss = loss + outputs.fm_loss

            if not loss_dict:
                loss = outputs.loss if outputs.loss is not None else loss
                loss_dict["foundation_loss"] = loss.detach()

            return loss, loss_dict

        self.base.postforward = _postforward_unified

    def _patch_preforward(self):
        """Patch base.preforward to handle VAE encoding and list-of-tensor fields."""
        original_preforward = self.base.preforward
        has_vae = hasattr(self, "_vae_model")

        def _preforward_with_vae(micro_batch):
            micro_batch = original_preforward(micro_batch)
            device = self.base.device
            # Determine model dtype — try param access, fallback to bfloat16
            try:
                model_dtype = next(self.base.model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.bfloat16

            # Move list-of-tensor fields to device (base preforward only handles plain tensors)
            for key in ("nested_attention_masks",):
                if key in micro_batch and isinstance(micro_batch[key], list):
                    micro_batch[key] = [
                        t.to(device=device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                        for t in micro_batch[key]
                    ]

            # Cast all floating-point tensors to model dtype
            for key, val in micro_batch.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point() and val.dtype != model_dtype:
                    micro_batch[key] = val.to(dtype=model_dtype)

            if has_vae and "padded_images" in micro_batch:
                vae = self._vae_model.to(device=device, dtype=model_dtype)
                with torch.no_grad():
                    micro_batch["padded_latent"] = vae.encode(
                        micro_batch.pop("padded_images")
                    )
            return micro_batch

        self.base.preforward = _preforward_with_vae

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
        elif model_type in ("janus", "multi_modality"):
            self._freeze_janus(model, args)
        else:
            super()._freeze_model_module()
            return

        self._log_trainable_summary(model)

    @staticmethod
    def _log_trainable_summary(model):
        from ..utils.model_utils import pretty_print_trainable_parameters
        from ..utils import helper
        pretty_print_trainable_parameters(model)
        helper.print_device_mem_info("VRAM usage after building model")

    # ------------------------------------------------------------------
    # Bagel / ThinkMorph
    # Ref: Bagel/train/pretrain_unified_navit.py
    # ------------------------------------------------------------------
    def _freeze_bagel(self, model, args):
        if args.train.freeze_vit and hasattr(model, "vit_model"):
            model.vit_model.eval()
            model.vit_model.requires_grad_(False)
            if hasattr(model, "connector"):
                model.connector.requires_grad_(True)

        if args.train.freeze_vae:
            if hasattr(model, "vae_encoder"):
                model.vae_encoder.requires_grad_(False)
            if hasattr(model, "vae_decoder"):
                model.vae_decoder.requires_grad_(False)

        if args.train.freeze_llm and hasattr(model, "language_model"):
            model.language_model.eval()
            model.language_model.requires_grad_(False)

        if args.train.freeze_und:
            if hasattr(model, "config"):
                model.config.freeze_und = True
                logger.info("Bagel: config.freeze_und=True (understanding path detached in forward)")
            if hasattr(model, "connector"):
                model.connector.requires_grad_(False)

    # ------------------------------------------------------------------
    # BLIP3o
    # Ref: BLIP3o/blip3o/train/train.py
    # ------------------------------------------------------------------
    def _freeze_blip3o(self, model, args):
        tunable_parts = args.train.mm_tunable_parts
        if tunable_parts:
            self._freeze_blip3o_tunable_parts(model, tunable_parts)
            return

        if args.train.freeze_vit and hasattr(model, "visual"):
            model.visual.eval()
            model.visual.requires_grad_(False)

        if args.train.freeze_vae and hasattr(model.model, "vae"):
            model.model.vae.requires_grad_(False)

        if args.train.freeze_gen_modules:
            if hasattr(model.model, "dit"):
                model.model.dit.requires_grad_(False)
            if hasattr(model.model, "gen_vision_tower"):
                model.model.gen_vision_tower.requires_grad_(False)

        self._freeze_blip3o_sana(model)

    def _freeze_blip3o_tunable_parts(self, model, tunable_parts_str):
        """BLIP3o mm_tunable_parts mode: freeze all, then selectively unfreeze."""
        model.requires_grad_(False)
        if hasattr(model, "visual"):
            model.visual.eval()
            model.visual.requires_grad_(False)

        parts = [p.strip() for p in tunable_parts_str.split(",")]
        if "mm_vision_tower" in parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model" in parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name:
                    param.requires_grad_(True)
        if "mm_embedding" in parts:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad_(True)

        self._freeze_blip3o_sana(model)

    @staticmethod
    def _freeze_blip3o_sana(model):
        """DiT (sana) always frozen, caption projection always trainable."""
        for name, param in model.named_parameters():
            if "sana" in name:
                param.requires_grad_(False)
        for name, param in model.named_parameters():
            if "caption" in name:
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    # NEO-Chat / SenseNova-U1
    # Ref: SenseNova-U1/training/sensenovavl/train/pipeline.py
    # ------------------------------------------------------------------
    def _freeze_neo_chat(self, model, args):
        if args.train.freeze_vit and hasattr(model, "vision_model"):
            model.vision_model.eval()
            model.vision_model.requires_grad_(False)

        if args.train.freeze_llm and hasattr(model, "language_model"):
            model.language_model.eval()
            model.language_model.requires_grad_(False)

        if args.train.freeze_mlp and hasattr(model, "mlp1"):
            model.mlp1.requires_grad_(False)

        if args.train.freeze_gen_modules and hasattr(model, "fm_modules"):
            model.fm_modules.requires_grad_(False)

        if args.train.unfreeze_vit_layers != 0 and hasattr(model, "vision_model"):
            if hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layers"):
                layers = model.vision_model.encoder.layers[args.train.unfreeze_vit_layers:]
                for name, param in layers.named_parameters():
                    logger.info(f"U1: unfreezing ViT layer: {name}")
                    param.requires_grad_(True)

        if args.train.unfreeze_lm_head and hasattr(model, "language_model"):
            if hasattr(model.language_model, "output"):
                model.language_model.output.requires_grad_(True)
            elif hasattr(model.language_model, "lm_head"):
                model.language_model.lm_head.requires_grad_(True)

        if args.train.unfreeze_mot_gen:
            for name, param in model.named_parameters():
                if "mot_gen" in name:
                    param.requires_grad_(True)
                    logger.info(f"U1: unfreezing MoT gen param: {name}")

    # ------------------------------------------------------------------
    # LatentUM
    # Ref: LatentUM/script/train_interleaved_lang_only.py
    #      LatentUM/script/train_interleaved_vision_only.py
    #      LatentUM/model/decoder/sd_decoder.py (load_mmdit_half_trainable)
    # ------------------------------------------------------------------
    def _freeze_latentum(self, model, args):
        stage = args.train.training_stage

        if stage == "full":
            self._freeze_latentum_full(model, args)
        elif stage == "lang_only":
            self._freeze_latentum_lang_only(model, args)
        elif stage == "vision_only":
            self._freeze_latentum_vision_only(model, args)
        else:
            raise ValueError(f"Unknown training_stage: {stage}. Must be 'full', 'lang_only', or 'vision_only'.")

    def _freeze_latentum_full(self, model, args):
        """Full training with optional selective freezing."""
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

        self._freeze_latentum_mmdit(model)

    def _freeze_latentum_lang_only(self, model, args):
        """Stage 1: train understanding / language branch only."""
        if hasattr(model, "internvl"):
            model.internvl.requires_grad_(False)
        if hasattr(model, "quantizer"):
            model.quantizer.requires_grad_(False)

        if hasattr(model.internvl, "language_model"):
            for name, param in model.internvl.language_model.model.layers.named_parameters():
                if "_vision" not in name:
                    param.requires_grad_(True)
            if hasattr(model.internvl.language_model, "lm_head"):
                model.internvl.language_model.lm_head.requires_grad_(True)

        self._freeze_latentum_mmdit(model)
        logger.info("LatentUM: stage=lang_only — training text path + lm_head")

    def _freeze_latentum_vision_only(self, model, args):
        """Stage 2: train generation / vision branch only."""
        if hasattr(model, "internvl"):
            model.internvl.requires_grad_(False)
        if hasattr(model, "quantizer"):
            model.quantizer.requires_grad_(False)

        if hasattr(model.internvl, "language_model"):
            for name, param in model.internvl.language_model.model.layers.named_parameters():
                if "_vision" in name:
                    param.requires_grad_(True)
        if hasattr(model.internvl, "ar_head"):
            model.internvl.ar_head.requires_grad_(True)

        self._freeze_latentum_mmdit(model)
        logger.info("LatentUM: stage=vision_only — training MoT vision path + ar_head")

    @staticmethod
    def _freeze_latentum_mmdit(model):
        """MMDiT decoder: freeze all except context_embedder and context_block."""
        if not hasattr(model, "transformer"):
            return
        transformer = model.transformer
        transformer.requires_grad_(False)
        if hasattr(transformer, "context_embedder"):
            transformer.context_embedder.requires_grad_(True)
        for name, param in transformer.named_parameters():
            if "context_block" in name:
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Janus
    # Ref: deepseek-ai/Janus (no official training code; freeze pattern
    # follows standard VLM convention: freeze vision encoders, train LLM)
    # ------------------------------------------------------------------
    def _freeze_janus(self, model, args):
        if args.train.freeze_vit and hasattr(model, "vision_model"):
            model.vision_model.eval()
            model.vision_model.requires_grad_(False)

        if args.train.freeze_vae and hasattr(model, "gen_vision_model"):
            model.gen_vision_model.requires_grad_(False)

    # ------------------------------------------------------------------
    # Data pipeline overrides for unified models
    # ------------------------------------------------------------------
    _UNIFIED_MODEL_TYPES = {"bagel", "thinkmorph", "blip3o_qwen", "neo_chat", "latentum", "janus", "multi_modality"}

    def _build_model_assets(self):
        model_type = self.base.model_config.model_type
        if model_type not in self._UNIFIED_MODEL_TYPES:
            super()._build_model_assets()
            return

        args: UnifiedVeOmniArguments = self.base.args
        from ..data.multimodal.multimodal_chat_template import build_multimodal_chat_template

        try:
            from ..models.auto import build_processor
            self.base.processor = build_processor(args.model.tokenizer_path)
        except Exception:
            from transformers import AutoTokenizer
            self.base.processor = AutoTokenizer.from_pretrained(
                args.model.tokenizer_path, trust_remote_code=True, padding_side="right"
            )
            logger.info("UnifiedMultimodalTrainer: loaded AutoTokenizer as processor fallback")

        tokenizer = getattr(self.base.processor, "tokenizer", self.base.processor)
        self.base.chat_template = build_multimodal_chat_template(
            args.data.chat_template, tokenizer
        )
        self.base.model_assets = [self.base.processor, self.base.chat_template]

    def _build_data_transform(self):
        model_type = self.base.model_config.model_type
        if model_type not in self._UNIFIED_MODEL_TYPES:
            super()._build_data_transform()
            return

        args: UnifiedVeOmniArguments = self.base.args
        tokenizer = getattr(self.base.processor, "tokenizer", self.base.processor)

        if model_type in ("bagel", "thinkmorph"):
            if args.train.mse_weight > 0:
                image_root = args.data.mm_configs.get("image_root", "") if hasattr(args.data, "mm_configs") and args.data.mm_configs else ""
                self.base.data_transform = partial(
                    _unified_bagel_multimodal_transform,
                    tokenizer=tokenizer,
                    max_seq_len=args.data.max_seq_len,
                    image_root=image_root,
                )
                logger.info(f"UnifiedMultimodalTrainer: using Bagel multimodal transform (CE+MSE) for {model_type}, image_root={image_root}")
            else:
                self.base.data_transform = partial(
                    _unified_bagel_text_only_transform,
                    tokenizer=tokenizer,
                    max_seq_len=args.data.max_seq_len,
                )
                logger.info(f"UnifiedMultimodalTrainer: using Bagel packed text-only transform for {model_type}")
        elif model_type == "neo_chat":
            if args.train.mse_weight > 0:
                image_root = args.data.mm_configs.get("image_root", "") if hasattr(args.data, "mm_configs") and args.data.mm_configs else ""
                max_pixels = int(args.data.mm_configs.get("image_max_pixels", 602112)) if hasattr(args.data, "mm_configs") and args.data.mm_configs else 602112
                self.base.data_transform = partial(
                    _unified_u1_multimodal_transform,
                    tokenizer=tokenizer,
                    max_seq_len=args.data.max_seq_len,
                    image_root=image_root,
                    max_pixels=max_pixels,
                )
                logger.info(f"UnifiedMultimodalTrainer: using U1 multimodal transform (CE+FM), image_root={image_root}")
            else:
                self.base.data_transform = partial(
                    _unified_text_only_transform,
                    tokenizer=tokenizer,
                    max_seq_len=args.data.max_seq_len,
                )
                logger.info("UnifiedMultimodalTrainer: using text-only data transform for neo_chat")
        elif model_type in ("janus", "multi_modality"):
            self.base.data_transform = partial(
                _unified_janus_text_only_transform,
                tokenizer=tokenizer,
                max_seq_len=args.data.max_seq_len,
            )
            logger.info("UnifiedMultimodalTrainer: using Janus DeepSeek-style text-only transform")
        else:
            self.base.data_transform = partial(
                _unified_text_only_transform,
                tokenizer=tokenizer,
                max_seq_len=args.data.max_seq_len,
            )
            logger.info(f"UnifiedMultimodalTrainer: using text-only data transform for {model_type}")

    def _build_collate_fn(self):
        model_type = self.base.model_config.model_type
        if model_type not in self._UNIFIED_MODEL_TYPES:
            super()._build_collate_fn()
            return

        if model_type in ("bagel", "thinkmorph"):
            self.base.collate_fn = _BagelPackedCollator()
            logger.info(f"UnifiedMultimodalTrainer: using Bagel packed collator for {model_type}")
        elif model_type == "neo_chat" and self.base.args.train.mse_weight > 0:
            self.base.collate_fn = _PassthroughCollator()
            logger.info("UnifiedMultimodalTrainer: using passthrough collator for U1 multimodal")
        else:
            from ..data import MainCollator
            self.base.collate_fn = MainCollator(
                pad_to_length=self.base.args.train.pad_to_length,
                seq_classification=False,
                data_collate_info={},
                metadata_collate_func=None,
            )


def _unified_text_only_transform(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """Simple text-only data transform for unified multimodal models.

    Strips <image>/<video> markers and tokenizes as plain conversation.
    """
    conversations = sample.get("conversations", sample)
    role_mapping = {"human": "user", "gpt": "assistant"}
    marker_re = re.compile(r"<image>|<video>|<audio>")

    input_ids, attention_mask, labels = [], [], []

    for message in conversations:
        role = role_mapping.get(message.get("from", ""), message.get("from", ""))
        value = marker_re.sub("", message.get("value", "")).strip()
        if not value:
            value = " "

        role_header = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        content_ids = tokenizer.encode(value, add_special_tokens=False)
        end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        msg_ids = role_header + content_ids + end_ids
        input_ids.extend(msg_ids)
        attention_mask.extend([1] * len(msg_ids))

        if role == "assistant":
            labels.extend(msg_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(msg_ids))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        labels = labels[:max_seq_len]

    return [{
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }]


def _unified_bagel_text_only_transform(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """Text-only data transform producing Bagel NaViT packed format.

    Aligned with Bagel/data/dataset_base.py pack_sequence + to_tensor.
    Converts conversation to packed_text_ids / packed_text_indexes /
    sample_lens / packed_position_ids / ce_loss_indexes / packed_label_ids.
    """
    conversations = sample.get("conversations", sample)
    role_mapping = {"human": "user", "gpt": "assistant"}
    marker_re = re.compile(r"<image>|<video>|<audio>")

    input_ids: List[int] = []
    labels: List[int] = []

    for message in conversations:
        role = role_mapping.get(message.get("from", ""), message.get("from", ""))
        value = marker_re.sub("", message.get("value", "")).strip()
        if not value:
            value = " "

        role_header = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        content_ids = tokenizer.encode(value, add_special_tokens=False)
        end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        msg_ids = role_header + content_ids + end_ids
        input_ids.extend(msg_ids)

        if role == "assistant":
            labels.extend(msg_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(msg_ids))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    seq_len = len(input_ids)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    valid_mask = labels_t != IGNORE_INDEX
    ce_loss_indexes = torch.where(valid_mask)[0]
    packed_label_ids = labels_t[valid_mask]

    result: Dict[str, Any] = {
        "sequence_length": seq_len,
        "packed_text_ids": input_ids_t,
        "packed_text_indexes": torch.arange(seq_len, dtype=torch.long),
        "sample_lens": [seq_len],
        "packed_position_ids": torch.arange(seq_len, dtype=torch.long),
        "labels": labels_t,
    }
    if ce_loss_indexes.numel() > 0:
        result["ce_loss_indexes"] = ce_loss_indexes
        result["packed_label_ids"] = packed_label_ids

    return [result]


def _unified_janus_text_only_transform(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """Text-only data transform for Janus (DeepSeek-style conversation format).

    Format: <|User|>: {message}\n\n<|Assistant|>: {response}<eos>
    Aligned with Janus/janus/utils/conversation.py (SeparatorStyle.DeepSeek).
    """
    conversations = sample.get("conversations", sample)
    role_mapping = {"human": "user", "gpt": "assistant"}
    marker_re = re.compile(r"<image>|<video>|<audio>")

    input_ids, attention_mask, labels = [], [], []

    eos_id = tokenizer.eos_token_id
    user_header_ids = tokenizer.encode("<|User|>: ", add_special_tokens=False)
    assistant_header_ids = tokenizer.encode("<|Assistant|>: ", add_special_tokens=False)
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    for message in conversations:
        role = role_mapping.get(message.get("from", ""), message.get("from", ""))
        value = marker_re.sub("", message.get("value", "")).strip()
        if not value:
            value = " "

        content_ids = tokenizer.encode(value, add_special_tokens=False)

        if role == "assistant":
            msg_ids = assistant_header_ids + content_ids + [eos_id]
            input_ids.extend(msg_ids)
            attention_mask.extend([1] * len(msg_ids))
            labels.extend([IGNORE_INDEX] * len(assistant_header_ids) + content_ids + [eos_id])
        else:
            msg_ids = user_header_ids + content_ids + sep_ids
            input_ids.extend(msg_ids)
            attention_mask.extend([1] * len(msg_ids))
            labels.extend([IGNORE_INDEX] * len(msg_ids))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        labels = labels[:max_seq_len]

    return [{
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }]


# ---------------------------------------------------------------------------
# Bagel/ThinkMorph multimodal helpers (ported from ThinkMorph/data/)
# ---------------------------------------------------------------------------

def _resize_image(img: Image.Image, max_size: int, min_size: int, stride: int, max_pixels: int = 14*14*9*1024):
    width, height = img.size
    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    new_w = max(stride, int(round(width * scale / stride) * stride))
    new_h = max(stride, int(round(height * scale / stride) * stride))
    if new_w * new_h > max_pixels:
        s = (max_pixels / (new_w * new_h)) ** 0.5
        new_w = max(stride, int(round(new_w * s / stride) * stride))
        new_h = max(stride, int(round(new_h * s / stride) * stride))
    return TF.resize(img, (new_h, new_w), InterpolationMode.BICUBIC, antialias=True)


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return t


def _patchify(image_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    c, h, w = image_tensor.shape
    assert h % patch_size == 0 and w % patch_size == 0
    image_tensor = image_tensor.reshape(c, h // patch_size, patch_size, w // patch_size, patch_size)
    image_tensor = torch.einsum("chpwq->hwpqc", image_tensor)
    return image_tensor.reshape(-1, patch_size**2 * c)


def _get_flattened_pos_ids(img_h: int, img_w: int, patch_size: int, max_patches_per_side: int) -> torch.Tensor:
    nh, nw = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, nh)
    coords_w = torch.arange(0, nw)
    return (coords_h[:, None] * max_patches_per_side + coords_w).flatten()


def _prepare_attention_mask(split_lens: List[int], attn_modes: List[str]) -> torch.Tensor:
    sample_len = sum(split_lens)
    mask = torch.zeros((sample_len, sample_len), dtype=torch.bool)
    csum = 0
    for s, mode in zip(split_lens, attn_modes):
        if mode == "causal":
            mask[csum:csum+s, csum:csum+s] = torch.ones((s, s)).tril().bool()
            mask[csum:csum+s, :csum] = True
        else:
            mask[csum:csum+s, csum:csum+s] = True
            mask[csum:csum+s, :csum] = True
        csum += s
    csum = 0
    for s, mode in zip(split_lens, attn_modes):
        if mode == "noise":
            mask[:, csum:csum+s] = False
            mask[csum:csum+s, csum:csum+s] = True
        csum += s
    return torch.zeros_like(mask, dtype=torch.float).masked_fill_(~mask, float("-inf"))


def _unified_bagel_multimodal_transform(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    image_root: str = "",
    vit_max_size: int = 980,
    vit_min_size: int = 224,
    vit_patch_size: int = 14,
    vae_max_size: int = 1024,
    vae_min_size: int = 512,
    vae_downsample: int = 16,
    max_latent_size: int = 64,
    latent_patch_size: int = 2,
    vit_max_num_patch_per_side: int = 70,
    cfg_text_dropout: float = 0.1,
    cfg_vit_dropout: float = 0.1,
    cfg_vae_dropout: float = 0.1,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Full multimodal data transform for Bagel/ThinkMorph NaViT packed format.

    Handles input images (ViT understanding) and output images (VAE generation).
    Follows the official ThinkMorph pack_sequence logic.
    """
    conversations = sample.get("conversations", [])
    image_paths = sample.get("images", [])

    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    role_mapping = {"human": "user", "gpt": "assistant"}
    image_iter = iter(image_paths)

    elements = []
    for msg in conversations:
        role = role_mapping.get(msg.get("from", ""), msg.get("from", ""))
        value = msg.get("value", "")
        is_assistant = (role == "assistant")

        parts = value.split("<image>")
        for idx, part in enumerate(parts):
            text = part.strip()
            if text:
                elements.append({"type": "text", "text": text, "loss": int(is_assistant)})
            if idx < len(parts) - 1:
                img_path = next(image_iter, None)
                if img_path is None:
                    continue
                full_path = os.path.join(image_root, img_path) if image_root else img_path
                if is_assistant:
                    elements.append({"type": "vae_image", "path": full_path, "loss": 1})
                else:
                    elements.append({"type": "vit_image", "path": full_path, "loss": 0})

    # Build packed sequence
    packed_text_ids = []
    packed_text_indexes = []
    packed_position_ids = []
    ce_loss_indexes = []
    packed_label_ids = []
    packed_vit_tokens = []
    packed_vit_token_indexes = []
    packed_vit_position_ids = []
    vit_token_seqlens = []
    packed_vae_token_indexes = []
    packed_timesteps = []
    mse_loss_indexes = []
    vae_image_tensors = []
    vae_latent_shapes = []
    vae_latent_position_ids = []
    split_lens = []
    attn_modes = []

    curr = 0
    curr_rope_id = 0

    for elem in elements:
        curr_split_len = 0

        if elem["type"] == "text":
            text_ids = tokenizer.encode(elem["text"], add_special_tokens=False)
            if not text_ids:
                continue
            if elem["loss"] == 0 and random.random() < cfg_text_dropout:
                continue

            shifted = [bos_id] + text_ids
            packed_text_ids.extend(shifted)
            packed_text_indexes.extend(range(curr, curr + len(shifted)))
            if elem["loss"] == 1:
                ce_loss_indexes.extend(range(curr, curr + len(shifted)))
                packed_label_ids.extend(text_ids + [eos_id])
            curr += len(shifted)
            curr_split_len += len(shifted)

            # eos token
            packed_text_ids.append(eos_id)
            packed_text_indexes.append(curr)
            curr += 1
            curr_split_len += 1

            attn_modes.append("causal")
            packed_position_ids.extend(range(curr_rope_id, curr_rope_id + curr_split_len))
            curr_rope_id += curr_split_len

        elif elem["type"] == "vit_image":
            if random.random() < cfg_vit_dropout:
                curr_rope_id += 1
                continue
            try:
                img = Image.open(elem["path"]).convert("RGB")
                img = _resize_image(img, vit_max_size, vit_min_size, vit_patch_size)
                img_tensor = _image_to_tensor(img)
            except Exception:
                continue

            # vision_start token
            packed_text_ids.append(vision_start_id)
            packed_text_indexes.append(curr)
            curr += 1
            curr_split_len += 1

            vit_tokens = _patchify(img_tensor, vit_patch_size)
            num_tokens = vit_tokens.shape[0]
            packed_vit_token_indexes.extend(range(curr, curr + num_tokens))
            packed_vit_tokens.append(vit_tokens)
            vit_token_seqlens.append(num_tokens)
            packed_vit_position_ids.append(
                _get_flattened_pos_ids(img_tensor.shape[1], img_tensor.shape[2],
                                      vit_patch_size, vit_max_num_patch_per_side)
            )
            curr += num_tokens
            curr_split_len += num_tokens

            # vision_end token
            packed_text_ids.append(vision_end_id)
            packed_text_indexes.append(curr)
            curr += 1
            curr_split_len += 1

            attn_modes.append("full")
            packed_position_ids.extend([curr_rope_id] * curr_split_len)
            curr_rope_id += 1

        elif elem["type"] == "vae_image":
            if elem["loss"] == 0 and random.random() < cfg_vae_dropout:
                curr_rope_id += 1
                continue
            try:
                img = Image.open(elem["path"]).convert("RGB")
                img = _resize_image(img, vae_max_size, vae_min_size, vae_downsample)
                img_tensor = _image_to_tensor(img)
            except Exception:
                continue

            # vision_start token
            packed_text_ids.append(vision_start_id)
            packed_text_indexes.append(curr)
            curr += 1
            curr_split_len += 1

            # VAE image
            vae_image_tensors.append(img_tensor)
            H, W = img_tensor.shape[1], img_tensor.shape[2]
            h, w = H // vae_downsample, W // vae_downsample
            vae_latent_shapes.append((h, w))
            vae_latent_position_ids.append(
                _get_flattened_pos_ids(H, W, vae_downsample, max_latent_size)
            )

            num_tokens = h * w
            packed_vae_token_indexes.extend(range(curr, curr + num_tokens))
            if elem["loss"] == 1:
                mse_loss_indexes.extend(range(curr, curr + num_tokens))
                timestep = float(np.random.randn())
            else:
                timestep = float("-inf")
            packed_timesteps.extend([timestep] * num_tokens)
            curr += num_tokens
            curr_split_len += num_tokens

            # vision_end token
            packed_text_ids.append(vision_end_id)
            packed_text_indexes.append(curr)
            curr += 1
            curr_split_len += 1

            if elem["loss"] == 1:
                attn_modes.append("noise")
            else:
                attn_modes.append("full")
            packed_position_ids.extend([curr_rope_id] * curr_split_len)
            curr_rope_id += 1

        if curr_split_len > 0:
            split_lens.append(curr_split_len)

    # Truncate if over max_seq_len
    if curr > max_seq_len:
        return _unified_bagel_text_only_transform(sample, tokenizer, max_seq_len)

    seq_len = curr

    # Generate per-sample attention mask only when noise segments exist
    # (noise mode is critical for generation; causal/full-only can use fast flash_attn)
    has_noise = "noise" in attn_modes
    nested_attention_masks = None
    if has_noise:
        attn_mask = _prepare_attention_mask(split_lens, attn_modes)
        nested_attention_masks = [attn_mask]

    # Build a labels tensor for count_loss_token compatibility
    labels_full = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)
    if ce_loss_indexes:
        for i, idx in enumerate(ce_loss_indexes):
            labels_full[idx] = packed_label_ids[i]

    result: Dict[str, Any] = {
        "sequence_length": seq_len,
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "sample_lens": [seq_len],
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        "labels": labels_full,
    }
    if nested_attention_masks is not None:
        result["nested_attention_masks"] = nested_attention_masks

    if ce_loss_indexes:
        result["ce_loss_indexes"] = torch.tensor(ce_loss_indexes, dtype=torch.long)
        result["packed_label_ids"] = torch.tensor(packed_label_ids, dtype=torch.long)

    if packed_vit_tokens:
        result["packed_vit_tokens"] = torch.cat(packed_vit_tokens, dim=0)
        result["packed_vit_position_ids"] = torch.cat(packed_vit_position_ids, dim=0)
        result["packed_vit_token_indexes"] = torch.tensor(packed_vit_token_indexes, dtype=torch.long)
        result["vit_token_seqlens"] = torch.tensor(vit_token_seqlens, dtype=torch.int)

    if vae_image_tensors:
        image_sizes = [t.shape for t in vae_image_tensors]
        max_img_size = [max(s) for s in zip(*image_sizes)]
        padded_images = torch.zeros(len(vae_image_tensors), *max_img_size)
        for i, t in enumerate(vae_image_tensors):
            padded_images[i, :, :t.shape[1], :t.shape[2]] = t
        result["padded_images"] = padded_images
        result["patchified_vae_latent_shapes"] = vae_latent_shapes
        result["packed_latent_position_ids"] = torch.cat(vae_latent_position_ids, dim=0)
        result["packed_vae_token_indexes"] = torch.tensor(packed_vae_token_indexes, dtype=torch.long)

    if packed_timesteps:
        result["packed_timesteps"] = torch.tensor(packed_timesteps, dtype=torch.float)
        result["mse_loss_indexes"] = torch.tensor(mse_loss_indexes, dtype=torch.long)

    return [result]


# ---------------------------------------------------------------------------
# U1 (neo_chat) multimodal transform
# ---------------------------------------------------------------------------

def _unified_u1_multimodal_transform(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    image_root: str = "",
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    max_pixels: int = 602112,
    min_pixels: int = 65536,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """Multimodal data transform for SenseNova-U1 (neo_chat).

    Aligned with official SenseNova-U1 training implementation.
    All images (understanding + generation) are unified into a single pixel_values
    tensor with image_for_gen_flags indicating which images are for generation.
    Images use ImageNet normalization (model handles conversion internally).
    """
    IMG_START_ID = 151670   # <img>
    IMG_END_ID = 151671     # </img>
    IMG_CONTEXT_ID = 151669 # <IMG_CONTEXT>

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    merge_size = int(1 / downsample_ratio)  # 2

    conversations = sample.get("conversations", [])
    images_list = sample.get("images", sample.get("image", []))
    if isinstance(images_list, str):
        images_list = [images_list]

    role_mapping = {"human": "user", "gpt": "assistant"}

    # Walk through conversations to determine image order and roles
    # Each <image> marker maps to the next image in images_list
    image_info = []  # list of (path, is_gen) in order of appearance
    img_idx = 0
    for msg in conversations:
        role = role_mapping.get(msg.get("from", ""), msg.get("from", ""))
        value = msg.get("value", "")
        count = value.count("<image>")
        for _ in range(count):
            if img_idx < len(images_list):
                is_gen = (role == "assistant")
                image_info.append((images_list[img_idx], is_gen))
                img_idx += 1

    def _load_and_patchify(path):
        """Load image, resize, normalize with ImageNet stats, patchify to [N, 3*p*p]."""
        full_path = os.path.join(image_root, path) if image_root and not os.path.isabs(path) else path
        img = Image.open(full_path).convert("RGB")
        w, h = img.size
        # Round to factor = patch_size * merge_size (32) to ensure grid dims are even
        factor = patch_size * merge_size
        pixels = w * h
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            w, h = int(w * scale), int(h * scale)
        elif pixels < min_pixels:
            scale = (min_pixels / pixels) ** 0.5
            w, h = int(w * scale), int(h * scale)
        w = max(factor, (w // factor) * factor)
        h = max(factor, (h // factor) * factor)
        # Clamp to max_pixels after rounding
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            w = max(factor, int(w * scale) // factor * factor)
            h = max(factor, int(h * scale) // factor * factor)
        img = img.resize((w, h), Image.BICUBIC)
        # ImageNet normalize
        t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # Patchify to [N_patches, 3*p*p]
        c, ph, pw = t.shape
        gh, gw = ph // patch_size, pw // patch_size
        patches = t.view(c, gh, patch_size, gw, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # [gh, gw, 3, p, p]
        patches = patches.view(gh * gw, c * patch_size * patch_size)  # [N, 3*p*p]
        return patches, (gh, gw)

    # Load all images in order, track grid_hw and gen flags
    all_pixel_values = []
    all_grid_hw = []
    all_is_gen = []
    num_tokens_per_image = []  # LLM tokens after downsample

    for path, is_gen in image_info:
        try:
            patches, (gh, gw) = _load_and_patchify(path)
            all_pixel_values.append(patches)
            all_grid_hw.append((gh, gw))
            all_is_gen.append(is_gen)
            num_tokens_per_image.append((gh // merge_size) * (gw // merge_size))
        except Exception as e:
            logger.warning(f"U1: failed to load image {path}: {e}")
            num_tokens_per_image.append(0)
            all_is_gen.append(is_gen)

    # Build token sequence
    input_ids = []
    labels = []
    image_gen_indicator_list = []
    img_order_idx = 0  # index into image_info order

    for msg in conversations:
        role = role_mapping.get(msg.get("from", ""), msg.get("from", ""))
        value = msg.get("value", "")

        role_header = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        parts = value.split("<image>")
        content_ids = []
        content_gen_indicators = []

        for i, part in enumerate(parts):
            if i > 0:
                # Insert image placeholder tokens
                if img_order_idx < len(num_tokens_per_image) and num_tokens_per_image[img_order_idx] > 0:
                    n_tokens = num_tokens_per_image[img_order_idx]
                    is_gen = all_is_gen[img_order_idx]
                    img_tokens = [IMG_START_ID] + [IMG_CONTEXT_ID] * n_tokens + [IMG_END_ID]
                    content_ids.extend(img_tokens)
                    if is_gen:
                        content_gen_indicators.extend([0] + [1] * n_tokens + [0])
                    else:
                        content_gen_indicators.extend([0] * len(img_tokens))
                img_order_idx += 1

            if part.strip():
                text_ids = tokenizer.encode(part, add_special_tokens=False)
                content_ids.extend(text_ids)
                content_gen_indicators.extend([0] * len(text_ids))

        msg_ids = role_header + content_ids + end_ids
        msg_gen = [0] * len(role_header) + content_gen_indicators + [0] * len(end_ids)

        input_ids.extend(msg_ids)
        image_gen_indicator_list.extend(msg_gen)

        if role == "assistant":
            # Role header: no loss; content+end: loss except gen tokens
            labels.extend([IGNORE_INDEX] * len(role_header))
            for tok_id, is_gen in zip(content_ids + end_ids, content_gen_indicators + [0] * len(end_ids)):
                labels.append(IGNORE_INDEX if is_gen else tok_id)
        else:
            labels.extend([IGNORE_INDEX] * len(msg_ids))

    # Truncate
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        image_gen_indicator_list = image_gen_indicator_list[:max_seq_len]

    seq_len = len(input_ids)

    if not all_pixel_values:
        return _unified_text_only_transform(sample, tokenizer, max_seq_len)

    # Build result — unified format matching official U1 forward
    result: Dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        "pixel_values": torch.cat(all_pixel_values, dim=0),  # [N_total_patches, 3*p*p]
        "grid_hw": torch.tensor(all_grid_hw, dtype=torch.long),  # [N_images, 2]
        "image_for_gen_flags": torch.tensor(all_is_gen, dtype=torch.bool),  # [N_images]
        "image_gen_indicators": torch.tensor(
            image_gen_indicator_list, dtype=torch.bool
        ).unsqueeze(0),  # [1, S]
    }

    return [result]


class _PassthroughCollator:
    """Passthrough collator for multimodal models.

    Returns single sample as-is. When dynamic batching sends multiple samples,
    only the first is used (multimodal data has variable shapes that cannot be stacked).
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return features[0]


class _BagelPackedCollator(_PassthroughCollator):
    """Passthrough collator for Bagel NaViT packed format."""
    pass
