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
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

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
            self.base.data_transform = partial(
                _unified_bagel_text_only_transform,
                tokenizer=tokenizer,
                max_seq_len=args.data.max_seq_len,
            )
            logger.info(f"UnifiedMultimodalTrainer: using Bagel packed text-only transform for {model_type}")
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


class _BagelPackedCollator:
    """Passthrough collator for Bagel NaViT packed format.

    Each sample from the transform is already a complete packed sequence.
    With micro_batch_size=1, just return it as-is.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) == 1:
            return features[0]
        raise NotImplementedError(
            "Bagel packed collator: micro_batch_size > 1 not supported for packed format. "
            "Use micro_batch_size=1."
        )
