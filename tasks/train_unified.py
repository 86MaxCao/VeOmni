"""Unified training entry point for all 5 multimodal models in VeOmni.

Supports: bagel, thinkmorph, blip3o, u1 (neo_chat), latentum.

Usage:
    torchrun --nproc_per_node=1 tasks/train_unified.py \
        --config configs/multimodal/bagel/sft.yaml
"""

from veomni.arguments import parse_args
from veomni.trainer.unified_multimodal_trainer import UnifiedMultimodalTrainer, UnifiedVeOmniArguments


if __name__ == "__main__":
    args = parse_args(UnifiedVeOmniArguments)
    trainer = UnifiedMultimodalTrainer(args)
    trainer.train()
