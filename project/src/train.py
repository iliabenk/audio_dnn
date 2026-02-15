#!/usr/bin/env python3
"""
HuBERT ASR Fine-tuning Training Script

Fine-tune a pre-trained HuBERT model on LibriSpeech for ASR using CTC loss.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/debug.yaml
    python -m src.train --config configs/gpu.yaml --resume outputs/checkpoint-500
"""

import argparse
import logging
from pathlib import Path

from .config import Config
from .data.dataset import LibriSpeechDataset
from .evaluation.metrics import WERCalculator
from .model.hubert_ctc import HuBERTForASR
from .training.trainer import create_trainer
from .utils.device import DeviceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune HuBERT for ASR on LibriSpeech"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config.from_yaml(args.config)

    # Setup device
    device = DeviceManager.get_device(config.device.prefer)
    device_info = DeviceManager.get_device_info(device)
    logger.info(f"Using device: {device_info}")

    # Build model and processor
    logger.info(f"Loading model: {config.model.name}")
    model_builder = HuBERTForASR(config.model)
    processor = model_builder.build_processor()
    model = model_builder.build_model(vocab_size=len(processor.tokenizer))

    # Log parameter counts
    param_info = model_builder.count_parameters(model)
    logger.info(
        f"Model parameters: {param_info['total']:,} total, "
        f"{param_info['trainable']:,} trainable "
        f"({param_info['trainable_percent']:.1f}%)"
    )

    # Move model to device
    model = model.to(device)

    # Load and prepare datasets
    logger.info("Loading LibriSpeech dataset...")
    dataset_loader = LibriSpeechDataset(
        dataset_config=config.dataset,
        audio_config=config.audio,
        processor=processor,
    )

    logger.info("Preparing training dataset...")
    train_dataset = dataset_loader.get_train_dataset()
    logger.info(f"Training samples: {len(train_dataset)}")

    logger.info("Preparing evaluation dataset...")
    eval_split = config.evaluation.eval_splits[0] if config.evaluation.eval_splits else "validation"
    eval_dataset = dataset_loader.get_eval_dataset(eval_split)
    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Create WER calculator for metrics
    wer_calculator = WERCalculator(processor)

    # Create trainer
    logger.info("Setting up trainer...")
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.training,
        device=device,
        compute_metrics=wer_calculator.compute_metrics_for_trainer,
    )

    # Train
    logger.info("Starting training...")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    final_path = Path(config.training.output_dir) / "final"
    logger.info(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    # Run final evaluation on test splits
    logger.info("Running final evaluation on test splits...")
    for test_split in config.evaluation.test_splits:
        try:
            test_dataset = dataset_loader.get_eval_dataset(test_split)
            eval_results = trainer.evaluate(test_dataset)
            wer = eval_results.get("eval_wer", eval_results.get("wer", "N/A"))
            logger.info(f"  {test_split}: WER = {WERCalculator.format_wer(wer) if isinstance(wer, float) else wer}")
        except Exception as e:
            logger.warning(f"  {test_split}: Failed to evaluate - {e}")

    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
