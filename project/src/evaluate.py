#!/usr/bin/env python3
"""
HuBERT ASR Evaluation Script

Evaluate a fine-tuned HuBERT model on LibriSpeech test splits.

Usage:
    python -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other
    python -m src.evaluate --model outputs/hubert-finetuned/final --output results.json
"""

import os
# Disable TorchCodec to avoid FFmpeg compatibility issues - use soundfile instead
os.environ["HF_AUDIO_DECODER"] = "soundfile"

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertForCTC, Wav2Vec2Processor

from .config import AudioConfig, DatasetConfig
from .data.collator import CTCDataCollator
from .data.dataset import LibriSpeechDataset
from .evaluation.metrics import WERCalculator
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
        description="Evaluate HuBERT ASR model on LibriSpeech"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test.clean", "test.other"],
        help="Dataset splits to evaluate (e.g., test.clean test.other)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="clean",
        help="LibriSpeech subset (clean or other)",
    )
    return parser.parse_args()


def evaluate_split(
    model,
    processor,
    wer_calculator,
    split: str,
    batch_size: int,
    subset: str,
    device: torch.device,
) -> dict:
    """Evaluate model on a single dataset split.

    Args:
        model: HuBERT model.
        processor: Wav2Vec2 processor.
        wer_calculator: WER calculator instance.
        split: Dataset split name.
        batch_size: Batch size for evaluation.
        subset: LibriSpeech subset.
        device: Compute device.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Configure dataset
    dataset_config = DatasetConfig(
        name="librispeech_asr",
        subset=subset,
        train_split="train.100",  # Not used for eval
    )
    audio_config = AudioConfig()

    # Load dataset
    dataset_loader = LibriSpeechDataset(
        dataset_config=dataset_config,
        audio_config=audio_config,
        processor=processor,
    )

    eval_dataset = dataset_loader.get_eval_dataset(split)

    # Create data collator and dataloader
    data_collator = CTCDataCollator(processor=processor)

    def collate_fn(batch):
        return data_collator(batch)

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Run inference
    all_predictions = []
    all_references = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            # Move to device
            input_values = batch["input_values"].to(device)

            # Forward pass
            outputs = model(input_values)
            logits = outputs.logits.cpu().numpy()

            # Decode predictions
            predictions = wer_calculator.decode_predictions(logits)
            all_predictions.extend(predictions)

            # Decode labels
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(labels, group_tokens=False)
            all_references.extend(references)

    # Compute WER
    metrics = wer_calculator.compute_wer(all_predictions, all_references)

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup device
    device = DeviceManager.get_device()
    device_info = DeviceManager.get_device_info(device)
    logger.info(f"Using device: {device_info}")

    # Load model and processor
    logger.info(f"Loading model from {args.model}")
    model = HubertForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    # Create WER calculator
    wer_calculator = WERCalculator(processor)

    # Evaluate on each split
    results = {}
    for split in args.splits:
        logger.info(f"Evaluating on {split}...")
        try:
            metrics = evaluate_split(
                model=model,
                processor=processor,
                wer_calculator=wer_calculator,
                split=split,
                batch_size=args.batch_size,
                subset=args.subset,
                device=device,
            )
            results[split] = metrics
            logger.info(f"  WER: {WERCalculator.format_wer(metrics['wer'])}")
            logger.info(
                f"  Details: S={metrics['substitutions']}, "
                f"I={metrics['insertions']}, D={metrics['deletions']}"
            )
        except Exception as e:
            logger.error(f"  Failed to evaluate {split}: {e}")
            results[split] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for split, metrics in results.items():
        if "error" in metrics:
            print(f"{split}: ERROR - {metrics['error']}")
        else:
            print(f"{split}: WER = {WERCalculator.format_wer(metrics['wer'])}")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
