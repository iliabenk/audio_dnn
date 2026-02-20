#!/usr/bin/env python3
"""
HuBERT ASR Evaluation Script

Evaluate a fine-tuned HuBERT model on LibriSpeech test splits.

Usage:
    # Single GPU
    python -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other

    # Multi-GPU (distributed)
    accelerate launch --multi_gpu -m src.evaluate --model outputs/hubert-finetuned/final --splits test.clean test.other

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
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertForCTC, Wav2Vec2Processor

from .config import AudioConfig, DatasetConfig
from .data.collator import CTCDataCollator
from .data.dataset import LibriSpeechDataset
from .evaluation.decoder import CTCDecoder
from .evaluation.metrics import WERCalculator

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
    # Language model decoding options
    parser.add_argument(
        "--lm_path",
        type=str,
        default=None,
        help="Path to KenLM language model (.arpa or .bin) for beam search decoding",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=100,
        help="Beam width for beam search (default: 100)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="LM weight for beam search (default: 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Word insertion bonus for beam search (default: 1.0)",
    )
    return parser.parse_args()


def evaluate_split(
    model,
    processor,
    wer_calculator,
    split: str,
    batch_size: int,
    subset: str,
    accelerator: Accelerator,
) -> dict:
    """Evaluate model on a single dataset split.

    Args:
        model: HuBERT model.
        processor: Wav2Vec2 processor.
        wer_calculator: WER calculator instance.
        split: Dataset split name.
        batch_size: Batch size for evaluation.
        subset: LibriSpeech subset.
        accelerator: Accelerator instance for distributed evaluation.

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

    # Prepare dataloader for distributed evaluation
    dataloader = accelerator.prepare(dataloader)

    # Run inference
    all_predictions = []
    all_references = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}", disable=not accelerator.is_main_process):
            # Input is already on correct device via accelerator
            input_values = batch["input_values"]

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

    # Gather predictions and references from all processes
    all_predictions_gathered = accelerator.gather_for_metrics(all_predictions)
    all_references_gathered = accelerator.gather_for_metrics(all_references)

    # Compute WER only on main process
    if accelerator.is_main_process:
        metrics = wer_calculator.compute_wer(all_predictions_gathered, all_references_gathered)
    else:
        metrics = {}

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup accelerator for distributed evaluation
    accelerator = Accelerator()

    # Configure logging only on main process
    if accelerator.is_main_process:
        logger.info(f"Using device: {accelerator.device}")
        if accelerator.num_processes > 1:
            logger.info(f"Running distributed evaluation with {accelerator.num_processes} processes")

    # Load model and processor
    if accelerator.is_main_process:
        logger.info(f"Loading model from {args.model}")
    model = HubertForCTC.from_pretrained(args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.model)

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model.eval()

    # Create decoder (beam search + LM if specified)
    decoder = None
    if args.lm_path:
        if accelerator.is_main_process:
            logger.info(f"Initializing CTC decoder with LM from {args.lm_path}")
        decoder = CTCDecoder(
            processor=processor,
            use_lm=True,
            lm_path=args.lm_path,
            beam_width=args.beam_width,
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        if accelerator.is_main_process:
            logger.info("Using greedy CTC decoding (no LM)")

    # Create WER calculator
    wer_calculator = WERCalculator(processor, decoder=decoder)

    # Evaluate on each split
    results = {}
    for split in args.splits:
        if accelerator.is_main_process:
            logger.info(f"Evaluating on {split}...")
        try:
            metrics = evaluate_split(
                model=model,
                processor=processor,
                wer_calculator=wer_calculator,
                split=split,
                batch_size=args.batch_size,
                subset=args.subset,
                accelerator=accelerator,
            )
            results[split] = metrics
            if accelerator.is_main_process:
                logger.info(f"  WER: {WERCalculator.format_wer(metrics['wer'])}")
                logger.info(
                    f"  Details: S={metrics['substitutions']}, "
                    f"I={metrics['insertions']}, D={metrics['deletions']}"
                )
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"  Failed to evaluate {split}: {e}")
            results[split] = {"error": str(e)}

    # Print summary and save results only on main process
    if accelerator.is_main_process:
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
