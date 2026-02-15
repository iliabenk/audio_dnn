#!/usr/bin/env python3
"""
HuBERT ASR Transcription Script

Transcribe audio files using a fine-tuned HuBERT model.

Usage:
    python -m src.transcribe --model outputs/hubert-finetuned/final --audio sample.wav
    python -m src.transcribe --model outputs/final --audio file1.wav file2.wav file3.wav
"""

import argparse
import logging
from pathlib import Path

import librosa
import torch
from transformers import HubertForCTC, Wav2Vec2Processor

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
        description="Transcribe audio files with HuBERT ASR model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to audio file(s) to transcribe",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)",
    )
    return parser.parse_args()


def transcribe_audio(
    model,
    processor,
    audio_path: str,
    sample_rate: int,
    device: torch.device,
) -> str:
    """Transcribe a single audio file.

    Args:
        model: HuBERT model.
        processor: Wav2Vec2 processor.
        audio_path: Path to audio file.
        sample_rate: Target sample rate.
        device: Compute device.

    Returns:
        Transcribed text.
    """
    # Load and resample audio
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Process audio
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    input_values = inputs.input_values.to(device)

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription


def main():
    """Main transcription function."""
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

    # Transcribe each audio file
    print("\n" + "=" * 60)
    print("TRANSCRIPTIONS")
    print("=" * 60)

    for audio_path in args.audio:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            print(f"ERROR: File not found: {audio_path}")
            continue

        try:
            transcription = transcribe_audio(
                model=model,
                processor=processor,
                audio_path=str(audio_path),
                sample_rate=args.sample_rate,
                device=device,
            )
            print(f"\n{audio_path.name}:")
            print(f"  {transcription}")
        except Exception as e:
            print(f"\n{audio_path.name}:")
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
