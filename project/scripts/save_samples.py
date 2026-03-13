"""Save audio samples from LibriSpeech for project submission."""

import os
import soundfile as sf
from datasets import load_dataset


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
NUM_SAMPLES = 3


def save_split(dataset, split_name, output_dir, info_lines):
    os.makedirs(output_dir, exist_ok=True)
    for i, example in enumerate(dataset.select(range(NUM_SAMPLES))):
        audio = example["audio"]
        filename = f"{split_name}_sample_{i+1}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio["array"], audio["sampling_rate"])
        info_lines.append(f"{filename}: {example['text']}")
        print(f"  Saved {filepath}")


def main():
    info_lines = []

    print("Loading train.clean.100...")
    train_ds = load_dataset("librispeech_asr", "clean", split="train.100")
    train_dir = os.path.join(SAMPLES_DIR, "train")
    save_split(train_ds, "train", train_dir, info_lines)

    print("Loading validation.clean...")
    val_ds = load_dataset("librispeech_asr", "clean", split="validation")
    val_dir = os.path.join(SAMPLES_DIR, "validation")
    save_split(val_ds, "validation", val_dir, info_lines)

    info_path = os.path.join(SAMPLES_DIR, "samples_info.txt")
    with open(info_path, "w") as f:
        f.write("Audio Samples from LibriSpeech\n")
        f.write("=" * 40 + "\n\n")
        for line in info_lines:
            f.write(line + "\n")
    print(f"\nSaved sample info to {info_path}")


if __name__ == "__main__":
    main()
