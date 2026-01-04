from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
N_FFT = int(0.025 * SAMPLE_RATE)  # 25ms window = 400 samples
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 10ms hop = 160 samples
N_MELS = 80

SEGMENTED_DIR = Path("Samples/Segmented")
PICTURES_DIR = Path("Pictures")


def push(fifo, value):
    fifo = np.roll(fifo, -1, axis=0)
    fifo[-1] = value
    return fifo


def agc(audio, target_rms_db=-20, noise_floor_db=-50, frame_length=400):
    hop_length = frame_length // 2

    target_rms = 10 ** (target_rms_db / 20)
    noise_floor = 10 ** (noise_floor_db / 20)

    rms_fifo = np.zeros((SAMPLE_RATE // frame_length, 1), dtype=np.float32)

    processed_audio = np.zeros(len(audio), dtype=np.float32)

    for i in range(0, len(audio), hop_length):
        frame = audio[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_fifo = push(rms_fifo, rms)
        if rms < noise_floor:
            gain = 1
        else:
            gain = target_rms / np.mean(rms_fifo, axis=0)
        new_frame = frame * gain
        new_frame = np.tanh(new_frame)
        new_frame = new_frame.astype(np.float32)
        if i + frame_length > len(audio):
            processed_audio[i:] = new_frame[:len(audio) - i]
        else:
            processed_audio[i:i + frame_length] = new_frame

    return processed_audio

def resample_audio_to_16khz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:

    target_sr = 16_000

    y, _sr = librosa.load(str(input_path), sr=target_sr, mono=True)

    sf.write(str(output_path), y, target_sr)



def compute_mel_spectrogram(audio_path, apply_agc=False):
    """Compute Mel Spectrogram for an audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if apply_agc:
        y = agc(y)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def plot_mel_spectrogram(mel_spec_db, title="Mel Spectrogram", ax=None):
    """Plot a Mel Spectrogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='mel',
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        ax=ax
    )
    ax.set_title(title)
    return img


def compare_within_speaker(speaker_name, digits=(1, 2)):
    """Compare spectrograms of different digits from the same speaker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Within Speaker Comparison: {speaker_name}")

    for i, digit in enumerate(digits):
        audio_path = SEGMENTED_DIR / speaker_name / f"segment_{digit:02d}.wav"
        mel_spec = compute_mel_spectrogram(audio_path)
        plot_mel_spectrogram(mel_spec, title=f"Digit {digit}", ax=axes[i])

    plt.tight_layout()
    plt.savefig(PICTURES_DIR / f"within_speaker_{speaker_name}.png", dpi=150)
    plt.show()


def compare_across_speakers(digit, speakers):
    """Compare spectrograms of the same digit across different speakers."""
    n_speakers = len(speakers)
    fig, axes = plt.subplots(1, n_speakers, figsize=(4 * n_speakers, 4))
    fig.suptitle(f"Across Speakers Comparison: Digit {digit}")

    for i, speaker in enumerate(speakers):
        audio_path = SEGMENTED_DIR / speaker / f"segment_{digit:02d}.wav"
        mel_spec = compute_mel_spectrogram(audio_path)
        plot_mel_spectrogram(mel_spec, title=speaker, ax=axes[i])

    plt.tight_layout()
    plt.savefig(PICTURES_DIR / f"across_speakers_digit_{digit}.png", dpi=150)
    plt.show()


def compare_speakers_across_digits(speakers, digits):
    """Compare multiple speakers across multiple digits (grid view)."""
    n_digits = len(digits)
    n_speakers = len(speakers)
    fig, axes = plt.subplots(n_digits, n_speakers, figsize=(4 * n_speakers, 4 * n_digits))
    fig.suptitle(f"Speakers vs Digits Comparison")

    for i, digit in enumerate(digits):
        for j, speaker in enumerate(speakers):
            audio_path = SEGMENTED_DIR / speaker / f"segment_{digit:02d}.wav"
            mel_spec = compute_mel_spectrogram(audio_path)
            ax = axes[i, j] if n_digits > 1 else axes[j]
            plot_mel_spectrogram(mel_spec, title=f"{speaker} - Digit {digit}", ax=ax)

    plt.tight_layout()
    speakers_str = "_".join(speakers)
    digits_str = "_".join(str(d) for d in digits)
    plt.savefig(PICTURES_DIR / f"comparison_{speakers_str}_digits_{digits_str}.png", dpi=150)
    plt.show()


def compute_all_spectrograms():
    """Compute and save all Mel Spectrograms."""
    all_specs = {}

    for speaker_dir in sorted(SEGMENTED_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_name = speaker_dir.name
        all_specs[speaker_name] = {}

        for segment in sorted(speaker_dir.glob("*.wav")):
            segment_idx = int(segment.stem.split("_")[1])
            mel_spec = compute_mel_spectrogram(segment)
            all_specs[speaker_name][segment_idx] = mel_spec

        print(f"Computed {len(all_specs[speaker_name])} spectrograms for {speaker_name}")

    return all_specs


if __name__ == "__main__":
    print(f"Mel Spectrogram Parameters:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Window Size: {N_FFT} samples ({N_FFT/SAMPLE_RATE*1000:.0f}ms)")
    print(f"  Hop Length: {HOP_LENGTH} samples ({HOP_LENGTH/SAMPLE_RATE*1000:.0f}ms)")
    print(f"  Mel Filters: {N_MELS}")
    print()

    # Compute all spectrograms
    all_specs = compute_all_spectrograms()

    # Within speaker comparison (Gal: digit 1 vs digit 2)
    print("\nGenerating within-speaker comparison (Gal: 1 vs 2)...")
    compare_within_speaker("Gal", digits=(1, 2))

    # Across speakers comparison (digit 2: male vs female)
    print("\nGenerating across-speakers comparison (digit 2)...")
    compare_across_speakers(2, ["Gal", "Roy", "Hagar", "Inbar"])

    # 3 speakers (2 males, 1 female) across 3 digits
    print("\nGenerating speakers vs digits comparison (Ofir, Roy, Shir - digits 0, 1, 2)...")
    compare_speakers_across_digits(["Ofir", "Roy", "Shir"], [1, 2, 3])
