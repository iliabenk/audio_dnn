import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import os


def segment_audio(audio_path, output_dir, top_db=25, min_silence_gap=0.3, target_sr=16000):
    """
    Segment audio file into individual words.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save segments
        top_db: Threshold for silence detection (higher = more sensitive)
        min_silence_gap: Minimum gap between segments to consider them separate words (seconds)
        target_sr: Target sample rate for output files

    Returns:
        List of segment info dictionaries
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)

    # Merge segments that are too close together
    merged_intervals = []
    for start, end in intervals:
        if merged_intervals and (start - merged_intervals[-1][1]) / sr < min_silence_gap:
            # Merge with previous segment
            merged_intervals[-1] = (merged_intervals[-1][0], end)
        else:
            merged_intervals.append((start, end))

    # Add padding around each segment
    padding_samples = int(0.05 * sr)  # 50ms padding

    segments = []
    speaker_name = Path(audio_path).stem

    # Create output directory
    speaker_dir = Path(output_dir) / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, end) in enumerate(merged_intervals):
        # Add padding
        padded_start = max(0, start - padding_samples)
        padded_end = min(len(y), end + padding_samples)

        # Extract segment
        segment = y[padded_start:padded_end]

        # Resample to target sample rate
        if sr != target_sr:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)

        # Save segment
        output_path = speaker_dir / f"segment_{i:02d}.wav"
        sf.write(output_path, segment, target_sr)

        segments.append({
            'index': i,
            'start': start / sr,
            'end': end / sr,
            'duration': (end - start) / sr,
            'path': str(output_path)
        })

    return segments


def process_all_speakers(raw_dir, output_dir, **kwargs):
    """Process all audio files in raw directory."""
    raw_path = Path(raw_dir)
    results = {}

    for audio_file in sorted(raw_path.glob("*.wav")):
        print(f"\nProcessing {audio_file.name}...")
        segments = segment_audio(audio_file, output_dir, **kwargs)
        results[audio_file.stem] = segments

        print(f"  Found {len(segments)} segments:")
        for seg in segments:
            print(f"    {seg['index']:2d}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")

        if len(segments) != 11:
            print(f"  WARNING: Expected 11 segments, got {len(segments)}")

    return results


if __name__ == "__main__":
    raw_dir = "Samples/Raw"
    output_dir = "Samples/Segmented"

    print("=" * 60)
    print("Automatic Audio Segmentation")
    print("=" * 60)
    print("\nExpected: 11 segments per speaker (digits 0-9 + random word)")

    results = process_all_speakers(raw_dir, output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for speaker, segments in results.items():
        status = "OK" if len(segments) == 11 else f"NEEDS REVIEW ({len(segments)} segments)"
        print(f"  {speaker}: {status}")

    print("\nSegments saved to:", output_dir)
    print("\nNext steps:")
    print("  1. Review segments in Samples/Segmented/<speaker>/")
    print("  2. Rename files to: 0.wav, 1.wav, ..., 9.wav, random.wav")
    print("  3. Delete any extra segments or re-segment problematic files")
