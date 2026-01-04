import librosa
import soundfile as sf
from pathlib import Path


def segment_audio(audio_path, output_dir, top_db=25, min_silence_gap=0.3, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)

    merged = []
    for start, end in intervals:
        if merged and (start - merged[-1][1]) / sr < min_silence_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    padding = int(0.05 * sr)
    speaker = Path(audio_path).stem
    speaker_dir = Path(output_dir) / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, end) in enumerate(merged):
        seg = y[max(0, start - padding):min(len(y), end + padding)]
        if sr != target_sr:
            seg = librosa.resample(seg, orig_sr=sr, target_sr=target_sr)
        sf.write(speaker_dir / f'segment_{i:02d}.wav', seg, target_sr)
        print(f"    {i}: {start/sr:.2f}s - {end/sr:.2f}s ({(end-start)/sr:.2f}s)")

    return len(merged)


def segment_audio_filtered(audio_path, output_dir, top_db=25, min_silence_gap=0.3,
                           min_duration=0.2, target_sr=16000):
    """Same as segment_audio but filters out short noise segments."""
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)

    merged = []
    for start, end in intervals:
        if merged and (start - merged[-1][1]) / sr < min_silence_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Filter out noise (very short segments)
    merged = [(s, e) for s, e in merged if (e - s) / sr > min_duration]

    padding = int(0.05 * sr)
    speaker = Path(audio_path).stem
    speaker_dir = Path(output_dir) / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    for f in speaker_dir.glob('*.wav'):
        f.unlink()

    for i, (start, end) in enumerate(merged):
        seg = y[max(0, start - padding):min(len(y), end + padding)]
        if sr != target_sr:
            seg = librosa.resample(seg, orig_sr=sr, target_sr=target_sr)
        sf.write(speaker_dir / f'segment_{i:02d}.wav', seg, target_sr)
        print(f"    {i}: {start/sr:.2f}s - {end/sr:.2f}s ({(end-start)/sr:.2f}s)")

    return len(merged)


if __name__ == "__main__":
    print('Re-processing Ido with smaller gap and noise filtering...')
    n = segment_audio_filtered('Samples/Raw/Ido.wav', 'Samples/Segmented',
                               top_db=25, min_silence_gap=0.1, min_duration=0.2)
    print(f'  Total: {n} segments')
