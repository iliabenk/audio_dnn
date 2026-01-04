from __future__ import annotations

from pathlib import Path
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import soundfile as sf


def resample_audio_to_16khz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:

    target_sr = 16_000

    y, _sr = librosa.load(str(input_path), sr=target_sr, mono=True)

    sf.write(str(output_path), y, target_sr)

import os
from typing import Iterable, Optional


def batch_resample_dir_to_16khz(
    input_dir: str,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        print(fname)
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue

        base, ext = os.path.splitext(fname)
        if ext.lower() != ".wav":
            continue

        out_name = f"{base}_16khz.wav"
        out_path = os.path.join(output_dir, out_name)

        # Uses your previously defined function:
        resample_audio_to_16khz(in_path, out_path)


def mel_spectogram(audio_path: str,
                   sr: int = 16000,
                   n_mels: int = 80,
                   to_db: bool = False,
                   window_size: float = 0.025,
                   hop: float = 0.01) -> Tuple[np.ndarray, int]:

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    win_length = int(round(window_size * sr))
    hop_length = int(round(hop * sr))
    n_fft = win_length

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        n_mels=n_mels,
        power=2.0,
        center=True,
    )

    if to_db:
        S = librosa.power_to_db(S, ref=np.max)

    return S, sr


if __name__ == "__main__":
    batch_resample_dir_to_16khz("Samples/Raw/","Samples/Resampled")