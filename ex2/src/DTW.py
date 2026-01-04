import numpy as np
import matplotlib.pyplot as plt
from mel_spectrogram import compute_mel_spectrogram


def dtw(m1, m2):

    distances = np.zeros((m1.shape[1], m2.shape[1]))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i][j] = np.sqrt(np.sum((m1[:,i] - m2[:,j])**2))

    dtw = np.zeros_like(distances)
    # dtw_backtracking = np.zeros_like(distances)

    dtw[0,0] = distances[0,0]
    for i in range(1,distances.shape[0]):
        dtw[i,0] = dtw[i-1,0] + distances[i,0]
        # dtw_backtracking[i,0] = 2
    for j in range(1,distances.shape[1]):
        dtw[0,j] = dtw[0,j-1] + distances[0,j]
        # dtw_backtracking[0,j] = 1

    for i in range(1, dtw.shape[0]):
        for j in range(1, dtw.shape[1]):
            a = dtw[i-1,j-1]
            b = dtw[i,j-1]
            c = dtw[i-1,j]
            # dtw_backtracking[i,j] = np.argmin(np.array([a, b, c]))
            dtw[i,j] = min(a, b, c) + distances[i,j]

    return dtw[dtw.shape[0] - 1][dtw.shape[1] - 1]

def calc_distance_matrix():
    reference_dir = "Samples/Segmented/Ido/"
    validation_dirs = ["Samples/Segmented/Adam/",
                       "Samples/Segmented/Roy/",
                       "Samples/Segmented/Hagar/",
                       "Samples/Segmented/Inbar/"]
    distances = np.zeros((4,10,11))
    for speaker_index in range(4):
        for reference_digit in range(10):
            reference_digit_path = reference_dir + f"segment_0{reference_digit}.wav"
            refence_mel = compute_mel_spectrogram(reference_digit_path)
            for evaluated_digit in range(10):
                evaluated_digit_path = validation_dirs[speaker_index] + f"segment_0{evaluated_digit}.wav"
                evaluated_mel = compute_mel_spectrogram(evaluated_digit_path)
                current_dtw = dtw(refence_mel, evaluated_mel)
                distances[speaker_index,reference_digit,evaluated_digit] = current_dtw

            evaluated_gorilla_path = reference_dir + f"segment_10.wav"
            evaluated_mel = compute_mel_spectrogram(evaluated_gorilla_path)
            current_dtw = dtw(refence_mel, evaluated_mel)
            distances[speaker_index,reference_digit,10] = current_dtw
    return distances

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_4_matrices_heatmaps_with_col_argmin(arr: np.ndarray, titles=None, marker="x") -> None:
    """
    Plot 4 heatmaps from an array of shape (4, 10, 11) and mark, for each column,
    the row index of the argmin value.

    Marks are drawn on top of each heatmap at (col, argmin_row).

    Parameters
    ----------
    arr : np.ndarray
        Shape must be (4, 10, 11).
    titles : list[str] | None
        Optional list of 4 titles.
    marker : str
        Matplotlib marker style for argmin points (default: "x").
    """
    arr = np.asarray(arr)
    if arr.shape != (4, 10, 11):
        raise ValueError(f"Expected shape (4, 10, 11), got {arr.shape}")

    if titles is None:
        titles = [f"Matrix {i}" for i in range(4)]
    if len(titles) != 4:
        raise ValueError("titles must be a list of length 4")

    n_rows, n_cols = arr.shape[1], arr.shape[2]
    cols = np.arange(n_cols)

    for i in range(4):
        mat = arr[i]

        # Argmin row per column (shape: (n_cols,))
        argmin_rows = np.argmin(mat, axis=0)

        plt.figure()
        im = plt.imshow(mat, aspect="auto")  # default colormap
        plt.title(titles[i])
        plt.xlabel("Column index")
        plt.ylabel("Row index")

        # Overlay argmin markers: x = column, y = argmin row
        plt.scatter(cols, argmin_rows, marker=marker, s=60)

        plt.colorbar(im, label="Value")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    distances = calc_distance_matrix()
    plot_4_matrices_heatmaps_with_col_argmin(distances)