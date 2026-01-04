import matplotlib.pyplot as plt
import numpy as np

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
    reference_dir = "Samples/Segmented/Gal/"
    train_dirs = ["Samples/Segmented/Adam/",
                       "Samples/Segmented/Ido/",
                       "Samples/Segmented/Hagar/",
                       "Samples/Segmented/Inbar/"]
    distances = np.zeros((4,10,11))
    for speaker_index in range(len(train_dirs)):
        for reference_digit in range(10):
            reference_digit_path = reference_dir + f"segment_0{reference_digit}.wav"
            refence_mel = compute_mel_spectrogram(reference_digit_path)

            for evaluated_digit in range(10):
                evaluated_digit_path = train_dirs[speaker_index] + f"segment_0{evaluated_digit}.wav"
                evaluated_mel = compute_mel_spectrogram(evaluated_digit_path)
                current_dtw = dtw(refence_mel, evaluated_mel)
                distances[speaker_index,reference_digit,evaluated_digit] = current_dtw

            evaluated_gorilla_path = train_dirs[speaker_index] + f"segment_10.wav"
            evaluated_mel = compute_mel_spectrogram(evaluated_gorilla_path)
            current_dtw = dtw(refence_mel, evaluated_mel)
            distances[speaker_index,reference_digit,10] = current_dtw

    return distances


def calculate_accuracy(distances, threshold=6976):
    speakers = ['Adam', 'Ido', 'Hagar', 'Inbar']

    correct_digits = 0
    rejected_as_noise = 0
    misclassified = 0
    total_digits = 40

    correct_noise = 0
    total_noise = 4

    for s in range(4):
        for true_digit in range(10):
            dists = distances[s, :, true_digit]
            predicted = np.argmin(dists)
            min_dist = dists[predicted]

            if min_dist >= threshold:
                rejected_as_noise += 1
            elif predicted == true_digit:
                correct_digits += 1
            else:
                misclassified += 1

        noise_dists = distances[s, :, 10]
        if np.min(noise_dists) >= threshold:
            correct_noise += 1

    digit_accuracy = correct_digits / total_digits
    noise_accuracy = correct_noise / total_noise
    overall_accuracy = (correct_digits + correct_noise) / (total_digits + total_noise)

    print(f"Threshold: {threshold}")
    print(f"Digit classification: {correct_digits}/{total_digits} ({digit_accuracy:.1%})")
    print(f"  Rejected as noise: {rejected_as_noise}")
    print(f"  Misclassified: {misclassified}")
    print(f"Noise rejection: {correct_noise}/{total_noise} ({noise_accuracy:.1%})")
    print(f"Overall accuracy: {correct_digits + correct_noise}/{total_digits + total_noise} ({overall_accuracy:.1%})")

    return overall_accuracy


def plot_4_matrices_heatmaps_with_col_argmin(arr: np.ndarray,thresh=6976) -> None:
    titles = ["Gal Vs Adam", "Gal Vs Ido", "Gal Vs Hagar", "Gal Vs Inbar"]
    cols = np.arange(arr.shape[2])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    if thresh is not None:
        arr = arr < thresh

    for i, ax in enumerate(axes):
        mat = arr[i]
        argmin_rows = np.argmin(mat, axis=0)

        im = ax.imshow(mat, aspect="auto")
        ax.set_title(titles[i])
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")

        ax.scatter(cols, argmin_rows, color="red", marker="x", s=80, linewidths=2)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")

    plt.show()




if __name__ == "__main__":
    distances = calc_distance_matrix()
    calculate_accuracy(distances, threshold=6976)
    plot_4_matrices_heatmaps_with_col_argmin(distances)