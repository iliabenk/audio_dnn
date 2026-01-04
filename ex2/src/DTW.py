import matplotlib.pyplot as plt
import numpy as np

from mel_spectrogram import compute_mel_spectrogram

REFERENCE_DIR = "Samples/Segmented/Gal/"
VALIDATION_DIR = ["Samples/Segmented/Nirit/","Samples/Segmented/Ofir/",
                       "Samples/Segmented/Roy/", "Samples/Segmented/Shir/"]

def dtw(m1, m2, normalize=False):
    n, m = m1.shape[1], m2.shape[1]
    distances = np.zeros((m1.shape[1], m2.shape[1]))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i][j] = np.sqrt(np.sum((m1[:,i] - m2[:,j])**2))

    dtw_matrix = np.zeros_like(distances)

    dtw_matrix[0,0] = distances[0,0]
    for i in range(1,distances.shape[0]):
        dtw_matrix[i,0] = dtw_matrix[i-1,0] + distances[i,0]

    for j in range(1,distances.shape[1]):
        dtw_matrix[0,j] = dtw_matrix[0,j-1] + distances[0,j]

    for i in range(1, dtw_matrix.shape[0]):
        for j in range(1, dtw_matrix.shape[1]):
            dtw_matrix[i,j] = min(dtw_matrix[i-1,j-1], dtw_matrix[i,j-1], dtw_matrix[i-1,j]) + distances[i,j]

    cost = dtw_matrix[n - 1][m - 1]
    if normalize:
        cost = cost / (n + m)
    return cost

def calc_distance_matrix(dir_list):
    distances = np.zeros((len(dir_list),10,11))
    for speaker_index in range(len(dir_list)):
        for reference_digit in range(10):
            reference_digit_path = REFERENCE_DIR + f"segment_0{reference_digit}.wav"
            refence_mel = compute_mel_spectrogram(reference_digit_path)

            for evaluated_digit in range(10):
                evaluated_digit_path = dir_list[speaker_index] + f"segment_0{evaluated_digit}.wav"
                evaluated_mel = compute_mel_spectrogram(evaluated_digit_path)
                current_dtw = dtw(refence_mel, evaluated_mel)
                distances[speaker_index,reference_digit,evaluated_digit] = current_dtw

            evaluated_gorilla_path = dir_list[speaker_index] + f"segment_10.wav"
            evaluated_mel = compute_mel_spectrogram(evaluated_gorilla_path)
            current_dtw = dtw(refence_mel, evaluated_mel)
            distances[speaker_index,reference_digit,10] = current_dtw

    return distances


def plot_distance_matrices(arr: np.ndarray, titles, thresh=6776) -> None:
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

def calculate_confusion_matrix(validation_dirs, thresh=6776):
    confusion_distances = calc_distance_matrix(validation_dirs)
    confusion_matrix = np.zeros((11,11))
    for i in range(confusion_distances.shape[0]):
        for j in range(11):
            scores = confusion_distances[i,:,j]
            result = np.argmin(scores)
            if scores[result]<thresh:
                confusion_matrix[j,result] += 1
            else:
                confusion_matrix[j,10] += 1
    return confusion_matrix


def calculate_accuracy_from_confusion_matrix(confusion_matrix):
    total_samples = np.sum(confusion_matrix)
    correct = np.trace(confusion_matrix)
    overall_accuracy = correct / total_samples

    digit_correct = np.trace(confusion_matrix[:10, :10])
    digit_total = np.sum(confusion_matrix[:10, :])
    digit_accuracy = digit_correct / digit_total

    noise_correct = confusion_matrix[10, 10]
    noise_total = np.sum(confusion_matrix[10, :])
    noise_accuracy = noise_correct / noise_total if noise_total > 0 else 0

    print(f"Digit accuracy: {digit_correct:.0f}/{digit_total:.0f} ({digit_accuracy:.1%})")
    print(f"Noise accuracy: {noise_correct:.0f}/{noise_total:.0f} ({noise_accuracy:.1%})")
    print(f"Overall accuracy: {correct:.0f}/{total_samples:.0f} ({overall_accuracy:.1%})")

    return overall_accuracy

def plot_confusion_matrix_from_threshold(val_dir, thresh=6776):
    conf_mat = calculate_confusion_matrix(val_dir, thresh)
    calculate_accuracy_from_confusion_matrix(conf_mat)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(conf_mat, aspect="auto")
    plt.title(f"Confusion Matrix (thresh={thresh})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(im)

    # annotate cells
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            txt = f"{int(conf_mat[i, j])}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_dir = ["Samples/Segmented/Adam/",
     "Samples/Segmented/Ido/",
     "Samples/Segmented/Hagar/",
     "Samples/Segmented/Inbar/"]
    distances = calc_distance_matrix(train_dir)
    plot_distance_matrices(distances,
                           ["Gal Vs Adam", "Gal Vs Ido", "Gal Vs Hagar", "Gal Vs Inbar"]
)
    plot_confusion_matrix_from_threshold(VALIDATION_DIR, thresh=6776)