import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

def ctc_collapse_b(seq: Union[str, List[str]], blank_char="^") -> str:
    """
    Collapses the raw output of ctc into the final word. For example, __H_EE_LL_LL_OOO__ will collapse to HELLO
    """

    # Loops over all chars, starting the second char to check if it matches the first
    assert len(seq) > 0, f"Dude WTH, are you trying to trick me?! That's an empty sequence"

    # If there is only 1 char, return it as is, unless that's a blank char and then return an empty string
    if len(seq) == 1:
        return seq[0] if seq[0] != blank_char else ""

    # Empty list to contain the sequence without adjacent duplications, but still with blanks
    no_dups_seq = []

    # Put the first char in the new list
    no_dups_seq.append(seq[0])

    # Loops over all chars and remove adjacent dups
    for char in seq[1:]:
        # If the char is the same as the last one added to the list, continue
        if char == no_dups_seq[-1]:
            continue

        # Found a non-duplicated char, add it to the list
        no_dups_seq.append(char)

    # Create the final word, without the blank char
    output = "".join([c for c in no_dups_seq if c != blank_char])

    return output


def forward(pred: np.ndarray, target: str) -> float:
    # Set the mapping
    mapping = {0: 'a', 1: 'b', 2: '^'}
    inverse_mapping = {v: k for k, v in mapping.items()}
    BLANK_IDX = 2

    # Put a blank between every char, start & end.
    S = [BLANK_IDX]
    for char in target:
        S.append(inverse_mapping[char])
        S.append(BLANK_IDX)

    # Init variables
    T = pred.shape[0] # The amount of time steps
    L = len(S) # The length of the target sequence (with blanks)
    alpha = np.zeros((T, L)) # Initialize alpha

    # Initialize alpha at the 0 with the actual 'pred' values
    alpha[0, 0] = pred[0, S[0]] # probability to start with a blank value
    alpha[0, 1] = pred[0, S[1]] # probability to start at the actual first char

    # Dynamic Programing loop
    for t in range(1, T):
        for s in range(L):
            # Probability to come from the exact same char as in the previous step
            prev_sum = alpha[t-1, s]

            if s > 0:
                # Probability to come from the previous char
                prev_sum += alpha[t-1, s-1]

            # Probability to jump directly from 2 chars before. This is only possible if this one is not blank, or the char at 2 positions before is not the same as the current one
            if s >= 2 and S[s] != BLANK_IDX and S[s] != S[s-2]:
                prev_sum += alpha[t-1, s-2]

            # Update the probability to reach this position at time t. It's the total sum of paths to reach this point, multiplied by the probability to get to this point at this time
            alpha[t, s] = prev_sum * pred[t, S[s]]

    # We sum both because both landing at the end on the last target char or blank is valid
    total_prob = alpha[T-1, L-1] + alpha[T-1, L-2]

    # Plotting pred matrix
    plt.figure(figsize=(8, 4))
    sns.heatmap(pred.T, annot=True, cmap="jet", fmt=".2f",
                xticklabels=range(pred.shape[0]),
                yticklabels=[mapping[i] for i in range(pred.shape[1])])
    plt.title("Prediction Matrix (pred)")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Characters")
    plt.tight_layout()
    output_path = "./outputs/ex5_pred_matrix.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    return float(total_prob)


def force_alignment(pred: np.ndarray, target: str, file_prefix: str,
                    mapping: Optional[Dict[int, str]] = None,
                    blank_symbol: str = "^",
                    annot=True,
                    fig_size='small') -> float:

    # Setup Mapping
    mapping = mapping or {0: 'a', 1: 'b', 2: '^'}
    inv_mapping = {v: k for k, v in mapping.items()}
    BLANK_IDX = inv_mapping[blank_symbol]

    # Expand Target Sequence (with blanks: ^ a ^ b ^ a ^)
    S = [BLANK_IDX]
    for char in target:
        S.append(inv_mapping[char])
        S.append(BLANK_IDX)

    T = pred.shape[0]  # Time steps (5)
    L = len(S)  # Expanded length (7 for 'aba')

    # Initialize
    alpha = np.zeros((T, L))
    backtrace = np.zeros((T, L), dtype=int)

    # Initial step (t=0)
    alpha[0, 0] = pred[0, S[0]]  # start with blank
    alpha[0, 1] = pred[0, S[1]]  # start with first char ('a')

    #  Forward Pass (max instead of sum)
    for t in range(1, T):
        for s in range(L):
            # Option 1: Stay in current state s
            max_val = alpha[t - 1, s]
            prev_s = s

            # Option 2: Move from previous state s-1
            if s > 0 and alpha[t - 1, s - 1] > max_val:
                max_val = alpha[t - 1, s - 1]
                prev_s = s - 1

            # Option 3: Skip blank (move from s-2)
            # Conditions: s >= 2, current is not blank, and current char != char 2 steps ago
            if s >= 2 and S[s] != BLANK_IDX and S[s] != S[s - 2]:
                if alpha[t - 1, s - 2] > max_val:
                    max_val = alpha[t - 1, s - 2]
                    prev_s = s - 2

            alpha[t, s] = max_val * pred[t, S[s]]
            backtrace[t, s] = prev_s

    # Backtrace the optimal path
    # Final state can be the last blank or the last character
    best_final_s = L - 1 if alpha[T - 1, L - 1] > alpha[T - 1, L - 2] else L - 2
    total_prob = alpha[T - 1, best_final_s]

    path_s = [best_final_s]
    for t in range(T - 1, 0, -1):
        path_s.append(backtrace[t, path_s[-1]])
    path_s.reverse()

    # Convert path indices to characters
    path_chars = "".join([mapping[S[s]] for s in path_s])

    # Convert path indices to the original 3-label mapping (0, 1, 2) for Plot D
    original_label_path = [S[s] for s in path_s]

    print(f"Path for '{target}': {path_chars}")
    print(f"Probability: {total_prob:.6f}")

    # Plot 6.d: Pred Matrix with Aligned Sequence
    if fig_size == "small":
        plt.figure(figsize=(8, 5))
    else:
        plt.figure(figsize=(20, 6))

    sns.heatmap(pred.T, annot=annot, fmt=".2f", cmap="YlGnBu",
                yticklabels=[mapping[i] for i in range(len(mapping))])

    # Plot the forced alignment path
    plt.step(np.arange(T) + 0.5, np.array(original_label_path) + 0.5,
             where='mid', color='red', linewidth=3, label='Forced Alignment')

    plt.title(f"6.d: Pred Matrix & Forced Alignment ('{target}')")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Output Labels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{file_prefix}_d_alignment.png")

    # Plot 6.e: Backtrace Matrix (Trellis)
    if fig_size == "small":
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(20, 20))

    # Use log scale for heatmap colors to make smaller probabilities visible
    log_alpha = np.log(alpha + 1e-12)

    sns.heatmap(log_alpha.T, annot=alpha.T if annot else False, fmt=".4f", cmap="magma",
                yticklabels=[f"S{s}: {mapping[S[s]]}" for s in range(L)])

    # Overlay the path through the trellis
    plt.plot(np.arange(T) + 0.5, np.array(path_s) + 0.5,
             color='cyan', marker='o', markersize=10, linewidth=2, label='Selected Path')

    plt.title(f"6.e: Trellis (Alpha) and Backtrace Path")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Expanded Sequence States (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{file_prefix}_e_backtrace.png")

    return total_prob


def test_ctc_collapse_b():
    assert ctc_collapse_b("^^H^^^EEEE^L^LLLL^OO^^^") == "HELLO", "ctc_collapse_b test FAILED"
    assert ctc_collapse_b("H^E^LL^LL^OO") == "HELLO", "ctc_collapse_b test FAILED"
    assert ctc_collapse_b("H^^EEEE^LLLLLL^LLLL^O^^") == "HELLO", "ctc_collapse_b test FAILED"

    print("ctc_collapse_b test PASSED")


def test_forward():
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00

    print(f"""Probability to get 'aba' = {forward(pred, target="aba"):.4f}""")


def test_force_alignment():
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00

    print(f"""Probability to get 'aba' with forced alignment (max instead of sum) = {force_alignment(pred, target="aba", file_prefix="ex6"):.4f}""")


def test_force_align_pkl_data():
    """
    Note when writing the summary: The output is the same as greedy decoding, but the probability is different. Need ot think why.
    Also need to think how to fix the plots

    """
    data = pkl.load(open("./force_align.pkl", 'rb'))

    print(f"""Probability to get '{data["gt_text"]}' with forced alignment (max instead of sum) = {force_alignment(pred=data["acoustic_model_out_probs"], 
                                                                                                     target=data["gt_text"],
                                                                                                     mapping=data["label_mapping"],
                                                                                                     file_prefix="ex7",
                                                                                                     annot=False,
                                                                                                     fig_size='large'):.8f}""")

if __name__ == "__main__":
    test_ctc_collapse_b()
    test_forward()
    test_force_alignment()
    test_force_align_pkl_data()