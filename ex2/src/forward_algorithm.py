import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List, Union

def ctc_collapse_b(seq: Union[str, List[str]], blank_char="_") -> str:
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
    S = [BLANK_IDX] + [inverse_mapping[t] for t in target] + [BLANK_IDX]
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
    output_path = "./outputs/pred_matrix.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    return float(total_prob)


def test_ctc_collapse_b():
    assert ctc_collapse_b("__H___EEEE_L_LLLL_OO___") == "HELLO", "ctc_collapse_b test FAILED"
    assert ctc_collapse_b("H_E_LL_LL_OO") == "HELLO", "ctc_collapse_b test FAILED"
    assert ctc_collapse_b("H__EEEE_LLLLLL_LLLL_O__") == "HELLO", "ctc_collapse_b test FAILED"

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

if __name__ == "__main__":
    test_ctc_collapse_b()
    test_forward()
