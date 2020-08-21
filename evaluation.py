import numpy as np
import pandas as pd


def validate(l_predicted, l_expected):
    num_pos = 0
    num_neg = 0

    for predicted, expected in zip(l_predicted, l_expected):
        # Token based computation using token ids from language models

        if predicted.shape != expected.shape or predicted.size == 0:
            raise ValueError(f"Incorrect input arrays {predicted} and {expected}")

        comparison = (predicted == expected)

        pos = np.count_nonzero(comparison)
        neg = len(comparison) - pos

        num_pos += pos
        num_neg += neg

    return num_pos / (num_pos + num_neg)
