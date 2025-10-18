"""Dataset splitting strategies tailored for time series."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float | int = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split time-ordered data into train/test partitions.

    Parameters
    ----------
    X, y:
        Feature matrix and target series sorted chronologically.
    test_size:
        Fraction (0 < test_size < 1) or absolute count used for the test split.
    """

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if len(X) == 0:
        return X.copy(), X.copy(), y.copy(), y.copy()

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size fraction must be between 0 and 1")
        test_count = max(1, int(round(len(X) * test_size)))
    else:
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        test_count = min(len(X), int(test_size))

    split_idx = len(X) - test_count
    if split_idx <= 0:
        split_idx = len(X) // 2 or 1

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test
