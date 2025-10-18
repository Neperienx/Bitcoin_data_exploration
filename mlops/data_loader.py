"""Utilities for loading datasets used in the MLOps workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load a parquet dataset from ``path``.

    Parameters
    ----------
    path:
        Location of the parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical candle schema sorted by timestamp.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    df = pd.read_parquet(file_path)
    if "timestamp" not in df.columns:
        raise ValueError("Loaded dataframe is missing required 'timestamp' column")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df
