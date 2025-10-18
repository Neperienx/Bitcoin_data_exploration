"""Feature engineering helpers for machine learning experiments."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["return_1m", "ma_ratio", "volatility", "volume_z"]


def make_basic_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Create a small feature set for price movement modelling.

    The returned dataframe keeps the same index/order as the input and includes
    the columns defined in :data:`FEATURE_COLUMNS`.
    """

    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = df.sort_values("timestamp").copy()
    df["return_1m"] = df["close"].pct_change()
    rolling_mean = df["close"].rolling(window=window, min_periods=1).mean()
    df["ma_ratio"] = df["close"] / rolling_mean
    returns = df["close"].pct_change()
    df["volatility"] = returns.rolling(window=window, min_periods=1).std()
    volume_mean = df["volume"].rolling(window=window, min_periods=1).mean()
    volume_std = df["volume"].rolling(window=window, min_periods=1).std(ddof=0)
    df["volume_z"] = (df["volume"] - volume_mean) / volume_std.replace(0, pd.NA)

    return df[FEATURE_COLUMNS]


def make_label(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Generate a binary label representing future price direction."""

    if df.empty:
        return pd.Series(dtype="float64")

    df = df.sort_values("timestamp")
    future_price = df["close"].shift(-horizon)
    label = (future_price > df["close"]).astype("float64")
    return label
