"""Utility script to split the consolidated BTC/USDT dataset into
training and test subsets suitable for machine learning workflows.

The script reads ``data/btc_usdt_1m_all.parquet`` and produces two
Parquet files under ``data/ml`` by default:

* ``train.parquet`` – the earliest portion of the series
* ``test.parquet`` – the most recent portion

The split is chronological to avoid leaking future information into the
training set.  The training partition contains all but the final year of
records, while the test partition captures the most recent 12 months.
Run ``python split_dataset.py --help`` for usage information.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("data/btc_usdt_1m_all.parquet")
DEFAULT_OUTPUT_DIR = Path("data/ml")


def _parse_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Return the timestamp column as timezone-aware datetimes."""

    ts = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        return pd.to_datetime(ts, utc=True)

    # Try millisecond resolution first (Binance default), fall back to raw parsing.
    parsed_ms = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    if parsed_ms.notna().all():
        return parsed_ms

    parsed_generic = pd.to_datetime(ts, utc=True, errors="coerce")
    if parsed_generic.notna().all():
        return parsed_generic

    raise ValueError(
        "Unable to convert 'timestamp' column to datetimes; "
        "ensure it contains integer milliseconds or ISO8601 strings."
    )


def split_dataset(
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Split the dataset into training data and the most recent year for testing."""

    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {source}. Run data_extraction or the Flask app first."
        )

    df = pd.read_parquet(source)
    if df.empty:
        raise ValueError("Source dataset is empty; nothing to split.")

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column for chronological sorting.")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    timestamp_dt = _parse_timestamp_column(df)
    most_recent = timestamp_dt.iloc[-1]
    cutoff = most_recent - pd.DateOffset(years=1)

    if cutoff <= timestamp_dt.iloc[0]:
        raise ValueError(
            "Dataset does not cover a full year; unable to create a one-year test set."
        )

    train_mask = timestamp_dt < cutoff
    test_mask = ~train_mask

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Train/test split failed; verify the dataset timestamps are diverse.")

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return train_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the consolidated Parquet dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the train/test Parquet files will be written (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, test_path = split_dataset(
        source=args.source,
        output_dir=args.output_dir,
    )
    print(f"Training data written to: {train_path}")
    print(f"Test data written to: {test_path}")


if __name__ == "__main__":
    main()
