"""Utility script to split the consolidated BTC/USDT dataset into
training and test subsets suitable for machine learning workflows.

The script reads ``data/btc_usdt_1m_all.parquet`` and produces two
Parquet files under ``data/ml`` by default:

* ``train.parquet`` – the earliest portion of the series
* ``test.parquet`` – the most recent portion

The split is chronological to avoid leaking future information into the
training set.  You can customise the train/test ratio or the output
location using command-line flags; run ``python split_dataset.py --help``
for full usage information.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("data/btc_usdt_1m_all.parquet")
DEFAULT_OUTPUT_DIR = Path("data/ml")


def split_dataset(
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.8,
) -> tuple[Path, Path]:
    """Split the consolidated dataset into chronological train/test sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

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

    split_index = int(len(df) * train_ratio)
    if split_index == 0 or split_index == len(df):
        raise ValueError(
            "train_ratio results in an empty train or test set; adjust the ratio or gather more data."
        )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

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
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of rows to allocate to the training set (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, test_path = split_dataset(
        source=args.source,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )
    print(f"Training data written to: {train_path}")
    print(f"Test data written to: {test_path}")


if __name__ == "__main__":
    main()
