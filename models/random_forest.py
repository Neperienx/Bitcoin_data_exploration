"""Random forest training helpers and CLI entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.data_loader import load_parquet
from mlops.evaluation import classification_metrics
from mlops.features import FEATURE_COLUMNS, make_basic_features, make_label
from mlops.split import time_train_test_split

FEATURE_ORDER = FEATURE_COLUMNS


def _ensure_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("Expected a pandas DataFrame with named feature columns")


def _ensure_series(y: Any) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = 6,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a random forest classifier on the provided data."""

    X_df = _ensure_dataframe(X_train).loc[:, FEATURE_ORDER]
    y_series = _ensure_series(y_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model.fit(X_df, y_series)
    return model


def predict_rf(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Generate binary predictions using the trained random forest."""

    X_df = _ensure_dataframe(X).loc[:, FEATURE_ORDER]
    predictions = model.predict(X_df)
    return predictions.astype(np.int64)


def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    """Persist the trained model to ``path`` using joblib."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)


def load_model(path: str | Path) -> RandomForestClassifier:
    """Load a serialized random forest model from ``path``."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    model = joblib.load(file_path)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Loaded object is not a RandomForestClassifier")
    return model


def _compute_strategy_curve(
    dataset: pd.DataFrame,
    predictions: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """Compute strategy and buy-and-hold cumulative returns for predictions."""

    future_close = dataset["close"].shift(-horizon)
    future_return = (future_close - dataset["close"]) / dataset["close"]

    frame = dataset.copy()
    frame["future_return"] = future_return
    frame = frame.dropna(subset=["future_return"])  # drop rows without horizon data

    aligned_predictions = predictions.reindex(frame.index).fillna(0)
    frame["strategy_return"] = frame["future_return"] * aligned_predictions
    frame["strategy_curve"] = (1 + frame["strategy_return"]).cumprod()
    frame["buy_hold_curve"] = (1 + frame["future_return"]).cumprod()
    return frame[["strategy_curve", "buy_hold_curve", "future_return", "strategy_return"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a random forest model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/btc_usdt_1m_all.parquet"),
        help="Path to the Parquet dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/artifacts/rf.pkl"),
        help="Where to store the trained model artifact.",
    )
    parser.add_argument(
        "--strategy",
        action="store_true",
        help="If set, compute and display a simple strategy curve on the test set.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Label horizon used for training and evaluation.",
    )
    args = parser.parse_args()

    df = load_parquet(args.data)
    features = make_basic_features(df)
    labels = make_label(df, horizon=args.horizon)

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    ).dropna()

    if dataset.empty:
        raise RuntimeError("Not enough data to train the random forest model")

    X = dataset.loc[:, FEATURE_ORDER]
    y = dataset.loc[:, "label"].astype(int)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_size=args.test_size)

    model = train_rf(X_train, y_train)
    save_model(model, args.model_path)

    y_pred = predict_rf(model, X_test)
    metrics = classification_metrics(y_test, y_pred)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    train_last = dataset.loc[X_train.index, "datetime"].iloc[-1] if not X_train.empty else None
    test_last = dataset.loc[X_test.index, "datetime"].iloc[-1] if not X_test.empty else None
    if train_last is not None:
        print(f"Train last timestamp: {pd.to_datetime(train_last).isoformat()}")
    if test_last is not None:
        print(f"Test last timestamp: {pd.to_datetime(test_last).isoformat()}")

    print("Classification metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"Model saved to {args.model_path.resolve()}")

    if args.strategy and not X_test.empty:
        prediction_series = pd.Series(y_pred, index=X_test.index)
        strategy = _compute_strategy_curve(dataset.loc[X_test.index], prediction_series, args.horizon)
        final_strategy = float(strategy["strategy_curve"].iloc[-1]) if not strategy.empty else float("nan")
        final_buy_hold = float(strategy["buy_hold_curve"].iloc[-1]) if not strategy.empty else float("nan")
        print("Strategy evaluation (cumulative return):")
        print(f"  Strategy curve final value: {final_strategy:.4f}")
        print(f"  Buy & hold final value: {final_buy_hold:.4f}")


if __name__ == "__main__":
    main()
