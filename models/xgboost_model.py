"""XGBoost training utilities and CLI entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.data_loader import load_parquet
from mlops.evaluation import classification_metrics
from mlops.features import FEATURE_COLUMNS, make_basic_features, make_label
from mlops.split import time_train_test_split

FEATURE_ORDER = FEATURE_COLUMNS

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42,
}


def _ensure_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("Expected a pandas DataFrame with named feature columns")


def _ensure_series(y: Any) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """Train an XGBoost classifier using the provided features and labels."""

    X_df = _ensure_dataframe(X_train).loc[:, FEATURE_ORDER]
    y_series = _ensure_series(y_train)
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        **model_params,
    )
    model.fit(X_df, y_series)
    return model


def predict_xgb(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict binary class labels using a trained XGBoost model."""

    X_df = _ensure_dataframe(X).loc[:, FEATURE_ORDER]
    probabilities = model.predict_proba(X_df)[:, 1]
    return (probabilities > 0.5).astype(np.int64)


def save_model(model: XGBClassifier, path: str | Path) -> None:
    """Persist the trained model to ``path`` using joblib."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)


def load_model(path: str | Path) -> XGBClassifier:
    """Load a serialized XGBoost classifier from ``path``."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    model = joblib.load(file_path)
    if not isinstance(model, XGBClassifier):
        raise TypeError("Loaded object is not an XGBClassifier")
    return model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/btc_usdt_1m_all.parquet"),
        help="Path to the Parquet dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/artifacts/xgb.pkl"),
        help="Where to store the trained model artifact.",
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
        raise RuntimeError("Not enough data to train the XGBoost model")

    X = dataset.loc[:, FEATURE_ORDER]
    y = dataset.loc[:, "label"].astype(int)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y)

    model = train_xgb(X_train, y_train)
    save_model(model, args.model_path)

    y_pred = predict_xgb(model, X_test)
    metrics = classification_metrics(y_test, y_pred)

    train_last = dataset.loc[X_train.index, "datetime"].iloc[-1] if not X_train.empty else None
    test_last = dataset.loc[X_test.index, "datetime"].iloc[-1] if not X_test.empty else None

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if train_last is not None:
        print(f"Train last timestamp: {pd.to_datetime(train_last).isoformat()}")
    if test_last is not None:
        print(f"Test last timestamp: {pd.to_datetime(test_last).isoformat()}")

    print("Classification metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"Model saved to {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
