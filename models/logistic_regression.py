"""Logistic regression training helpers and CLI entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Train a logistic regression pipeline on the provided data."""

    X_df = _ensure_dataframe(X_train).loc[:, FEATURE_ORDER]
    y_series = _ensure_series(y_train)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", n_jobs=1),
            ),
        ]
    )
    pipeline.fit(X_df, y_series)
    return pipeline


def predict_logreg(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Generate binary predictions using the trained pipeline."""

    X_df = _ensure_dataframe(X).loc[:, FEATURE_ORDER]
    predictions = model.predict(X_df)
    return predictions.astype(np.int64)


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist the trained model to ``path`` using joblib."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)


def load_model(path: str | Path) -> Pipeline:
    """Load a serialized model from ``path``."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    model = joblib.load(file_path)
    return model


if __name__ == "__main__":
    DATA_PATH = Path("data/btc_usdt_1m_all.parquet")
    MODEL_PATH = Path("models/artifacts/logreg.pkl")

    df = load_parquet(DATA_PATH)
    features = make_basic_features(df)
    labels = make_label(df, horizon=5)

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime"]],
        ],
        axis=1,
    ).dropna()

    if dataset.empty:
        raise RuntimeError("Not enough data to train the logistic regression model")

    X = dataset.loc[:, FEATURE_ORDER]
    y = dataset.loc[:, "label"].astype(int)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y)

    model = train_logreg(X_train, y_train)
    save_model(model, MODEL_PATH)

    y_pred = predict_logreg(model, X_test)
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

    print(f"Model saved to {MODEL_PATH.resolve()}")
