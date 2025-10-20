"""Flask application that keeps BTC/USDT minute data current and serves
an interactive candlestick dashboard."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request
from sklearn.pipeline import Pipeline

from mlops.data_loader import load_parquet
from mlops.features import make_basic_features, make_label
from mlops.split import time_train_test_split
from models.logistic_regression import (
    FEATURE_ORDER as LOGREG_FEATURE_ORDER,
    load_model as load_logreg_model,
    predict_logreg,
)
from models.random_forest import (
    FEATURE_ORDER as RF_FEATURE_ORDER,
    load_model as load_rf_model,
    predict_rf,
)
from models.lstm_model import (
    FEATURE_ORDER as LSTM_FEATURE_ORDER,
    load_model as load_lstm_model,
    predict_lstm_proba,
)
from models.xgboost_model import (
    FEATURE_ORDER as XGB_FEATURE_ORDER,
    load_model as load_xgb_model,
    predict_xgb,
)
from trade_bot import generate_trade_report, serialise_trade_report

APP_UPDATE_INTERVAL = 60  # seconds
DATA_FILE = Path("data/btc_usdt_1m_all.parquet")
ANALYTICS_TEST_PATH = Path("data/ml/test.parquet")
DEFAULT_ANALYTICS_INITIAL_CASH = 1000.0
DEFAULT_ANALYTICS_BET_SIZE = 100.0
DEFAULT_ANALYTICS_TRANSACTION_FEE = 0.0  # percent
DEFAULT_ANALYTICS_BUY_CONFIDENCE = 0.0  # percent
ANALYTICS_MAX_POINTS = 5000
LOGREG_MODEL_PATH = Path("models/artifacts/logreg.pkl")
RF_MODEL_PATH = Path("models/artifacts/rf.pkl")
XGB_MODEL_PATH = Path("models/artifacts/xgb.pkl")
LSTM_MODEL_PATH = Path("models/artifacts/lstm.pt")
DEFAULT_LOOKBACK = timedelta(days=7)
BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_KLINES = 1000
LOGREG_CONTEXT_WINDOW = 60
RF_CONTEXT_WINDOW = 60
XGB_CONTEXT_WINDOW = 60
LSTM_CONTEXT_WINDOW = 60

app = Flask(__name__)

data_lock = threading.Lock()
background_started = False
candles_df: pd.DataFrame | None = None
logreg_model = None
rf_model = None
xgb_model = None
lstm_model = None
logreg_samples: List[Dict[str, object]] = []
logreg_latest_sample: Dict[str, object] | None = None
rf_samples: List[Dict[str, object]] = []
rf_latest_sample: Dict[str, object] | None = None
xgb_samples: List[Dict[str, object]] = []
xgb_latest_sample: Dict[str, object] | None = None
lstm_samples: List[Dict[str, object]] = []
lstm_latest_sample: Dict[str, object] | None = None
FORECAST_MAX_STEPS = 120


def _empty_candles_frame() -> pd.DataFrame:
    """Return a dataframe with the expected candle columns but no rows."""

    return pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
    )


def _ensure_logreg_model_loaded() -> None:
    """Lazy-load the logistic regression model if the artifact exists."""

    global logreg_model
    if logreg_model is not None:
        return
    if not LOGREG_MODEL_PATH.exists():
        return
    try:
        logreg_model = load_logreg_model(LOGREG_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to load logistic regression model: %s", exc)
        logreg_model = None


def _recompute_logreg_cache(df: pd.DataFrame | None) -> None:
    """Update cached logistic regression samples for the web UI."""

    global logreg_samples, logreg_latest_sample

    if df is None or df.empty:
        with data_lock:
            logreg_samples = []
            logreg_latest_sample = None
        return

    _ensure_logreg_model_loaded()
    if logreg_model is None:
        with data_lock:
            logreg_samples = []
            logreg_latest_sample = None
        return

    try:
        features = make_basic_features(df)
        labels = make_label(df, horizon=5)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to prepare features for logistic regression: %s", exc)
        with data_lock:
            logreg_samples = []
            logreg_latest_sample = None
        return

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    )
    dataset = dataset.dropna(subset=LOGREG_FEATURE_ORDER + ["label"])

    samples: List[Dict[str, object]] = []
    if not dataset.empty:
        X = dataset.loc[:, LOGREG_FEATURE_ORDER]
        y = dataset.loc[:, "label"].astype(int)
        X_train, X_test, y_train, y_test = time_train_test_split(X, y)
        if not X_test.empty:
            preds = predict_logreg(logreg_model, X_test)
            try:
                probs = logreg_model.predict_proba(X_test)[:, 1]
            except Exception:  # pragma: no cover - fallback for unexpected models
                probs = preds.astype(float)
            meta = dataset.loc[X_test.index, ["timestamp", "datetime", "close"]].copy()
            meta["ground_truth"] = y_test
            meta["prediction"] = preds
            meta["probability"] = probs
            for feature in LOGREG_FEATURE_ORDER:
                meta[feature] = dataset.loc[X_test.index, feature]
            meta["anchor_timestamp"] = dataset.loc[X_test.index, "timestamp"].astype("int64")
            samples = [
                {
                    "mode": "random",
                    "timestamp": int(row["timestamp"]),
                    "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                    "close": float(row["close"]),
                    "ground_truth": int(row["ground_truth"]),
                    "prediction": int(row["prediction"]),
                    "probability": float(row["probability"]),
                    "features": {
                        feature: float(row[feature]) for feature in LOGREG_FEATURE_ORDER
                    },
                    "anchor_timestamp": int(row["anchor_timestamp"]),
                }
                for _, row in meta.iterrows()
            ]

    latest_record: Dict[str, object] | None = None
    feature_view = features.loc[:, LOGREG_FEATURE_ORDER].dropna()
    if not feature_view.empty:
        latest_index = feature_view.index[-1]
        latest_row = feature_view.loc[[latest_index]]
        latest_pred = int(predict_logreg(logreg_model, latest_row)[0])
        try:
            latest_prob = float(logreg_model.predict_proba(latest_row)[:, 1][0])
        except Exception:  # pragma: no cover - fallback for unexpected models
            latest_prob = float(latest_pred)
        label_value = labels.loc[latest_index] if latest_index in labels.index else None
        ground_truth = None
        if label_value is not None and not pd.isna(label_value):
            ground_truth = int(label_value)
        latest_record = {
            "mode": "latest",
            "timestamp": int(df.loc[latest_index, "timestamp"]),
            "datetime": pd.to_datetime(df.loc[latest_index, "datetime"]).isoformat(),
            "close": float(df.loc[latest_index, "close"]),
            "ground_truth": ground_truth,
            "prediction": latest_pred,
            "probability": latest_prob,
            "features": {
                feature: float(latest_row.iloc[0][feature])
                for feature in LOGREG_FEATURE_ORDER
            },
            "anchor_timestamp": int(df.loc[latest_index, "timestamp"]),
        }

    with data_lock:
        logreg_samples = samples
        logreg_latest_sample = latest_record


def _ensure_rf_model_loaded() -> None:
    """Lazy-load the random forest model if the artifact exists."""

    global rf_model
    if rf_model is not None:
        return
    if not RF_MODEL_PATH.exists():
        return
    try:
        rf_model = load_rf_model(RF_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to load random forest model: %s", exc)
        rf_model = None


def _recompute_rf_cache(df: pd.DataFrame | None) -> None:
    """Update cached random forest samples for the web UI."""

    global rf_samples, rf_latest_sample

    if df is None or df.empty:
        with data_lock:
            rf_samples = []
            rf_latest_sample = None
        return

    _ensure_rf_model_loaded()
    if rf_model is None:
        with data_lock:
            rf_samples = []
            rf_latest_sample = None
        return

    try:
        features = make_basic_features(df)
        labels = make_label(df, horizon=5)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to prepare features for random forest: %s", exc)
        with data_lock:
            rf_samples = []
            rf_latest_sample = None
        return

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    )
    dataset = dataset.dropna(subset=RF_FEATURE_ORDER + ["label"])

    samples: List[Dict[str, object]] = []
    if not dataset.empty:
        X = dataset.loc[:, RF_FEATURE_ORDER]
        y = dataset.loc[:, "label"].astype(int)
        X_train, X_test, y_train, y_test = time_train_test_split(X, y)
        if not X_test.empty:
            preds = predict_rf(rf_model, X_test)
            try:
                probs = rf_model.predict_proba(X_test)[:, 1]
            except Exception:  # pragma: no cover - fallback for unexpected models
                probs = preds.astype(float)
            meta = dataset.loc[X_test.index, ["timestamp", "datetime", "close"]].copy()
            meta["ground_truth"] = y_test
            meta["prediction"] = preds
            meta["probability"] = probs
            for feature in RF_FEATURE_ORDER:
                meta[feature] = dataset.loc[X_test.index, feature]
            meta["anchor_timestamp"] = dataset.loc[X_test.index, "timestamp"].astype("int64")
            samples = [
                {
                    "mode": "random",
                    "timestamp": int(row["timestamp"]),
                    "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                    "close": float(row["close"]),
                    "ground_truth": int(row["ground_truth"]),
                    "prediction": int(row["prediction"]),
                    "probability": float(row["probability"]),
                    "features": {
                        feature: float(row[feature]) for feature in RF_FEATURE_ORDER
                    },
                    "anchor_timestamp": int(row["anchor_timestamp"]),
                }
                for _, row in meta.iterrows()
            ]

    latest_record: Dict[str, object] | None = None
    feature_view = features.loc[:, RF_FEATURE_ORDER].dropna()
    if not feature_view.empty:
        latest_index = feature_view.index[-1]
        latest_row = feature_view.loc[[latest_index]]
        latest_pred = int(predict_rf(rf_model, latest_row)[0])
        try:
            latest_prob = float(rf_model.predict_proba(latest_row)[:, 1][0])
        except Exception:  # pragma: no cover - fallback for unexpected models
            latest_prob = float(latest_pred)
        label_value = labels.loc[latest_index] if latest_index in labels.index else None
        ground_truth = None
        if label_value is not None and not pd.isna(label_value):
            ground_truth = int(label_value)
        latest_record = {
            "mode": "latest",
            "timestamp": int(df.loc[latest_index, "timestamp"]),
            "datetime": pd.to_datetime(df.loc[latest_index, "datetime"]).isoformat(),
            "close": float(df.loc[latest_index, "close"]),
            "ground_truth": ground_truth,
            "prediction": latest_pred,
            "probability": latest_prob,
            "features": {
                feature: float(latest_row.iloc[0][feature]) for feature in RF_FEATURE_ORDER
            },
            "anchor_timestamp": int(df.loc[latest_index, "timestamp"]),
        }

    with data_lock:
        rf_samples = samples
        rf_latest_sample = latest_record


def _ensure_xgb_model_loaded() -> None:
    """Lazy-load the XGBoost model if the artifact exists."""

    global xgb_model
    if xgb_model is not None:
        return
    if not XGB_MODEL_PATH.exists():
        return
    try:
        xgb_model = load_xgb_model(XGB_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to load XGBoost model: %s", exc)
        xgb_model = None


def _recompute_xgb_cache(df: pd.DataFrame | None) -> None:
    """Update cached XGBoost samples for the web UI."""

    global xgb_samples, xgb_latest_sample

    if df is None or df.empty:
        with data_lock:
            xgb_samples = []
            xgb_latest_sample = None
        return

    _ensure_xgb_model_loaded()
    if xgb_model is None:
        with data_lock:
            xgb_samples = []
            xgb_latest_sample = None
        return

    try:
        features = make_basic_features(df)
        labels = make_label(df, horizon=5)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to prepare features for XGBoost: %s", exc)
        with data_lock:
            xgb_samples = []
            xgb_latest_sample = None
        return

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    )
    dataset = dataset.dropna(subset=XGB_FEATURE_ORDER + ["label"])

    samples: List[Dict[str, object]] = []
    if not dataset.empty:
        X = dataset.loc[:, XGB_FEATURE_ORDER]
        y = dataset.loc[:, "label"].astype(int)
        X_train, X_test, y_train, y_test = time_train_test_split(X, y)
        if not X_test.empty:
            preds = predict_xgb(xgb_model, X_test)
            try:
                probs = xgb_model.predict_proba(X_test)[:, 1]
            except Exception:  # pragma: no cover - fallback for unexpected models
                probs = preds.astype(float)
            meta = dataset.loc[X_test.index, ["timestamp", "datetime", "close"]].copy()
            meta["ground_truth"] = y_test
            meta["prediction"] = preds
            meta["probability"] = probs
            for feature in XGB_FEATURE_ORDER:
                meta[feature] = dataset.loc[X_test.index, feature]
            meta["anchor_timestamp"] = dataset.loc[X_test.index, "timestamp"].astype("int64")
            samples = [
                {
                    "mode": "random",
                    "timestamp": int(row["timestamp"]),
                    "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                    "close": float(row["close"]),
                    "ground_truth": int(row["ground_truth"]),
                    "prediction": int(row["prediction"]),
                    "probability": float(row["probability"]),
                    "features": {
                        feature: float(row[feature]) for feature in XGB_FEATURE_ORDER
                    },
                    "anchor_timestamp": int(row["anchor_timestamp"]),
                }
                for _, row in meta.iterrows()
            ]

    latest_record: Dict[str, object] | None = None
    feature_view = features.loc[:, XGB_FEATURE_ORDER].dropna()
    if not feature_view.empty:
        latest_index = feature_view.index[-1]
        latest_row = feature_view.loc[[latest_index]]
        latest_pred = int(predict_xgb(xgb_model, latest_row)[0])
        try:
            latest_prob = float(xgb_model.predict_proba(latest_row)[:, 1][0])
        except Exception:  # pragma: no cover - fallback for unexpected models
            latest_prob = float(latest_pred)
        label_value = labels.loc[latest_index] if latest_index in labels.index else None
        ground_truth = None
        if label_value is not None and not pd.isna(label_value):
            ground_truth = int(label_value)
        latest_record = {
            "mode": "latest",
            "timestamp": int(df.loc[latest_index, "timestamp"]),
            "datetime": pd.to_datetime(df.loc[latest_index, "datetime"]).isoformat(),
            "close": float(df.loc[latest_index, "close"]),
            "ground_truth": ground_truth,
            "prediction": latest_pred,
            "probability": latest_prob,
            "features": {
                feature: float(latest_row.iloc[0][feature]) for feature in XGB_FEATURE_ORDER
            },
            "anchor_timestamp": int(df.loc[latest_index, "timestamp"]),
        }

    with data_lock:
        xgb_samples = samples
        xgb_latest_sample = latest_record


def _ensure_lstm_model_loaded() -> None:
    """Lazy-load the LSTM model if the artifact exists."""

    global lstm_model
    if lstm_model is not None:
        return
    if not LSTM_MODEL_PATH.exists():
        return
    try:
        lstm_model = load_lstm_model(LSTM_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to load LSTM model: %s", exc)
        lstm_model = None


def _recompute_lstm_cache(df: pd.DataFrame | None) -> None:
    """Update cached LSTM samples for the web UI."""

    global lstm_samples, lstm_latest_sample

    if df is None or df.empty:
        with data_lock:
            lstm_samples = []
            lstm_latest_sample = None
        return

    _ensure_lstm_model_loaded()
    if lstm_model is None:
        with data_lock:
            lstm_samples = []
            lstm_latest_sample = None
        return

    try:
        features = make_basic_features(df)
        labels = make_label(df, horizon=5)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning("Failed to prepare features for LSTM model: %s", exc)
        with data_lock:
            lstm_samples = []
            lstm_latest_sample = None
        return

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    )
    dataset = dataset.dropna(subset=LSTM_FEATURE_ORDER + ["label"])
    sequence_length = getattr(lstm_model, "sequence_length", LSTM_CONTEXT_WINDOW)
    if len(dataset) < sequence_length:
        with data_lock:
            lstm_samples = []
            lstm_latest_sample = None
        return

    dataset = dataset.iloc[sequence_length - 1 :]

    samples: List[Dict[str, object]] = []
    if not dataset.empty:
        X = dataset.loc[:, LSTM_FEATURE_ORDER]
        y = dataset.loc[:, "label"].astype(int)
        X_train, X_test, y_train, y_test = time_train_test_split(X, y)
        if not X_test.empty:
            probs = predict_lstm_proba(lstm_model, X_test)
            if not probs.empty:
                preds = (probs >= 0.5).astype(int)
                aligned_truth = y_test.loc[probs.index]
                meta = dataset.loc[probs.index, ["timestamp", "datetime", "close"]].copy()
                meta["ground_truth"] = aligned_truth
                meta["prediction"] = preds
                meta["probability"] = probs
                for feature in LSTM_FEATURE_ORDER:
                    meta[feature] = dataset.loc[probs.index, feature]
                meta["anchor_timestamp"] = dataset.loc[probs.index, "timestamp"].astype("int64")
                samples = [
                    {
                        "mode": "random",
                        "timestamp": int(row["timestamp"]),
                        "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                        "close": float(row["close"]),
                        "ground_truth": int(row["ground_truth"]),
                        "prediction": int(row["prediction"]),
                        "probability": float(row["probability"]),
                        "features": {
                            feature: float(row[feature]) for feature in LSTM_FEATURE_ORDER
                        },
                        "anchor_timestamp": int(row["anchor_timestamp"]),
                    }
                    for _, row in meta.iterrows()
                ]

    latest_record: Dict[str, object] | None = None
    feature_view = features.loc[:, LSTM_FEATURE_ORDER].dropna()
    if len(feature_view) >= sequence_length:
        feature_view = feature_view.iloc[sequence_length - 1 :]
        if not feature_view.empty:
            probs_full = predict_lstm_proba(lstm_model, feature_view)
            if not probs_full.empty:
                latest_index = probs_full.index[-1]
                latest_prob = float(probs_full.iloc[-1])
                latest_pred = int(latest_prob >= 0.5)
                label_value = labels.loc[latest_index] if latest_index in labels.index else None
                ground_truth = None
                if label_value is not None and not pd.isna(label_value):
                    ground_truth = int(label_value)
                latest_record = {
                    "mode": "latest",
                    "timestamp": int(df.loc[latest_index, "timestamp"]),
                    "datetime": pd.to_datetime(df.loc[latest_index, "datetime"]).isoformat(),
                    "close": float(df.loc[latest_index, "close"]),
                    "ground_truth": ground_truth,
                    "prediction": latest_pred,
                    "probability": latest_prob,
                    "features": {
                        feature: float(feature_view.loc[latest_index, feature])
                        for feature in LSTM_FEATURE_ORDER
                    },
                    "anchor_timestamp": int(df.loc[latest_index, "timestamp"]),
                }

    with data_lock:
        lstm_samples = samples
        lstm_latest_sample = latest_record


def _ensure_candles_ready(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
        )
    if df.empty:
        return df
    required = {"timestamp", "open", "high", "low", "close", "volume", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candlestick dataframe missing columns: {sorted(missing)}")
    return df.sort_values("timestamp").copy()


def _wick_statistics(history: pd.DataFrame) -> Tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    recent = history.tail(120)
    high_ref = recent[["open", "close"]].max(axis=1)
    low_ref = recent[["open", "close"]].min(axis=1)
    upper = (recent["high"] - high_ref).clip(lower=0)
    lower = (low_ref - recent["low"]).clip(lower=0)
    return float(upper.mean(skipna=True) or 0.0), float(lower.mean(skipna=True) or 0.0)


def _return_statistics(history: pd.DataFrame) -> Tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    returns = history.sort_values("timestamp")["close"].pct_change()
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    pos_mean = float(positive.mean(skipna=True) or 0.0)
    neg_mean = float(negative.mean(skipna=True) or 0.0)
    if not np.isfinite(pos_mean):
        pos_mean = 0.0
    if not np.isfinite(neg_mean):
        neg_mean = 0.0
    return pos_mean, neg_mean


def _latest_probability(
    model: Any,
    feature_view: pd.DataFrame,
    order: Iterable[str],
) -> float:
    if feature_view.empty:
        raise ValueError("Feature view is empty")
    ordered = feature_view.loc[:, list(order)]
    if hasattr(model, "predict_latest_proba"):
        return float(model.predict_latest_proba(ordered))
    latest_row = ordered.iloc[[-1]]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(latest_row)
        if isinstance(proba, np.ndarray):
            return float(proba[0][1] if proba.ndim == 2 else proba[0])
    prediction = model.predict(latest_row)
    if isinstance(prediction, pd.Series):
        return float(prediction.iloc[-1])
    if isinstance(prediction, np.ndarray):
        return float(prediction[-1])
    try:
        return float(prediction)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise TypeError("Unsupported model prediction format") from exc


def _forecast_candles(
    model: Any,
    history: pd.DataFrame,
    steps: int,
    interval_ms: int = 60_000,
    feature_order: Iterable[str] | None = None,
) -> List[Dict[str, object]]:
    """Generate sequential candle forecasts using the supplied classification model."""

    prepared = _ensure_candles_ready(history)
    if prepared.empty or steps <= 0:
        return []

    order = list(feature_order or LOGREG_FEATURE_ORDER)
    pos_mean, neg_mean = _return_statistics(prepared)
    upper_wick, lower_wick = _wick_statistics(prepared)
    recent_volumes = prepared["volume"].tail(120)
    fallback_volume = float(recent_volumes.mean(skipna=True) or 0.0)

    forecasts: List[Dict[str, object]] = []
    working = prepared.copy()

    for _ in range(steps):
        features = make_basic_features(working)
        try:
            feature_view = features.loc[:, order].dropna()
        except KeyError:
            break
        if feature_view.empty:
            break
        try:
            proba = _latest_probability(model, feature_view, order)
        except Exception:  # pragma: no cover - fallback for unexpected models
            latest_row = feature_view.iloc[[-1]]
            proba = float(model.predict(latest_row)[0])
        expected_return = proba * pos_mean + (1 - proba) * neg_mean

        last_row = working.iloc[-1]
        previous_close = float(last_row["close"])
        predicted_close = previous_close * (1 + expected_return)
        predicted_open = previous_close
        high_body = max(predicted_open, predicted_close)
        low_body = min(predicted_open, predicted_close)
        predicted_high = high_body + upper_wick
        predicted_low = max(low_body - lower_wick, 0.0)
        predicted_volume = float(working["volume"].tail(30).mean(skipna=True) or fallback_volume)

        timestamp = int(last_row["timestamp"]) + interval_ms
        datetime_value = pd.to_datetime(last_row["datetime"]) + pd.Timedelta(milliseconds=interval_ms)

        forecast_row = {
            "timestamp": timestamp,
            "datetime": datetime_value,
            "open": float(predicted_open),
            "high": float(predicted_high),
            "low": float(predicted_low),
            "close": float(predicted_close),
            "volume": float(predicted_volume),
            "probability": proba,
            "expected_return": float(expected_return),
        }
        forecasts.append(forecast_row)

        appended = pd.DataFrame([forecast_row])
        working = pd.concat([working, appended], ignore_index=True)

    return forecasts


def _ensure_dataset_dir() -> None:
    if DATA_FILE.parent and not DATA_FILE.parent.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_analytics_dataset() -> pd.DataFrame:
    """Load the hold-out test dataset used for analytics simulations."""

    if not ANALYTICS_TEST_PATH.exists():
        raise FileNotFoundError(
            "Test dataset not found. Run split_dataset.py to generate data/ml/test.parquet"
        )
    try:
        df = load_parquet(ANALYTICS_TEST_PATH)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Failed to load analytics dataset: {exc}") from exc
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe uses the canonical schema."""
    rename_map = {}
    if "ts" in df.columns:
        rename_map["ts"] = "timestamp"
    if "open_time" in df.columns and "timestamp" not in rename_map:
        rename_map["open_time"] = "timestamp"
    if "volume" not in df.columns and "quote_asset_volume" in df.columns:
        rename_map["quote_asset_volume"] = "volume"
    df = df.rename(columns=rename_map)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet dataset is missing required columns: {sorted(missing)}")

    # Enforce dtypes and add datetime helper column.
    df = df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def _read_local_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return _empty_candles_frame()
    try:
        df = pd.read_parquet(DATA_FILE)
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.warning(
            "Failed to read parquet dataset %s (%s); treating as empty.", DATA_FILE, exc
        )
        return _empty_candles_frame()
    return _normalise_columns(df)


def _save_dataset(df: pd.DataFrame) -> None:
    _ensure_dataset_dir()
    df.to_parquet(DATA_FILE, index=False)


def _round_down_to_minute(ts: int) -> int:
    """Return the largest millisecond timestamp rounded down to the minute."""
    return (ts // 60_000) * 60_000


def _fetch_binance_klines(start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    """Fetch 1m klines from Binance between start_ms (inclusive) and end_ms (exclusive)."""
    rows: List[List[float]] = []
    cursor = start_ms
    # Binance expects startTime < endTime. Align end to closed minute.
    effective_end = end_ms or _round_down_to_minute(int(time.time() * 1000))
    while cursor < effective_end:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": cursor,
            "limit": MAX_KLINES,
        }
        if effective_end:
            params["endTime"] = effective_end
        resp = requests.get(BINANCE_REST, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            break

        rows.extend(payload)
        last_open_time = int(payload[-1][0])
        # Binance returns candles up to and including last_open_time.
        next_cursor = last_open_time + 60_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(payload) < MAX_KLINES:
            break
        time.sleep(0.2)  # polite rate limiting

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

    data = {
        "timestamp": [int(r[0]) for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[df["timestamp"] < effective_end]


def _get_last_timestamp_ms(df: pd.DataFrame) -> int | None:
    if df.empty:
        return None
    return int(df["timestamp"].max())


def _initial_start_timestamp() -> int:
    lookback = datetime.now(timezone.utc) - DEFAULT_LOOKBACK
    return int(lookback.timestamp() * 1000)


def update_dataset() -> bool:
    """Update the cached dataframe and parquet file with the latest Binance data."""
    global candles_df
    with data_lock:
        current = candles_df.copy() if candles_df is not None else _read_local_dataset()
    if current is None or current.empty:
        start_ms = _initial_start_timestamp()
    else:
        last_ts = _get_last_timestamp_ms(current)
        start_ms = (last_ts + 60_000) if last_ts is not None else _initial_start_timestamp()

    now_ms = _round_down_to_minute(int(time.time() * 1000))
    if start_ms >= now_ms:
        return False

    fresh = _fetch_binance_klines(start_ms, end_ms=now_ms)
    if fresh.empty:
        return False

    combined = pd.concat([current, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)

    with data_lock:
        candles_df = combined
    _save_dataset(combined)

    _recompute_logreg_cache(combined.copy())
    _recompute_rf_cache(combined.copy())
    _recompute_xgb_cache(combined.copy())
    _recompute_lstm_cache(combined.copy())
    return True


def ensure_dataset_loaded() -> None:
    global candles_df
    df = _read_local_dataset()
    with data_lock:
        candles_df = df
    _recompute_logreg_cache(df.copy())
    _recompute_rf_cache(df.copy())
    _recompute_xgb_cache(df.copy())
    _recompute_lstm_cache(df.copy())
    try:
        update_dataset()
    except requests.RequestException as exc:
        app.logger.warning("Initial data update failed: %s", exc)


@app.before_request
def _start_background_updater() -> None:
    """Ensure the background updater thread is started once per process."""
    global background_started
    with data_lock:
        if background_started:
            return
        background_started = True

    def _runner() -> None:
        while True:
            try:
                update_dataset()
            except Exception as exc:  # pragma: no cover - defensive logging
                app.logger.exception("Background update failed: %s", exc)
            time.sleep(APP_UPDATE_INTERVAL)

    thread = threading.Thread(target=_runner, daemon=True, name="binance-updater")
    thread.start()


def _resample(df: pd.DataFrame, interval: str, limit: int) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    if interval == "1m":
        return df.tail(limit).copy()

    rule_map = {
        "5m": "5T",
        "15m": "15T",
        "1h": "1H",
        "4h": "4H",
    }
    if interval not in rule_map:
        raise ValueError(f"Unsupported interval: {interval}")

    indexed = df.set_index("datetime")
    aggregated = indexed.resample(rule_map[interval]).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna()
    aggregated["timestamp"] = (aggregated.index.view("int64") // 1_000_000).astype("int64")
    aggregated["datetime"] = aggregated.index
    return aggregated.tail(limit).reset_index(drop=True)


def _serialise_candles(rows: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    serialised = []
    for row in rows:
        serialised.append(
            {
                "timestamp": int(row["timestamp"]),
                "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
        )
    return serialised


def _serialise_forecast(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    serialised: List[Dict[str, object]] = []
    for row in rows:
        serialised.append(
            {
                "timestamp": int(row["timestamp"]),
                "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "probability": float(row.get("probability", 0.0)),
                "expected_return": float(row.get("expected_return", 0.0)),
            }
        )
    return serialised


def _analytics_probability_series(df: pd.DataFrame, model_name: str) -> pd.Series:
    """Compute model probabilities for each candle in ``df``."""

    features = make_basic_features(df)
    if features.empty:
        return pd.Series(dtype="float64")

    model_key = model_name.lower()
    order_map = {
        "logreg": LOGREG_FEATURE_ORDER,
        "rf": RF_FEATURE_ORDER,
        "xgb": XGB_FEATURE_ORDER,
        "lstm": LSTM_FEATURE_ORDER,
    }
    if model_key not in order_map:
        raise ValueError(f"Unsupported model for analytics: {model_name}")

    order = list(order_map[model_key])
    feature_view = features.loc[:, order].dropna()
    if feature_view.empty:
        return pd.Series(dtype="float64")

    if model_key == "logreg":
        _ensure_logreg_model_loaded()
        if logreg_model is None:
            raise RuntimeError("Logistic regression model is unavailable.")
        proba = logreg_model.predict_proba(feature_view)
        values = proba[:, 1] if isinstance(proba, np.ndarray) else np.asarray(proba, dtype=float)
        return pd.Series(values.astype(float), index=feature_view.index)

    if model_key == "rf":
        _ensure_rf_model_loaded()
        if rf_model is None:
            raise RuntimeError("Random forest model is unavailable.")
        proba = rf_model.predict_proba(feature_view)
        values = proba[:, 1] if isinstance(proba, np.ndarray) else np.asarray(proba, dtype=float)
        return pd.Series(values.astype(float), index=feature_view.index)

    if model_key == "xgb":
        _ensure_xgb_model_loaded()
        if xgb_model is None:
            raise RuntimeError("XGBoost model is unavailable.")
        proba = xgb_model.predict_proba(feature_view)
        values = proba[:, 1] if isinstance(proba, np.ndarray) else np.asarray(proba, dtype=float)
        return pd.Series(values.astype(float), index=feature_view.index)

    _ensure_lstm_model_loaded()
    if lstm_model is None:
        raise RuntimeError("LSTM model is unavailable.")
    probabilities = predict_lstm_proba(lstm_model, feature_view)
    if probabilities.empty:
        return probabilities.astype(float)
    return probabilities.astype(float)


def _build_analytics_frame(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Prepare the dataframe used for analytics backtesting."""

    probabilities = _analytics_probability_series(df, model_name)
    if probabilities.empty:
        return pd.DataFrame(columns=["timestamp", "datetime", "close", "probability", "expected_return"])

    ordered = df.sort_values("timestamp")
    returns = ordered["close"].pct_change()
    positive = returns.where(returns > 0)
    negative = returns.where(returns < 0)
    pos_sum = positive.fillna(0.0).cumsum()
    pos_count = positive.notna().cumsum().replace(0, np.nan)
    neg_sum = negative.fillna(0.0).cumsum()
    neg_count = negative.notna().cumsum().replace(0, np.nan)
    pos_mean = (pos_sum / pos_count).fillna(0.0)
    neg_mean = (neg_sum / neg_count).fillna(0.0)

    frame = ordered.loc[probabilities.index, ["timestamp", "datetime", "close"]].copy()
    frame["probability"] = probabilities.loc[frame.index].astype(float)
    frame["positive_mean"] = pos_mean.loc[frame.index].astype(float)
    frame["negative_mean"] = neg_mean.loc[frame.index].astype(float)
    frame["expected_return"] = (
        frame["probability"] * frame["positive_mean"]
        + (1.0 - frame["probability"]) * frame["negative_mean"]
    )
    return frame


def _sample_points(sequence: List[Dict[str, object]], max_points: int) -> List[Dict[str, object]]:
    if not sequence:
        return []
    if len(sequence) <= max_points:
        return list(sequence)
    step = max(1, len(sequence) // max_points)
    sampled = sequence[::step]
    if sampled[-1] is not sequence[-1]:
        if sampled[-1].get("timestamp") != sequence[-1].get("timestamp"):
            sampled.append(sequence[-1])
    return sampled


def _simulate_trades(
    frame: pd.DataFrame,
    mode: str,
    initial_cash: float,
    bet_size: float,
    transaction_fee_pct: float,
    buy_confidence_pct: float,
) -> Dict[str, object]:
    """Simulate trades over ``frame`` using the trade bot heuristics."""

    if frame.empty:
        return {
            "equity_curve": [],
            "trades": [],
            "buys": [],
            "sells": [],
            "stats": {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "return_pct": 0.0,
                "final_equity": float(initial_cash),
                "max_drawdown": 0.0,
                "avg_trade_return": 0.0,
                "avg_trade_duration_minutes": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "exposure_ratio": 0.0,
            },
        }

    cash = float(initial_cash)
    position_units = 0.0
    current_trade: Dict[str, float] | None = None
    trades: List[Dict[str, object]] = []
    buy_markers: List[Dict[str, object]] = []
    sell_markers: List[Dict[str, object]] = []
    equity_points: List[Dict[str, object]] = []
    exposure_minutes = 0.0
    total_fees = 0.0

    mode_key = mode.lower()
    if mode_key not in {"cumulative", "fixed"}:
        raise ValueError(f"Unsupported analytics mode: {mode}")

    bet_size = max(0.0, float(bet_size))
    fee_rate = max(0.0, float(transaction_fee_pct)) / 100.0
    buy_confidence = max(0.0, float(buy_confidence_pct)) / 100.0

    timestamps = frame["timestamp"].to_numpy(dtype="int64")

    for idx, row in frame.iterrows():
        price = float(row["close"])
        if price <= 0:
            equity_points.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                    "equity": cash + position_units * price,
                }
            )
            continue

        probability = float(row["probability"])
        expected_return = float(row["expected_return"])
        timestamp = int(row["timestamp"])
        datetime_value = pd.to_datetime(row["datetime"]).isoformat()

        if current_trade is None:
            meets_confidence = expected_return > 0 and expected_return >= buy_confidence
            if meets_confidence and probability >= 0.52 and cash > 0:
                stake = cash if mode_key == "cumulative" else min(bet_size, cash)
                if stake > 0:
                    max_affordable = cash / (1.0 + fee_rate) if fee_rate > 0 else cash
                    if stake > max_affordable:
                        stake = max_affordable
                    if stake > 0:
                        buy_fee = stake * fee_rate
                        units = stake / price if price else 0.0
                        total_cost = stake + buy_fee
                        if units > 0 and total_cost <= cash + 1e-12:
                            cash -= total_cost
                            position_units = units
                            current_trade = {
                                "entry_timestamp": timestamp,
                                "entry_datetime": datetime_value,
                                "entry_price": price,
                                "stake": stake,
                                "units": units,
                                "buy_fee": buy_fee,
                            }
                            total_fees += buy_fee
                            buy_markers.append(
                                {"timestamp": timestamp, "datetime": datetime_value, "price": price}
                            )
        else:
            if expected_return < 0 and probability <= 0.48:
                units = float(current_trade["units"])
                proceeds = units * price
                sell_fee = proceeds * fee_rate
                net_proceeds = proceeds - sell_fee
                cash += net_proceeds
                pnl = net_proceeds - (
                    float(current_trade["stake"]) + float(current_trade.get("buy_fee", 0.0))
                )
                entry_price = float(current_trade["entry_price"])
                cost_basis = float(current_trade["stake"]) + float(current_trade.get("buy_fee", 0.0))
                return_pct = (pnl / cost_basis) if cost_basis else 0.0
                duration_minutes = (timestamp - float(current_trade["entry_timestamp"])) / 60000.0
                exposure_minutes += max(duration_minutes, 0.0)
                trade = {
                    **current_trade,
                    "exit_timestamp": timestamp,
                    "exit_datetime": datetime_value,
                    "exit_price": price,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "duration_minutes": duration_minutes,
                    "sell_fee": sell_fee,
                    "total_fees": float(current_trade.get("buy_fee", 0.0)) + sell_fee,
                }
                trades.append(trade)
                sell_markers.append(
                    {"timestamp": timestamp, "datetime": datetime_value, "price": price}
                )
                position_units = 0.0
                current_trade = None
                total_fees += sell_fee

        equity = cash + position_units * price
        equity_points.append({"timestamp": timestamp, "datetime": datetime_value, "equity": equity})

    if current_trade is not None and not frame.empty:
        last_row = frame.iloc[-1]
        price = float(last_row["close"])
        timestamp = int(last_row["timestamp"])
        datetime_value = pd.to_datetime(last_row["datetime"]).isoformat()
        units = float(current_trade["units"])
        proceeds = units * price
        sell_fee = proceeds * fee_rate
        net_proceeds = proceeds - sell_fee
        cash += net_proceeds
        pnl = net_proceeds - (
            float(current_trade["stake"]) + float(current_trade.get("buy_fee", 0.0))
        )
        entry_price = float(current_trade["entry_price"])
        cost_basis = float(current_trade["stake"]) + float(current_trade.get("buy_fee", 0.0))
        return_pct = (pnl / cost_basis) if cost_basis else 0.0
        duration_minutes = (timestamp - float(current_trade["entry_timestamp"])) / 60000.0
        exposure_minutes += max(duration_minutes, 0.0)
        trade = {
            **current_trade,
            "exit_timestamp": timestamp,
            "exit_datetime": datetime_value,
            "exit_price": price,
            "pnl": pnl,
            "return_pct": return_pct,
            "duration_minutes": duration_minutes,
            "sell_fee": sell_fee,
            "total_fees": float(current_trade.get("buy_fee", 0.0)) + sell_fee,
        }
        trades.append(trade)
        sell_markers.append({"timestamp": timestamp, "datetime": datetime_value, "price": price})
        equity_points[-1]["equity"] = cash
        position_units = 0.0
        total_fees += sell_fee

    equity_values = np.array([point["equity"] for point in equity_points], dtype=float)
    if equity_values.size:
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = np.where(running_max > 0, equity_values / running_max - 1.0, 0.0)
        max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0
    else:
        max_drawdown = 0.0

    final_equity = float(equity_values[-1]) if equity_values.size else float(initial_cash)
    total_pnl = final_equity - float(initial_cash)
    return_pct = (final_equity / float(initial_cash) - 1.0) if initial_cash else 0.0
    trade_returns = [float(trade["return_pct"]) for trade in trades]
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0
    avg_duration = (
        float(np.mean([trade["duration_minutes"] for trade in trades])) if trades else 0.0
    )
    best_trade = max(trade_returns) if trade_returns else 0.0
    worst_trade = min(trade_returns) if trade_returns else 0.0
    wins = sum(1 for trade in trades if float(trade["pnl"]) > 0)
    win_rate = wins / len(trades) if trades else 0.0

    total_minutes = (
        (timestamps[-1] - timestamps[0]) / 60000.0 if len(timestamps) > 1 else 0.0
    )
    exposure_ratio = (exposure_minutes / total_minutes) if total_minutes > 0 else 0.0

    stats = {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "final_equity": final_equity,
        "max_drawdown": abs(max_drawdown),
        "avg_trade_return": avg_trade_return,
        "avg_trade_duration_minutes": avg_duration,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "exposure_ratio": exposure_ratio,
        "total_fees": total_fees,
    }

    return {
        "equity_curve": equity_points,
        "trades": trades,
        "buys": buy_markers,
        "sells": sell_markers,
        "stats": stats,
    }


def _serialise_price_series(frame: pd.DataFrame, max_points: int) -> List[Dict[str, object]]:
    if frame.empty:
        return []
    ordered = frame.sort_values("timestamp")
    step = max(1, len(ordered) // max_points)
    subset = ordered.iloc[::step]
    if subset.iloc[-1]["timestamp"] != ordered.iloc[-1]["timestamp"]:
        subset = pd.concat([subset, ordered.iloc[[-1]]]).drop_duplicates("timestamp", keep="last")
    serialised: List[Dict[str, object]] = []
    for _, row in subset.iterrows():
        serialised.append(
            {
                "timestamp": int(row["timestamp"]),
                "datetime": pd.to_datetime(row["datetime"]).isoformat(),
                "price": float(row["close"]),
            }
        )
    return serialised


def _serialise_equity_curve(points: List[Dict[str, object]], max_points: int) -> List[Dict[str, object]]:
    if not points:
        return []
    sampled = _sample_points(points, max_points)
    return [
        {
            "timestamp": int(point["timestamp"]),
            "datetime": pd.to_datetime(point["datetime"]).isoformat(),
            "equity": float(point["equity"]),
        }
        for point in sampled
    ]


@app.route("/api/candles")
def api_candles() -> Dict[str, object]:
    interval = request.args.get("interval", "1m")
    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    limit = max(10, min(limit, 1000))

    with data_lock:
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if df is None or df.empty:
        return jsonify({"interval": interval, "candles": [], "last_update": None})

    try:
        subset = _resample(df, interval, limit)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "interval": interval,
            "candles": _serialise_candles(subset.to_dict("records")),
            "last_update": pd.Timestamp.utcnow().isoformat(),
        }
    )


def _coerce_minutes(value: str | None, default: int = 15) -> int:
    try:
        minutes = int(value) if value is not None else default
    except ValueError:
        minutes = default
    return max(1, min(minutes, FORECAST_MAX_STEPS))


def _extract_minutes(args) -> int:
    if "minutes" in args:
        return _coerce_minutes(args.get("minutes"))
    return _coerce_minutes(args.get("steps"))


@app.route("/api/logreg/sample")
def api_logreg_sample():
    mode = request.args.get("mode", "random").lower()
    horizon_minutes = _extract_minutes(request.args)

    with data_lock:
        model_available = logreg_model is not None
        samples = [sample.copy() for sample in logreg_samples]
        latest = logreg_latest_sample.copy() if logreg_latest_sample else None
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if not model_available:
        return jsonify({"error": "Logistic regression model is unavailable."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503
    ordered = prepared.sort_values("timestamp").reset_index(drop=True)

    if mode == "latest":
        if latest is None:
            return jsonify({"error": "Latest sample unavailable."}), 404
        payload = latest.copy()
        payload["mode"] = "latest"
    else:
        if not samples:
            return jsonify({"error": "No evaluation samples available."}), 404
        payload = random.choice(samples)
        payload["mode"] = "random"

    anchor_ts = int(payload.get("anchor_timestamp", payload.get("timestamp", 0)))
    matches = ordered.index[ordered["timestamp"] == anchor_ts]
    if matches.empty:
        return jsonify({"error": "Anchor candle unavailable in dataset."}), 404

    anchor_pos = int(matches[-1])
    history_full = ordered.iloc[: anchor_pos + 1].copy()
    if history_full.empty:
        return jsonify({"error": "Not enough history to generate sample context."}), 503

    forecasts = _forecast_candles(
        logreg_model,
        history_full,
        horizon_minutes,
        feature_order=LOGREG_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    history_preview = history_full.tail(LOGREG_CONTEXT_WINDOW)
    gt_slice = ordered.iloc[anchor_pos + 1 : anchor_pos + 1 + len(forecasts)]

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_slice.to_dict("records"),
    )

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))
    payload["trade_bot"] = serialise_trade_report(trade_report)

    return jsonify(payload)


@app.route("/api/logreg/forecast")
def api_logreg_forecast():
    mode = request.args.get("mode", "random").lower()
    steps = _extract_minutes(request.args)

    with data_lock:
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if df.empty:
        return jsonify({"error": "No candle data available."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503

    if mode not in {"latest", "random"}:
        return jsonify({"error": f"Unsupported forecast mode: {mode}"}), 400

    if mode == "latest":
        anchor_idx = len(prepared) - 1
    else:
        if len(prepared) < steps + 2:
            return jsonify({"error": "Not enough historical data for random forecast."}), 503
        anchor_idx = random.randint(0, len(prepared) - steps - 2)

    anchor_row = prepared.iloc[anchor_idx]
    history = prepared.iloc[: anchor_idx + 1]

    _ensure_logreg_model_loaded()
    if logreg_model is None:
        return jsonify({"error": "Logistic regression model is unavailable."}), 503

    forecasts = _forecast_candles(
        logreg_model,
        history,
        steps,
        feature_order=LOGREG_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    ground_truth: List[Dict[str, object]] = []
    gt_records: List[Dict[str, object]] = []
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        gt_records = gt_slice.to_dict("records")
        ground_truth = _serialise_candles(gt_records)

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(LOGREG_CONTEXT_WINDOW)

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_records,
    )

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
            "trade_bot": serialise_trade_report(trade_report),
        }
    )


@app.route("/api/xgb/sample")
def api_xgb_sample():
    mode = request.args.get("mode", "random").lower()
    horizon_minutes = _extract_minutes(request.args)

    _ensure_xgb_model_loaded()

    with data_lock:
        model_available = xgb_model is not None
        samples = [sample.copy() for sample in xgb_samples]
        latest = xgb_latest_sample.copy() if xgb_latest_sample else None
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if not model_available:
        return jsonify({"error": "XGBoost model is unavailable."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503
    ordered = prepared.sort_values("timestamp").reset_index(drop=True)

    if mode == "latest":
        if latest is None:
            return jsonify({"error": "Latest sample unavailable."}), 404
        payload = latest.copy()
        payload["mode"] = "latest"
    else:
        if not samples:
            return jsonify({"error": "No evaluation samples available."}), 404
        payload = random.choice(samples)
        payload["mode"] = "random"

    anchor_ts = int(payload.get("anchor_timestamp", payload.get("timestamp", 0)))
    matches = ordered.index[ordered["timestamp"] == anchor_ts]
    if matches.empty:
        return jsonify({"error": "Anchor candle unavailable in dataset."}), 404

    anchor_pos = int(matches[-1])
    history_full = ordered.iloc[: anchor_pos + 1].copy()
    if history_full.empty:
        return jsonify({"error": "Not enough history to generate sample context."}), 503

    forecasts = _forecast_candles(
        xgb_model,
        history_full,
        horizon_minutes,
        feature_order=XGB_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    history_preview = history_full.tail(XGB_CONTEXT_WINDOW)
    gt_slice = ordered.iloc[anchor_pos + 1 : anchor_pos + 1 + len(forecasts)]

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_slice.to_dict("records"),
    )

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))
    payload["trade_bot"] = serialise_trade_report(trade_report)

    return jsonify(payload)


@app.route("/api/xgb/forecast")
def api_xgb_forecast():
    mode = request.args.get("mode", "random").lower()
    steps = _extract_minutes(request.args)

    with data_lock:
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if df.empty:
        return jsonify({"error": "No candle data available."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503

    if mode not in {"latest", "random"}:
        return jsonify({"error": f"Unsupported forecast mode: {mode}"}), 400

    if mode == "latest":
        anchor_idx = len(prepared) - 1
    else:
        if len(prepared) < steps + 2:
            return jsonify({"error": "Not enough historical data for random forecast."}), 503
        anchor_idx = random.randint(0, len(prepared) - steps - 2)

    anchor_row = prepared.iloc[anchor_idx]
    history = prepared.iloc[: anchor_idx + 1]

    _ensure_xgb_model_loaded()
    if xgb_model is None:
        return jsonify({"error": "XGBoost model is unavailable."}), 503

    forecasts = _forecast_candles(
        xgb_model,
        history,
        steps,
        feature_order=XGB_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    ground_truth: List[Dict[str, object]] = []
    gt_records: List[Dict[str, object]] = []
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        gt_records = gt_slice.to_dict("records")
        ground_truth = _serialise_candles(gt_records)

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(XGB_CONTEXT_WINDOW)

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_records,
    )

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
            "trade_bot": serialise_trade_report(trade_report),
        }
    )


@app.route("/api/lstm/sample")
def api_lstm_sample():
    mode = request.args.get("mode", "random").lower()
    horizon_minutes = _extract_minutes(request.args)

    _ensure_lstm_model_loaded()

    with data_lock:
        model_available = lstm_model is not None
        samples = [sample.copy() for sample in lstm_samples]
        latest = lstm_latest_sample.copy() if lstm_latest_sample else None
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if not model_available:
        return jsonify({"error": "LSTM model is unavailable."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503
    ordered = prepared.sort_values("timestamp").reset_index(drop=True)

    if mode == "latest":
        if latest is None:
            return jsonify({"error": "Latest sample unavailable."}), 404
        payload = latest.copy()
        payload["mode"] = "latest"
    else:
        if not samples:
            return jsonify({"error": "No evaluation samples available."}), 404
        payload = random.choice(samples)
        payload["mode"] = "random"

    anchor_ts = int(payload.get("anchor_timestamp", payload.get("timestamp", 0)))
    matches = ordered.index[ordered["timestamp"] == anchor_ts]
    if matches.empty:
        return jsonify({"error": "Anchor candle unavailable in dataset."}), 404

    anchor_pos = int(matches[-1])
    history_full = ordered.iloc[: anchor_pos + 1].copy()
    if history_full.empty:
        return jsonify({"error": "Not enough history to generate sample context."}), 503

    forecasts = _forecast_candles(
        lstm_model,
        history_full,
        horizon_minutes,
        feature_order=LSTM_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    history_preview = history_full.tail(LSTM_CONTEXT_WINDOW)
    gt_slice = ordered.iloc[anchor_pos + 1 : anchor_pos + 1 + len(forecasts)]

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_slice.to_dict("records"),
    )

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))
    payload["trade_bot"] = serialise_trade_report(trade_report)

    return jsonify(payload)


@app.route("/api/lstm/forecast")
def api_lstm_forecast():
    mode = request.args.get("mode", "random").lower()
    steps = _extract_minutes(request.args)

    with data_lock:
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if df.empty:
        return jsonify({"error": "No candle data available."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503

    if mode not in {"latest", "random"}:
        return jsonify({"error": f"Unsupported forecast mode: {mode}"}), 400

    if mode == "latest":
        anchor_idx = len(prepared) - 1
    else:
        if len(prepared) < steps + 2:
            return jsonify({"error": "Not enough historical data for random forecast."}), 503
        anchor_idx = random.randint(0, len(prepared) - steps - 2)

    anchor_row = prepared.iloc[anchor_idx]
    history = prepared.iloc[: anchor_idx + 1]

    _ensure_lstm_model_loaded()
    if lstm_model is None:
        return jsonify({"error": "LSTM model is unavailable."}), 503

    forecasts = _forecast_candles(
        lstm_model,
        history,
        steps,
        feature_order=LSTM_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    ground_truth: List[Dict[str, object]] = []
    gt_records: List[Dict[str, object]] = []
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        gt_records = gt_slice.to_dict("records")
        ground_truth = _serialise_candles(gt_records)

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(LSTM_CONTEXT_WINDOW)

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_records,
    )

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
            "trade_bot": serialise_trade_report(trade_report),
        }
    )


@app.route("/api/rf/sample")
def api_rf_sample():
    mode = request.args.get("mode", "random").lower()
    horizon_minutes = _extract_minutes(request.args)

    _ensure_rf_model_loaded()

    with data_lock:
        model_available = rf_model is not None
        samples = [sample.copy() for sample in rf_samples]
        latest = rf_latest_sample.copy() if rf_latest_sample else None
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if not model_available:
        return jsonify({"error": "Random forest model is unavailable."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503
    ordered = prepared.sort_values("timestamp").reset_index(drop=True)

    if mode == "latest":
        if latest is None:
            return jsonify({"error": "Latest sample unavailable."}), 404
        payload = latest.copy()
        payload["mode"] = "latest"
    else:
        if not samples:
            return jsonify({"error": "No evaluation samples available."}), 404
        payload = random.choice(samples)
        payload["mode"] = "random"

    anchor_ts = int(payload.get("anchor_timestamp", payload.get("timestamp", 0)))
    matches = ordered.index[ordered["timestamp"] == anchor_ts]
    if matches.empty:
        return jsonify({"error": "Anchor candle unavailable in dataset."}), 404

    anchor_pos = int(matches[-1])
    history_full = ordered.iloc[: anchor_pos + 1].copy()
    if history_full.empty:
        return jsonify({"error": "Not enough history to generate sample context."}), 503

    forecasts = _forecast_candles(
        rf_model,
        history_full,
        horizon_minutes,
        feature_order=RF_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    history_preview = history_full.tail(RF_CONTEXT_WINDOW)
    gt_slice = ordered.iloc[anchor_pos + 1 : anchor_pos + 1 + len(forecasts)]

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_slice.to_dict("records"),
    )

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))
    payload["trade_bot"] = serialise_trade_report(trade_report)

    return jsonify(payload)


@app.route("/api/rf/forecast")
def api_rf_forecast():
    mode = request.args.get("mode", "random").lower()
    steps = _extract_minutes(request.args)

    with data_lock:
        df = candles_df.copy() if candles_df is not None else pd.DataFrame()

    if df.empty:
        return jsonify({"error": "No candle data available."}), 503

    prepared = _ensure_candles_ready(df)
    if prepared.empty:
        return jsonify({"error": "No candle data available."}), 503

    if mode not in {"latest", "random"}:
        return jsonify({"error": f"Unsupported forecast mode: {mode}"}), 400

    if mode == "latest":
        anchor_idx = len(prepared) - 1
    else:
        if len(prepared) < steps + 2:
            return jsonify({"error": "Not enough historical data for random forecast."}), 503
        anchor_idx = random.randint(0, len(prepared) - steps - 2)

    anchor_row = prepared.iloc[anchor_idx]
    history = prepared.iloc[: anchor_idx + 1]

    _ensure_rf_model_loaded()
    if rf_model is None:
        return jsonify({"error": "Random forest model is unavailable."}), 503

    forecasts = _forecast_candles(
        rf_model,
        history,
        steps,
        feature_order=RF_FEATURE_ORDER,
    )
    if not forecasts:
        return jsonify({"error": "Unable to generate forecast."}), 500

    ground_truth: List[Dict[str, object]] = []
    gt_records: List[Dict[str, object]] = []
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        gt_records = gt_slice.to_dict("records")
        ground_truth = _serialise_candles(gt_records)

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(RF_CONTEXT_WINDOW)

    trade_report = generate_trade_report(
        forecasts,
        ground_truth=gt_records,
    )

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
            "trade_bot": serialise_trade_report(trade_report),
        }
    )


@app.route("/api/analytics/simulation")
def api_analytics_simulation() -> Tuple[Any, int] | Any:
    model = request.args.get("model", "logreg").lower()
    mode = request.args.get("mode", "cumulative").lower()
    try:
        initial_cash = float(request.args.get("initial_cash", DEFAULT_ANALYTICS_INITIAL_CASH))
    except ValueError:
        initial_cash = DEFAULT_ANALYTICS_INITIAL_CASH
    try:
        bet_size = float(request.args.get("bet_size", DEFAULT_ANALYTICS_BET_SIZE))
    except ValueError:
        bet_size = DEFAULT_ANALYTICS_BET_SIZE
    try:
        transaction_fee = float(
            request.args.get("transaction_fee", DEFAULT_ANALYTICS_TRANSACTION_FEE)
        )
    except ValueError:
        transaction_fee = DEFAULT_ANALYTICS_TRANSACTION_FEE
    try:
        buy_confidence = float(
            request.args.get("buy_confidence", DEFAULT_ANALYTICS_BUY_CONFIDENCE)
        )
    except ValueError:
        buy_confidence = DEFAULT_ANALYTICS_BUY_CONFIDENCE

    try:
        dataset = _load_analytics_dataset()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    if dataset.empty:
        return jsonify({"error": "Test dataset is empty."}), 503

    try:
        frame = _build_analytics_frame(dataset, model)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    if frame.empty:
        return jsonify({"error": "Not enough data to run the simulation."}), 503

    try:
        simulation = _simulate_trades(
            frame,
            mode,
            initial_cash,
            bet_size,
            transaction_fee,
            buy_confidence,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    price_series = _serialise_price_series(frame, ANALYTICS_MAX_POINTS)
    equity_curve = _serialise_equity_curve(
        simulation.get("equity_curve", []), ANALYTICS_MAX_POINTS
    )

    trades_payload = []
    for trade in simulation.get("trades", []):
        trades_payload.append(
            {
                "entry_timestamp": int(trade["entry_timestamp"]),
                "entry_datetime": trade["entry_datetime"],
                "entry_price": float(trade["entry_price"]),
                "exit_timestamp": int(trade.get("exit_timestamp", trade["entry_timestamp"])),
                "exit_datetime": trade.get("exit_datetime", trade["entry_datetime"]),
                "exit_price": float(trade.get("exit_price", trade["entry_price"])),
                "pnl": float(trade.get("pnl", 0.0)),
                "return_pct": float(trade.get("return_pct", 0.0)),
                "duration_minutes": float(trade.get("duration_minutes", 0.0)),
                "stake": float(trade.get("stake", 0.0)),
                "units": float(trade.get("units", 0.0)),
            }
        )

    buy_markers = [
        {
            "timestamp": int(marker["timestamp"]),
            "datetime": marker["datetime"],
            "price": float(marker["price"]),
        }
        for marker in simulation.get("buys", [])
    ]
    sell_markers = [
        {
            "timestamp": int(marker["timestamp"]),
            "datetime": marker["datetime"],
            "price": float(marker["price"]),
        }
        for marker in simulation.get("sells", [])
    ]

    response = {
        "model": model,
        "mode": mode,
        "initial_cash": float(initial_cash),
        "bet_size": float(bet_size),
        "transaction_fee": float(transaction_fee),
        "buy_confidence": float(buy_confidence),
        "stats": simulation.get("stats", {}),
        "trades": trades_payload,
        "price_series": price_series,
        "equity_curve": equity_curve,
        "signals": {"buys": buy_markers, "sells": sell_markers},
        "data_points": int(len(frame)),
        "sampled_points": int(len(price_series)),
    }

    return jsonify(response)


@app.route("/")
def index():
    template = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Bitcoin Data Exploration</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
          :root { color-scheme: dark; }
          body { font-family: 'Inter', sans-serif; margin: 0; background: #0f172a; color: #f1f5f9; }
          header { padding: 1.5rem 2rem; background: #111827; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
          h1 { margin: 0; font-size: 1.6rem; }
          main { padding: 1rem 2rem 2rem; }
          button { background: #2563eb; border: none; border-radius: 999px; padding: 0.6rem 1.2rem; color: #f8fafc; font-weight: 600; cursor: pointer; transition: background 0.2s ease, opacity 0.2s ease; }
          button:hover { background: #1d4ed8; }
          button:disabled { opacity: 0.6; cursor: not-allowed; }
          .tabs { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
          .tab-button { background: #1e293b; border: 1px solid rgba(148,163,184,0.3); color: #e2e8f0; }
          .tab-button.active { background: #2563eb; border-color: #60a5fa; }
          .controls { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
          .interval-button.active { background: #22c55e; }
          #chart { width: 100%; height: 70vh; }
          .status { margin-top: 0.75rem; font-size: 0.9rem; color: #cbd5f5; }
          .tab-content { display: none; }
          .tab-content.active { display: block; }
          .analytics-controls { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; align-items: flex-end; }
          .analytics-control { display: flex; flex-direction: column; gap: 0.35rem; min-width: 160px; }
          .analytics-control label { font-size: 0.85rem; color: #cbd5f5; }
          .analytics-control-title { font-size: 0.85rem; color: #cbd5f5; margin-bottom: 0.35rem; display: block; }
          .analytics-control input, .analytics-control select { background: #0f172a; color: #e2e8f0; border: 1px solid rgba(148,163,184,0.4); border-radius: 0.5rem; padding: 0.5rem 0.75rem; font-size: 0.95rem; }
          .analytics-mode-group { display: flex; gap: 1rem; align-items: center; }
          .analytics-mode-group label { display: flex; align-items: center; gap: 0.35rem; font-size: 0.9rem; cursor: pointer; }
          .analytics-status { margin-bottom: 1.5rem; color: #cbd5f5; font-size: 0.95rem; }
          .analytics-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
          .analytics-card { background: #1e293b; border: 1px solid rgba(148,163,184,0.25); border-radius: 0.75rem; padding: 1rem; }
          .analytics-card h3 { margin: 0 0 0.35rem 0; font-size: 0.95rem; color: #bfdbfe; }
          .analytics-card p { margin: 0; font-size: 1.15rem; font-weight: 600; color: #f8fafc; }
          .analytics-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }
          .analytics-chart { background: #0f172a; border: 1px solid rgba(148,163,184,0.25); border-radius: 0.75rem; padding: 0.75rem; }
          .analytics-chart h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #bfdbfe; }
          .analytics-table-wrapper { overflow-x: auto; border: 1px solid rgba(148,163,184,0.25); border-radius: 0.75rem; }
          table.analytics-table { width: 100%; border-collapse: collapse; min-width: 640px; }
          table.analytics-table thead { background: rgba(15, 23, 42, 0.75); }
          table.analytics-table th, table.analytics-table td { padding: 0.65rem 0.85rem; text-align: left; border-bottom: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          table.analytics-table tbody tr:nth-child(even) { background: rgba(30, 41, 59, 0.4); }
          table.analytics-table tbody tr:hover { background: rgba(59, 130, 246, 0.15); }
          .analytics-empty { padding: 1rem; text-align: center; color: #94a3b8; }
          .logreg-controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
          .logreg-card { background: #111827; border-radius: 1rem; padding: 1.5rem; border: 1px solid rgba(148,163,184,0.2); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35); }
          .logreg-card h2 { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.3rem; }
          .logreg-time { margin: 0; font-size: 0.95rem; color: #cbd5f5; }
          .logreg-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.25rem 0; }
          .logreg-metrics div { background: #0f172a; border-radius: 0.9rem; padding: 0.9rem 1rem; border: 1px solid rgba(148,163,184,0.2); }
          .logreg-metrics h3 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #cbd5f5; }
          .logreg-metrics .value { margin: 0; font-size: 1.4rem; font-weight: 600; }
          .result-up { color: #22c55e; }
          .result-down { color: #f97316; }
          .logreg-features h3 { margin: 0 0 0.6rem 0; }
          .logreg-features ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.6rem; }
          .logreg-features li { background: #0f172a; border-radius: 0.8rem; padding: 0.6rem 0.8rem; border: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          .logreg-sample-chart { margin-top: 1.5rem; }
          .logreg-sample-chart h3 { margin: 0 0 0.6rem 0; font-size: 1rem; }
          #logreg-sample-chart { width: 100%; height: 45vh; }
          .trade-bot { margin-top: 1.5rem; background: #0f172a; border-radius: 0.9rem; padding: 1rem 1.25rem; border: 1px solid rgba(148,163,184,0.2); }
          .trade-bot h3 { margin: 0 0 0.6rem 0; font-size: 1rem; }
          .trade-bot-summary { margin: 0 0 0.8rem 0; font-size: 0.95rem; color: #cbd5f5; }
          .trade-bot-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin-bottom: 0.75rem; }
          .trade-bot-metrics p { margin: 0; font-size: 0.95rem; }
          .trade-bot-orders h4 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #cbd5f5; }
          .trade-bot-orders ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.5rem; }
          .trade-bot-orders li { background: #111827; border-radius: 0.7rem; padding: 0.6rem 0.8rem; border: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          .trade-profit { color: #22c55e; }
          .trade-loss { color: #ef4444; }
          .toggle { display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; user-select: none; }
          .toggle input { width: 1.1rem; height: 1.1rem; }
          .logreg-forecast { margin-top: 2rem; }
          .xgb-controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
          .xgb-card { background: #111827; border-radius: 1rem; padding: 1.5rem; border: 1px solid rgba(148,163,184,0.2); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35); }
          .xgb-card h2 { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.3rem; }
          .xgb-time { margin: 0; font-size: 0.95rem; color: #cbd5f5; }
          .xgb-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.25rem 0; }
          .xgb-metrics div { background: #0f172a; border-radius: 0.9rem; padding: 0.9rem 1rem; border: 1px solid rgba(148,163,184,0.2); }
          .xgb-metrics h3 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #cbd5f5; }
          .xgb-metrics .value { margin: 0; font-size: 1.4rem; font-weight: 600; }
          .xgb-features h3 { margin: 0 0 0.6rem 0; }
          .xgb-features ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.6rem; }
          .xgb-features li { background: #0f172a; border-radius: 0.8rem; padding: 0.6rem 0.8rem; border: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          .xgb-sample-chart { margin-top: 1.5rem; }
          .xgb-sample-chart h3 { margin: 0 0 0.6rem 0; font-size: 1rem; }
          #xgb-sample-chart { width: 100%; height: 45vh; }
          .xgb-forecast { margin-top: 2rem; }
          #xgb-forecast-chart { width: 100%; height: 60vh; }
          .lstm-controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
          .lstm-card { background: #111827; border-radius: 1rem; padding: 1.5rem; border: 1px solid rgba(148,163,184,0.2); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35); }
          .lstm-card h2 { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.3rem; }
          .lstm-time { margin: 0; font-size: 0.95rem; color: #cbd5f5; }
          .lstm-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.25rem 0; }
          .lstm-metrics div { background: #0f172a; border-radius: 0.9rem; padding: 0.9rem 1rem; border: 1px solid rgba(148,163,184,0.2); }
          .lstm-metrics h3 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #cbd5f5; }
          .lstm-metrics .value { margin: 0; font-size: 1.4rem; font-weight: 600; }
          .lstm-features h3 { margin: 0 0 0.6rem 0; }
          .lstm-features ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.6rem; }
          .lstm-features li { background: #0f172a; border-radius: 0.8rem; padding: 0.6rem 0.8rem; border: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          .lstm-sample-chart { margin-top: 1.5rem; }
          .lstm-sample-chart h3 { margin: 0 0 0.6rem 0; font-size: 1rem; }
          #lstm-sample-chart { width: 100%; height: 45vh; }
          .lstm-forecast { margin-top: 2rem; }
          #lstm-forecast-chart { width: 100%; height: 60vh; }
          .forecast-controls { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }
          .forecast-controls label { display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; }
          .forecast-controls input { background: #0f172a; border: 1px solid rgba(148,163,184,0.3); border-radius: 0.5rem; padding: 0.45rem 0.6rem; color: #e2e8f0; width: 5.5rem; }
          .forecast-controls input:focus { outline: none; border-color: #60a5fa; box-shadow: 0 0 0 1px rgba(96,165,250,0.4); }
          #forecast-chart { width: 100%; height: 60vh; }
          .rf-controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
          .rf-card { background: #111827; border-radius: 1rem; padding: 1.5rem; border: 1px solid rgba(148,163,184,0.2); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35); }
          .rf-card h2 { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.3rem; }
          .rf-time { margin: 0; font-size: 0.95rem; color: #cbd5f5; }
          .rf-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.25rem 0; }
          .rf-metrics div { background: #0f172a; border-radius: 0.9rem; padding: 0.9rem 1rem; border: 1px solid rgba(148,163,184,0.2); }
          .rf-metrics h3 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #cbd5f5; }
          .rf-metrics .value { margin: 0; font-size: 1.4rem; font-weight: 600; }
          .rf-features h3 { margin: 0 0 0.6rem 0; }
          .rf-features ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.6rem; }
          .rf-features li { background: #0f172a; border-radius: 0.8rem; padding: 0.6rem 0.8rem; border: 1px solid rgba(148,163,184,0.2); font-size: 0.9rem; }
          .rf-sample-chart { margin-top: 1.5rem; }
          .rf-sample-chart h3 { margin: 0 0 0.6rem 0; font-size: 1rem; }
          #rf-sample-chart { width: 100%; height: 45vh; }
          .rf-forecast { margin-top: 2rem; }
        </style>
      </head>
      <body>
        <header>
          <h1>Bitcoin Candle Explorer</h1>
          <p>Up-to-date BTC/USDT candles aggregated from Binance.</p>
        </header>
        <main>
          <div class="tabs">
            <button class="tab-button active" data-tab="candles">Candlestick chart</button>
            <button class="tab-button" data-tab="logreg">Logistic regression</button>
            <button class="tab-button" data-tab="xgb">XGBoost</button>
            <button class="tab-button" data-tab="lstm">LSTM</button>
            <button class="tab-button" data-tab="rf">Random forest</button>
            <button class="tab-button" data-tab="analytics">Analytics</button>
          </div>
          <section id="candles-tab" class="tab-content active">
            <div class="controls">
              <button data-interval="1m" class="interval-button active">1 Minute</button>
              <button data-interval="5m" class="interval-button">5 Minutes</button>
              <button data-interval="15m" class="interval-button">15 Minutes</button>
              <button data-interval="1h" class="interval-button">1 Hour</button>
              <button data-interval="4h" class="interval-button">4 Hours</button>
            </div>
            <div id="chart"></div>
            <div class="status" id="status">Loading...</div>
          </section>
          <section id="analytics-tab" class="tab-content">
            <div class="analytics-controls">
              <div class="analytics-control">
                <label for="analytics-model">Model</label>
                <select id="analytics-model">
                  <option value="logreg">Logistic regression</option>
                  <option value="rf">Random forest</option>
                  <option value="xgb">XGBoost</option>
                  <option value="lstm">LSTM</option>
                </select>
              </div>
              <div class="analytics-control">
                <label for="analytics-initial-cash">Initial capital (USDT)</label>
                <input type="number" id="analytics-initial-cash" min="1" step="10" value="1000">
              </div>
              <div class="analytics-control">
                <span class="analytics-control-title">Simulation mode</span>
                <div class="analytics-mode-group">
                  <label><input type="radio" name="analytics-mode" value="cumulative" checked> Cumulative</label>
                  <label><input type="radio" name="analytics-mode" value="fixed"> Fixed bet</label>
                </div>
              </div>
              <div class="analytics-control" id="analytics-bet-size-group" hidden>
                <label for="analytics-bet-size">Fixed bet size (USDT)</label>
                <input type="number" id="analytics-bet-size" min="0" step="10" value="100">
              </div>
              <div class="analytics-control">
                <label for="analytics-transaction-fee">Transaction fee (%)</label>
                <input type="number" id="analytics-transaction-fee" min="0" step="0.01" value="0">
              </div>
              <div class="analytics-control">
                <label for="analytics-buy-confidence">Buy confidence threshold (%)</label>
                <input type="number" id="analytics-buy-confidence" min="0" step="0.01" value="0">
              </div>
              <button id="analytics-run">Run simulation</button>
            </div>
            <div id="analytics-status" class="analytics-status">
              Select a model and run the simulation to evaluate trade bot performance on the test dataset.
            </div>
            <div class="analytics-stats">
              <div class="analytics-card">
                <h3>Total trades</h3>
                <p id="analytics-stat-trades">0</p>
              </div>
              <div class="analytics-card">
                <h3>Win rate</h3>
                <p id="analytics-stat-winrate">0%</p>
              </div>
              <div class="analytics-card">
                <h3>Total P/L</h3>
                <p id="analytics-stat-pnl">0 USDT</p>
              </div>
              <div class="analytics-card">
                <h3>Return</h3>
                <p id="analytics-stat-return">0%</p>
              </div>
              <div class="analytics-card">
                <h3>Max drawdown</h3>
                <p id="analytics-stat-drawdown">0%</p>
              </div>
            </div>
            <div class="analytics-charts">
              <div class="analytics-chart">
                <h3>Price with trade signals</h3>
                <div id="analytics-price-chart"></div>
              </div>
              <div class="analytics-chart">
                <h3>Portfolio value</h3>
                <div id="analytics-equity-chart"></div>
              </div>
            </div>
            <div class="analytics-table-wrapper">
              <table class="analytics-table">
                <thead>
                  <tr>
                    <th>Entry time</th>
                    <th>Exit time</th>
                    <th>Stake (USDT)</th>
                    <th>P/L (USDT)</th>
                    <th>Return %</th>
                    <th>Duration (min)</th>
                  </tr>
                </thead>
                <tbody id="analytics-trades-body">
                  <tr><td colspan="6" class="analytics-empty">Run the simulation to list trades.</td></tr>
                </tbody>
              </table>
            </div>
          </section>
          <section id="logreg-tab" class="tab-content">
            <div class="logreg-controls">
              <button id="logreg-refresh">Pick random sample</button>
              <label class="toggle">
                <input type="checkbox" id="logreg-latest-toggle">
                <span>Use latest candle</span>
              </label>
            </div>
            <div class="status" id="logreg-status">Activate the tab to load logistic regression insights.</div>
            <div id="logreg-card" class="logreg-card" hidden>
              <h2>Logistic regression signal</h2>
              <p class="logreg-time" id="logreg-time"></p>
              <div class="logreg-metrics">
                <div>
                  <h3>Ground truth</h3>
                  <p id="logreg-ground-truth" class="value"></p>
                </div>
                <div>
                  <h3>Prediction</h3>
                  <p id="logreg-prediction" class="value"></p>
                </div>
                <div>
                  <h3>Up probability</h3>
                  <p id="logreg-probability" class="value"></p>
                </div>
              </div>
              <div class="logreg-features">
                <h3>Feature values</h3>
                <ul id="logreg-feature-list"></ul>
              </div>
              <div class="trade-bot" id="logreg-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="logreg-trade-summary" class="trade-bot-summary">Activate the card to load trade bot insights.</p>
                <div class="trade-bot-metrics">
                  <p id="logreg-trade-state"></p>
                  <p id="logreg-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="logreg-trade-orders"></ul>
                </div>
              </div>
              <div class="logreg-sample-chart">
                <h3>Model input and predicted candles</h3>
                <div id="logreg-sample-chart"></div>
              </div>
            </div>
            <div class="logreg-forecast">
              <div class="forecast-controls">
                <label>
                  Forecast horizon (minutes)
                  <input type="number" id="forecast-steps" min="1" max="120" value="15">
                </label>
                <button id="forecast-run">Run forecast</button>
                <label class="toggle">
                  <input type="checkbox" id="forecast-latest-toggle">
                  <span>Forecast from latest candle</span>
                </label>
              </div>
              <div class="status" id="forecast-status">Set a horizon and generate a forecast to compare predicted candles with historical ground truth.</div>
              <div class="trade-bot" id="forecast-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="forecast-trade-summary" class="trade-bot-summary">Run a forecast to see trade actions.</p>
                <div class="trade-bot-metrics">
                  <p id="forecast-trade-state"></p>
                  <p id="forecast-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="forecast-trade-orders"></ul>
                </div>
              </div>
              <div id="forecast-chart"></div>
            </div>
          </section>
          <section id="xgb-tab" class="tab-content">
            <div class="xgb-controls">
              <button id="xgb-refresh">Pick random sample</button>
              <label class="toggle">
                <input type="checkbox" id="xgb-latest-toggle">
                <span>Use latest candle</span>
              </label>
            </div>
            <div class="status" id="xgb-status">Activate the tab to load XGBoost insights.</div>
            <div id="xgb-card" class="xgb-card" hidden>
              <h2>XGBoost signal</h2>
              <p class="xgb-time" id="xgb-time"></p>
              <div class="xgb-metrics">
                <div>
                  <h3>Ground truth</h3>
                  <p id="xgb-ground-truth" class="value"></p>
                </div>
                <div>
                  <h3>Prediction</h3>
                  <p id="xgb-prediction" class="value"></p>
                </div>
                <div>
                  <h3>Up probability</h3>
                  <p id="xgb-probability" class="value"></p>
                </div>
              </div>
              <div class="xgb-features">
                <h3>Feature values</h3>
                <ul id="xgb-feature-list"></ul>
              </div>
              <div class="trade-bot" id="xgb-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="xgb-trade-summary" class="trade-bot-summary">Activate the card to load trade bot insights.</p>
                <div class="trade-bot-metrics">
                  <p id="xgb-trade-state"></p>
                  <p id="xgb-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="xgb-trade-orders"></ul>
                </div>
              </div>
              <div class="xgb-sample-chart">
                <h3>Model input and predicted candles</h3>
                <div id="xgb-sample-chart"></div>
              </div>
            </div>
            <div class="xgb-forecast">
              <div class="forecast-controls">
                <label>
                  Forecast horizon (minutes)
                  <input type="number" id="xgb-forecast-steps" min="1" max="120" value="15">
                </label>
                <button id="xgb-forecast-run">Run forecast</button>
                <label class="toggle">
                  <input type="checkbox" id="xgb-forecast-latest-toggle">
                  <span>Forecast from latest candle</span>
                </label>
              </div>
              <div class="status" id="xgb-forecast-status">Set a horizon and generate a forecast to compare predicted candles with historical ground truth.</div>
              <div class="trade-bot" id="xgb-forecast-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="xgb-forecast-trade-summary" class="trade-bot-summary">Run a forecast to see trade actions.</p>
                <div class="trade-bot-metrics">
                  <p id="xgb-forecast-trade-state"></p>
                  <p id="xgb-forecast-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="xgb-forecast-trade-orders"></ul>
                </div>
              </div>
              <div id="xgb-forecast-chart"></div>
            </div>
          </section>
          <section id="lstm-tab" class="tab-content">
            <div class="lstm-controls">
              <button id="lstm-refresh">Pick random sample</button>
              <label class="toggle">
                <input type="checkbox" id="lstm-latest-toggle">
                <span>Use latest candle</span>
              </label>
            </div>
            <div class="status" id="lstm-status">Activate the tab to load LSTM insights.</div>
            <div id="lstm-card" class="lstm-card" hidden>
              <h2>LSTM signal</h2>
              <p class="lstm-time" id="lstm-time"></p>
              <div class="lstm-metrics">
                <div>
                  <h3>Ground truth</h3>
                  <p id="lstm-ground-truth" class="value"></p>
                </div>
                <div>
                  <h3>Prediction</h3>
                  <p id="lstm-prediction" class="value"></p>
                </div>
                <div>
                  <h3>Up probability</h3>
                  <p id="lstm-probability" class="value"></p>
                </div>
              </div>
              <div class="lstm-features">
                <h3>Feature values</h3>
                <ul id="lstm-feature-list"></ul>
              </div>
              <div class="trade-bot" id="lstm-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="lstm-trade-summary" class="trade-bot-summary">Activate the card to load trade bot insights.</p>
                <div class="trade-bot-metrics">
                  <p id="lstm-trade-state"></p>
                  <p id="lstm-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="lstm-trade-orders"></ul>
                </div>
              </div>
              <div class="lstm-sample-chart">
                <h3>Model input and predicted candles</h3>
                <div id="lstm-sample-chart"></div>
              </div>
            </div>
            <div class="lstm-forecast">
              <div class="forecast-controls">
                <label>
                  Forecast horizon (minutes)
                  <input type="number" id="lstm-forecast-steps" min="1" max="120" value="15">
                </label>
                <button id="lstm-forecast-run">Run forecast</button>
                <label class="toggle">
                  <input type="checkbox" id="lstm-forecast-latest-toggle">
                  <span>Forecast from latest candle</span>
                </label>
              </div>
              <div class="status" id="lstm-forecast-status">Set a horizon and generate a forecast to compare predicted candles with historical ground truth.</div>
              <div class="trade-bot" id="lstm-forecast-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="lstm-forecast-trade-summary" class="trade-bot-summary">Run a forecast to see trade actions.</p>
                <div class="trade-bot-metrics">
                  <p id="lstm-forecast-trade-state"></p>
                  <p id="lstm-forecast-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="lstm-forecast-trade-orders"></ul>
                </div>
              </div>
              <div id="lstm-forecast-chart"></div>
            </div>
          </section>
          <section id="rf-tab" class="tab-content">
            <div class="rf-controls">
              <button id="rf-refresh">Pick random sample</button>
              <label class="toggle">
                <input type="checkbox" id="rf-latest-toggle">
                <span>Use latest candle</span>
              </label>
            </div>
            <div class="status" id="rf-status">Activate the tab to load random forest insights.</div>
            <div id="rf-card" class="rf-card" hidden>
              <h2>Random forest signal</h2>
              <p class="rf-time" id="rf-time"></p>
              <div class="rf-metrics">
                <div>
                  <h3>Ground truth</h3>
                  <p id="rf-ground-truth" class="value"></p>
                </div>
                <div>
                  <h3>Prediction</h3>
                  <p id="rf-prediction" class="value"></p>
                </div>
                <div>
                  <h3>Up probability</h3>
                  <p id="rf-probability" class="value"></p>
                </div>
              </div>
              <div class="rf-features">
                <h3>Feature values</h3>
                <ul id="rf-feature-list"></ul>
              </div>
              <div class="trade-bot" id="rf-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="rf-trade-summary" class="trade-bot-summary">Activate the card to load trade bot insights.</p>
                <div class="trade-bot-metrics">
                  <p id="rf-trade-state"></p>
                  <p id="rf-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="rf-trade-orders"></ul>
                </div>
              </div>
              <div class="rf-sample-chart">
                <h3>Model input and predicted candles</h3>
                <div id="rf-sample-chart"></div>
              </div>
            </div>
            <div class="rf-forecast">
              <div class="forecast-controls">
                <label>
                  Forecast horizon (minutes)
                  <input type="number" id="rf-forecast-steps" min="1" max="120" value="15">
                </label>
                <button id="rf-forecast-run">Run forecast</button>
                <label class="toggle">
                  <input type="checkbox" id="rf-forecast-latest-toggle">
                  <span>Forecast from latest candle</span>
                </label>
              </div>
              <div class="status" id="rf-forecast-status">Set a horizon and generate a forecast to compare predicted candles with historical ground truth.</div>
              <div class="trade-bot" id="rf-forecast-trade" hidden>
                <h3>Trade bot recommendation</h3>
                <p id="rf-forecast-trade-summary" class="trade-bot-summary">Run a forecast to see trade actions.</p>
                <div class="trade-bot-metrics">
                  <p id="rf-forecast-trade-state"></p>
                  <p id="rf-forecast-trade-performance"></p>
                </div>
                <div class="trade-bot-orders">
                  <h4>Orders</h4>
                  <ul id="rf-forecast-trade-orders"></ul>
                </div>
              </div>
              <div id="rf-forecast-chart"></div>
            </div>
          </section>
        </main>
        <script>
          const tabButtons = document.querySelectorAll('.tab-button');
          const tabContents = document.querySelectorAll('.tab-content');
          const intervalButtons = document.querySelectorAll('button[data-interval]');
          const logregRefreshButton = document.getElementById('logreg-refresh');
          const logregToggle = document.getElementById('logreg-latest-toggle');
          const logregStatus = document.getElementById('logreg-status');
          const logregCard = document.getElementById('logreg-card');
          const logregTime = document.getElementById('logreg-time');
          const logregGroundTruth = document.getElementById('logreg-ground-truth');
          const logregPrediction = document.getElementById('logreg-prediction');
          const logregProbability = document.getElementById('logreg-probability');
          const logregFeatureList = document.getElementById('logreg-feature-list');
          const logregSampleChart = document.getElementById('logreg-sample-chart');
          const tradeUi = {
            logregSample: {
              container: document.getElementById('logreg-trade'),
              summary: document.getElementById('logreg-trade-summary'),
              state: document.getElementById('logreg-trade-state'),
              performance: document.getElementById('logreg-trade-performance'),
              orders: document.getElementById('logreg-trade-orders'),
            },
            logregForecast: {
              container: document.getElementById('forecast-trade'),
              summary: document.getElementById('forecast-trade-summary'),
              state: document.getElementById('forecast-trade-state'),
              performance: document.getElementById('forecast-trade-performance'),
              orders: document.getElementById('forecast-trade-orders'),
            },
            xgbSample: {
              container: document.getElementById('xgb-trade'),
              summary: document.getElementById('xgb-trade-summary'),
              state: document.getElementById('xgb-trade-state'),
              performance: document.getElementById('xgb-trade-performance'),
              orders: document.getElementById('xgb-trade-orders'),
            },
            xgbForecast: {
              container: document.getElementById('xgb-forecast-trade'),
              summary: document.getElementById('xgb-forecast-trade-summary'),
              state: document.getElementById('xgb-forecast-trade-state'),
              performance: document.getElementById('xgb-forecast-trade-performance'),
              orders: document.getElementById('xgb-forecast-trade-orders'),
            },
            lstmSample: {
              container: document.getElementById('lstm-trade'),
              summary: document.getElementById('lstm-trade-summary'),
              state: document.getElementById('lstm-trade-state'),
              performance: document.getElementById('lstm-trade-performance'),
              orders: document.getElementById('lstm-trade-orders'),
            },
            lstmForecast: {
              container: document.getElementById('lstm-forecast-trade'),
              summary: document.getElementById('lstm-forecast-trade-summary'),
              state: document.getElementById('lstm-forecast-trade-state'),
              performance: document.getElementById('lstm-forecast-trade-performance'),
              orders: document.getElementById('lstm-forecast-trade-orders'),
            },
            rfSample: {
              container: document.getElementById('rf-trade'),
              summary: document.getElementById('rf-trade-summary'),
              state: document.getElementById('rf-trade-state'),
              performance: document.getElementById('rf-trade-performance'),
              orders: document.getElementById('rf-trade-orders'),
            },
            rfForecast: {
              container: document.getElementById('rf-forecast-trade'),
              summary: document.getElementById('rf-forecast-trade-summary'),
              state: document.getElementById('rf-forecast-trade-state'),
              performance: document.getElementById('rf-forecast-trade-performance'),
              orders: document.getElementById('rf-forecast-trade-orders'),
            },
          };
          const forecastRunButton = document.getElementById('forecast-run');
          const forecastStepsInput = document.getElementById('forecast-steps');
          const forecastStatus = document.getElementById('forecast-status');
          const forecastToggle = document.getElementById('forecast-latest-toggle');
          const xgbRefreshButton = document.getElementById('xgb-refresh');
          const xgbToggle = document.getElementById('xgb-latest-toggle');
          const xgbStatus = document.getElementById('xgb-status');
          const xgbCard = document.getElementById('xgb-card');
          const xgbTime = document.getElementById('xgb-time');
          const xgbGroundTruth = document.getElementById('xgb-ground-truth');
          const xgbPrediction = document.getElementById('xgb-prediction');
          const xgbProbability = document.getElementById('xgb-probability');
          const xgbFeatureList = document.getElementById('xgb-feature-list');
          const xgbSampleChart = document.getElementById('xgb-sample-chart');
          const xgbForecastRunButton = document.getElementById('xgb-forecast-run');
          const xgbForecastStepsInput = document.getElementById('xgb-forecast-steps');
          const xgbForecastStatus = document.getElementById('xgb-forecast-status');
          const xgbForecastToggle = document.getElementById('xgb-forecast-latest-toggle');
          const xgbForecastChart = document.getElementById('xgb-forecast-chart');
          const lstmRefreshButton = document.getElementById('lstm-refresh');
          const lstmToggle = document.getElementById('lstm-latest-toggle');
          const lstmStatus = document.getElementById('lstm-status');
          const lstmCard = document.getElementById('lstm-card');
          const lstmTime = document.getElementById('lstm-time');
          const lstmGroundTruth = document.getElementById('lstm-ground-truth');
          const lstmPrediction = document.getElementById('lstm-prediction');
          const lstmProbability = document.getElementById('lstm-probability');
          const lstmFeatureList = document.getElementById('lstm-feature-list');
          const lstmSampleChart = document.getElementById('lstm-sample-chart');
          const lstmForecastRunButton = document.getElementById('lstm-forecast-run');
          const lstmForecastStepsInput = document.getElementById('lstm-forecast-steps');
          const lstmForecastStatus = document.getElementById('lstm-forecast-status');
          const lstmForecastToggle = document.getElementById('lstm-forecast-latest-toggle');
          const lstmForecastChart = document.getElementById('lstm-forecast-chart');
          const rfRefreshButton = document.getElementById('rf-refresh');
          const rfToggle = document.getElementById('rf-latest-toggle');
          const rfStatus = document.getElementById('rf-status');
          const rfCard = document.getElementById('rf-card');
          const rfTime = document.getElementById('rf-time');
          const rfGroundTruth = document.getElementById('rf-ground-truth');
          const rfPrediction = document.getElementById('rf-prediction');
          const rfProbability = document.getElementById('rf-probability');
          const rfFeatureList = document.getElementById('rf-feature-list');
          const rfSampleChart = document.getElementById('rf-sample-chart');
          const rfForecastRunButton = document.getElementById('rf-forecast-run');
          const rfForecastStepsInput = document.getElementById('rf-forecast-steps');
          const rfForecastStatus = document.getElementById('rf-forecast-status');
          const rfForecastToggle = document.getElementById('rf-forecast-latest-toggle');
          const rfForecastChart = document.getElementById('rf-forecast-chart');
          const analyticsModelSelect = document.getElementById('analytics-model');
          const analyticsInitialCashInput = document.getElementById('analytics-initial-cash');
          const analyticsBetSizeGroup = document.getElementById('analytics-bet-size-group');
          const analyticsBetSizeInput = document.getElementById('analytics-bet-size');
          const analyticsTransactionFeeInput = document.getElementById('analytics-transaction-fee');
          const analyticsBuyConfidenceInput = document.getElementById('analytics-buy-confidence');
          const analyticsModeRadios = document.querySelectorAll('input[name="analytics-mode"]');
          const analyticsRunButton = document.getElementById('analytics-run');
          const analyticsStatus = document.getElementById('analytics-status');
          const analyticsPriceChart = document.getElementById('analytics-price-chart');
          const analyticsEquityChart = document.getElementById('analytics-equity-chart');
          const analyticsTradesBody = document.getElementById('analytics-trades-body');
          const analyticsStats = {
            trades: document.getElementById('analytics-stat-trades'),
            winRate: document.getElementById('analytics-stat-winrate'),
            pnl: document.getElementById('analytics-stat-pnl'),
            returnPct: document.getElementById('analytics-stat-return'),
            drawdown: document.getElementById('analytics-stat-drawdown'),
          };
          let currentInterval = '1m';
          let logregHasLoaded = false;
          let xgbHasLoaded = false;
          let lstmHasLoaded = false;
          let rfHasLoaded = false;
          let analyticsHasLoaded = false;

          tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
              const target = btn.dataset.tab;
              tabButtons.forEach(b => b.classList.toggle('active', b === btn));
              tabContents.forEach(section => section.classList.toggle('active', section.id === `${target}-tab`));
              if (target === 'logreg' && !logregHasLoaded) {
                updateLogregControls();
                loadLogregSample();
                runForecast(true);
                logregHasLoaded = true;
              } else if (target === 'xgb' && !xgbHasLoaded) {
                updateXgbControls();
                loadXgbSample();
                runXgbForecast(true);
                xgbHasLoaded = true;
              } else if (target === 'lstm' && !lstmHasLoaded) {
                updateLstmControls();
                loadLstmSample();
              runLstmForecast(true);
              lstmHasLoaded = true;
            } else if (target === 'rf' && !rfHasLoaded) {
              updateRfControls();
              loadRfSample();
              runRfForecast(true);
              rfHasLoaded = true;
            } else if (target === 'analytics' && !analyticsHasLoaded) {
              updateAnalyticsControls();
              runAnalyticsSimulation(true);
              analyticsHasLoaded = true;
            }
          });
        });

          intervalButtons.forEach(btn => {
            btn.addEventListener('click', () => {
              currentInterval = btn.dataset.interval;
              intervalButtons.forEach(b => b.classList.toggle('active', b === btn));
              fetchAndRender();
            });
          });

          function updateLogregControls() {
            if (!logregRefreshButton) return;
            const useLatest = logregToggle && logregToggle.checked;
            logregRefreshButton.disabled = !!useLatest;
            if (logregRefreshButton.disabled) {
              logregRefreshButton.title = 'Disable the toggle to draw a random test example.';
            } else {
              logregRefreshButton.title = '';
            }
          }

          if (logregRefreshButton) {
            logregRefreshButton.addEventListener('click', () => {
              loadLogregSample();
            });
          }

          if (logregToggle) {
            logregToggle.addEventListener('change', () => {
              updateLogregControls();
              loadLogregSample();
            });
          }

          if (forecastRunButton) {
            forecastRunButton.addEventListener('click', () => {
              runForecast();
            });
          }

          if (forecastToggle) {
            forecastToggle.addEventListener('change', () => {
              runForecast();
            });
          }

          if (forecastStepsInput) {
            forecastStepsInput.addEventListener('keydown', event => {
              if (event.key === 'Enter') {
                event.preventDefault();
                runForecast();
              }
            });
          }

          function updateXgbControls() {
            if (!xgbRefreshButton) return;
            const useLatest = xgbToggle && xgbToggle.checked;
            xgbRefreshButton.disabled = !!useLatest;
            if (xgbRefreshButton.disabled) {
              xgbRefreshButton.title = 'Disable the toggle to draw a random test example.';
            } else {
              xgbRefreshButton.title = '';
            }
          }

          if (xgbRefreshButton) {
            xgbRefreshButton.addEventListener('click', () => {
              loadXgbSample();
            });
          }

          if (xgbToggle) {
            xgbToggle.addEventListener('change', () => {
              updateXgbControls();
              loadXgbSample();
            });
          }

          if (xgbForecastRunButton) {
            xgbForecastRunButton.addEventListener('click', () => {
              runXgbForecast();
            });
          }

          if (xgbForecastToggle) {
            xgbForecastToggle.addEventListener('change', () => {
              runXgbForecast();
            });
          }

          if (xgbForecastStepsInput) {
            xgbForecastStepsInput.addEventListener('keydown', event => {
              if (event.key === 'Enter') {
                event.preventDefault();
                runXgbForecast();
              }
            });
          }

          function updateLstmControls() {
            if (!lstmRefreshButton) return;
            const useLatest = lstmToggle && lstmToggle.checked;
            lstmRefreshButton.disabled = !!useLatest;
            if (lstmRefreshButton.disabled) {
              lstmRefreshButton.title = 'Disable the toggle to draw a random test example.';
            } else {
              lstmRefreshButton.title = '';
            }
          }

          if (lstmRefreshButton) {
            lstmRefreshButton.addEventListener('click', () => {
              loadLstmSample();
            });
          }

          if (lstmToggle) {
            lstmToggle.addEventListener('change', () => {
              updateLstmControls();
              loadLstmSample();
            });
          }

          if (lstmForecastRunButton) {
            lstmForecastRunButton.addEventListener('click', () => {
              runLstmForecast();
            });
          }

          if (lstmForecastToggle) {
            lstmForecastToggle.addEventListener('change', () => {
              runLstmForecast();
            });
          }

          if (lstmForecastStepsInput) {
            lstmForecastStepsInput.addEventListener('keydown', event => {
              if (event.key === 'Enter') {
                event.preventDefault();
                runLstmForecast();
              }
            });
          }

          function updateRfControls() {
            if (!rfRefreshButton) return;
            const useLatest = rfToggle && rfToggle.checked;
            rfRefreshButton.disabled = !!useLatest;
            if (rfRefreshButton.disabled) {
              rfRefreshButton.title = 'Disable the toggle to draw a random test example.';
            } else {
              rfRefreshButton.title = '';
            }
          }

          if (rfRefreshButton) {
            rfRefreshButton.addEventListener('click', () => {
              loadRfSample();
            });
          }

          if (rfToggle) {
            rfToggle.addEventListener('change', () => {
              updateRfControls();
              loadRfSample();
            });
          }

          if (rfForecastRunButton) {
            rfForecastRunButton.addEventListener('click', () => {
              runRfForecast();
            });
          }

          if (rfForecastToggle) {
            rfForecastToggle.addEventListener('change', () => {
              runRfForecast();
            });
          }

          if (analyticsRunButton) {
            analyticsRunButton.addEventListener('click', () => {
              runAnalyticsSimulation();
            });
          }

          analyticsModeRadios.forEach(radio => {
            radio.addEventListener('change', () => {
              updateAnalyticsControls();
            });
          });

          if (rfForecastStepsInput) {
            rfForecastStepsInput.addEventListener('keydown', event => {
              if (event.key === 'Enter') {
                event.preventDefault();
                runRfForecast();
              }
            });
          }

          function formatOutcome(value) {
            if (value === null || value === undefined) {
              return 'Unknown';
            }
            return Number(value) === 1 ? 'Price up' : 'Price down';
          }

          function applyOutcome(element, value) {
            element.textContent = formatOutcome(value);
            element.classList.toggle('result-up', Number(value) === 1);
            element.classList.toggle('result-down', Number(value) === 0);
            if (value === null || value === undefined || Number.isNaN(Number(value))) {
              element.classList.remove('result-up', 'result-down');
            }
          }

          function formatUsd(value, { withSign = true } = {}) {
            const num = Number(value);
            if (!Number.isFinite(num)) {
              return 'N/A';
            }
            const formatted = num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            if (!withSign) {
              return `${formatted} USDT`;
            }
            if (num > 0) {
              return `+${formatted} USDT`;
            }
            return `${formatted} USDT`;
          }

          function formatUnits(value, unit = 'BTC') {
            const num = Number(value);
            if (!Number.isFinite(num)) {
              return `0.0000 ${unit}`;
            }
            return `${num.toFixed(4)} ${unit}`;
          }

          function formatPercent(value) {
            const num = Number(value);
            if (!Number.isFinite(num)) {
              return 'N/A';
            }
            return `${(num * 100).toFixed(1)}%`;
          }

          function renderTradeBotReport(report, elements) {
            if (!elements || !elements.container) {
              return;
            }
            const { container, summary, state, performance, orders } = elements;
            if (!report) {
              container.hidden = true;
              if (summary) {
                summary.textContent = 'Trade bot insights unavailable.';
              }
              if (state) {
                state.textContent = '';
              }
              if (performance) {
                performance.textContent = '';
              }
              if (orders) {
                orders.innerHTML = '';
              }
              return;
            }

            container.hidden = false;
            if (summary) {
              summary.textContent = report.summary || 'Trade bot analysis ready.';
            }

            const finalState = report.final_state || {};
            if (state) {
              const cashText = formatUsd(finalState.cash);
              const positionText = formatUnits(finalState.position);
              const equityText = formatUsd(finalState.total_value, { withSign: false });
              state.textContent = `Cash: ${cashText} · Position: ${positionText} · Equity: ${equityText}`;
            }

            if (performance) {
              const evaluation = report.evaluation || {};
              if (evaluation.available) {
                const realized = Number(evaluation.realized_pl);
                const realizedText = formatUsd(realized);
                const realizedClass = realized >= 0 ? 'trade-profit' : 'trade-loss';
                const totalValue = formatUsd(evaluation.total_value, { withSign: false });
                performance.innerHTML = `Realized P/L: <span class="${realizedClass}">${realizedText}</span> · Total value: ${totalValue}`;
              } else {
                performance.textContent = 'Ground truth not yet available to score this plan.';
              }
            }

            if (orders) {
              orders.innerHTML = '';
              const list = Array.isArray(report.orders) ? report.orders : [];
              if (!list.length) {
                const li = document.createElement('li');
                li.textContent = 'No buy or sell signals — staying in cash.';
                orders.appendChild(li);
              } else {
                list.forEach((order) => {
                  const li = document.createElement('li');
                  const action = typeof order.action === 'string' ? order.action.toUpperCase() : String(order.action || 'HOLD');
                  const dt = order.datetime ? new Date(order.datetime) : null;
                  const stepIndex = Number.isFinite(Number(order.step)) ? Number(order.step) + 1 : '';
                  const when = dt && !Number.isNaN(dt.getTime()) ? dt.toLocaleString() : stepIndex ? `Step ${stepIndex}` : 'Unknown time';
                  const priceText = formatUsd(order.price, { withSign: false });
                  const probText = formatPercent(order.probability);
                  const expected = Number(order.expected_return);
                  const expectedText = Number.isFinite(expected) ? `${(expected * 100).toFixed(2)}% exp. return` : 'Exp. return N/A';
                  li.textContent = `${action} @ ${when} · Price ${priceText} · Prob ${probText} · ${expectedText}`;
                  orders.appendChild(li);
                });
              }
            }
          }

          function getAnalyticsMode() {
            const radios = Array.from(analyticsModeRadios || []);
            const selected = radios.find(radio => radio.checked);
            return selected ? selected.value : 'cumulative';
          }

          function updateAnalyticsControls() {
            const mode = getAnalyticsMode();
            if (analyticsBetSizeGroup) {
              analyticsBetSizeGroup.hidden = mode !== 'fixed';
            }
          }

          function renderAnalytics(data) {
            if (!analyticsStatus) return;
            const stats = data && typeof data === 'object' ? data.stats || {} : {};
            const rawModel = data && data.model ? String(data.model) : '';
            const modelMap = {
              logreg: 'Logistic regression',
              rf: 'Random forest',
              xgb: 'XGBoost',
              lstm: 'LSTM',
            };
            const modelLabel = rawModel ? (modelMap[rawModel.toLowerCase()] || rawModel.toUpperCase()) : 'Analytics';
            const finalEquity = formatUsd(stats.final_equity || 0, { withSign: false });
            const totalPoints = Number(data && data.data_points ? data.data_points : 0);
            const scopeText = totalPoints ? `${totalPoints.toLocaleString()} minutes` : 'test dataset';
            analyticsStatus.textContent = `Simulation complete (${modelLabel}). Final equity ${finalEquity} across ${scopeText}.`;

            if (analyticsStats.trades) {
              const tradesValue = Number(stats.trades || 0);
              analyticsStats.trades.textContent = tradesValue.toLocaleString();
            }
            if (analyticsStats.winRate) {
              analyticsStats.winRate.textContent = formatPercent(stats.win_rate || 0);
            }
            if (analyticsStats.pnl) {
              analyticsStats.pnl.textContent = formatUsd(stats.total_pnl || 0);
            }
            if (analyticsStats.returnPct) {
              analyticsStats.returnPct.textContent = formatPercent(stats.return_pct || 0);
            }
            if (analyticsStats.drawdown) {
              const drawdownValue = Number(stats.max_drawdown || 0);
              analyticsStats.drawdown.textContent = drawdownValue
                ? `-${(drawdownValue * 100).toFixed(1)}%`
                : '0%';
            }

            const priceSeries = Array.isArray(data && data.price_series) ? data.price_series : [];
            const buySignals = data && data.signals && Array.isArray(data.signals.buys) ? data.signals.buys : [];
            const sellSignals = data && data.signals && Array.isArray(data.signals.sells) ? data.signals.sells : [];

            const priceTraces = [];
            if (priceSeries.length) {
              priceTraces.push({
                x: priceSeries.map(point => point.datetime),
                y: priceSeries.map(point => Number(point.price)),
                type: 'scatter',
                mode: 'lines',
                name: 'Close price',
                line: { color: '#38bdf8', width: 1.5 },
              });
            }
            if (buySignals.length) {
              priceTraces.push({
                x: buySignals.map(point => point.datetime),
                y: buySignals.map(point => Number(point.price)),
                mode: 'markers',
                name: 'Buy',
                marker: { color: '#22c55e', size: 7, symbol: 'triangle-up' },
              });
            }
            if (sellSignals.length) {
              priceTraces.push({
                x: sellSignals.map(point => point.datetime),
                y: sellSignals.map(point => Number(point.price)),
                mode: 'markers',
                name: 'Sell',
                marker: { color: '#ef4444', size: 7, symbol: 'triangle-down' },
              });
            }
            const priceLayout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 20, t: 30, b: 50 },
              showlegend: true,
              legend: { orientation: 'h' },
            };
            if (analyticsPriceChart && window.Plotly) {
              Plotly.react('analytics-price-chart', priceTraces, priceLayout, { responsive: true, displaylogo: false });
            }

            const equityCurve = Array.isArray(data && data.equity_curve) ? data.equity_curve : [];
            const equityTrace = {
              x: equityCurve.map(point => point.datetime),
              y: equityCurve.map(point => Number(point.equity)),
              type: 'scatter',
              mode: 'lines',
              name: 'Equity',
              line: { color: '#f97316', width: 1.5 },
            };
            const equityLayout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 20, t: 30, b: 50 },
              showlegend: false,
            };
            if (analyticsEquityChart && window.Plotly) {
              Plotly.react('analytics-equity-chart', [equityTrace], equityLayout, { responsive: true, displaylogo: false });
            }

            if (analyticsTradesBody) {
              analyticsTradesBody.innerHTML = '';
              const trades = Array.isArray(data && data.trades) ? data.trades : [];
              if (!trades.length) {
                analyticsTradesBody.innerHTML = '<tr><td colspan="6" class="analytics-empty">No completed trades in the simulation.</td></tr>';
              } else {
                trades.forEach(trade => {
                  const row = document.createElement('tr');
                  const entryDate = trade.entry_datetime ? new Date(trade.entry_datetime) : null;
                  const exitDate = trade.exit_datetime ? new Date(trade.exit_datetime) : null;

                  const entryCell = document.createElement('td');
                  entryCell.textContent = entryDate && !Number.isNaN(entryDate.getTime()) ? entryDate.toLocaleString() : '—';

                  const exitCell = document.createElement('td');
                  exitCell.textContent = exitDate && !Number.isNaN(exitDate.getTime()) ? exitDate.toLocaleString() : '—';

                  const stakeCell = document.createElement('td');
                  stakeCell.textContent = formatUsd(trade.stake || 0, { withSign: false });

                  const pnlCell = document.createElement('td');
                  const pnlValue = Number(trade.pnl || 0);
                  pnlCell.textContent = formatUsd(pnlValue);
                  pnlCell.className = pnlValue >= 0 ? 'trade-profit' : 'trade-loss';

                  const returnCell = document.createElement('td');
                  const retValue = Number(trade.return_pct || 0);
                  if (Number.isFinite(retValue)) {
                    returnCell.textContent = `${(retValue * 100).toFixed(2)}%`;
                  } else {
                    returnCell.textContent = 'N/A';
                  }
                  returnCell.className = retValue >= 0 ? 'trade-profit' : 'trade-loss';

                  const durationCell = document.createElement('td');
                  const durationValue = Number(trade.duration_minutes || 0);
                  durationCell.textContent = Number.isFinite(durationValue) ? durationValue.toFixed(1) : '0.0';

                  row.append(entryCell, exitCell, stakeCell, pnlCell, returnCell, durationCell);
                  analyticsTradesBody.appendChild(row);
                });
              }
            }
          }

          async function runAnalyticsSimulation(auto = false) {
            if (!analyticsStatus) return;
            const mode = getAnalyticsMode();
            const params = new URLSearchParams();
            const modelValue = analyticsModelSelect ? analyticsModelSelect.value : 'logreg';
            params.set('model', modelValue);
            params.set('mode', mode);

            const initialCashValue = analyticsInitialCashInput ? Number(analyticsInitialCashInput.value) : 1000;
            if (Number.isFinite(initialCashValue) && initialCashValue > 0) {
              params.set('initial_cash', String(initialCashValue));
            }
            if (mode === 'fixed') {
              const betSizeValue = analyticsBetSizeInput ? Number(analyticsBetSizeInput.value) : 100;
              if (Number.isFinite(betSizeValue) && betSizeValue >= 0) {
                params.set('bet_size', String(betSizeValue));
              }
            }
            const transactionFeeValue = analyticsTransactionFeeInput ? Number(analyticsTransactionFeeInput.value) : 0;
            if (Number.isFinite(transactionFeeValue) && transactionFeeValue >= 0) {
              params.set('transaction_fee', String(transactionFeeValue));
            }
            const buyConfidenceValue = analyticsBuyConfidenceInput ? Number(analyticsBuyConfidenceInput.value) : 0;
            if (Number.isFinite(buyConfidenceValue) && buyConfidenceValue >= 0) {
              params.set('buy_confidence', String(buyConfidenceValue));
            }

            analyticsStatus.textContent = 'Running simulation…';
            if (analyticsRunButton) {
              analyticsRunButton.disabled = true;
            }

            try {
              const response = await fetch(`/api/analytics/simulation?${params.toString()}`);
              const payload = await response.json();
              if (!response.ok || (payload && payload.error)) {
                throw new Error(payload && payload.error ? payload.error : `API error ${response.status}`);
              }
              renderAnalytics(payload);
            } catch (error) {
              console.error(error);
              analyticsStatus.textContent = error && error.message ? error.message : 'Failed to run simulation.';
              if (analyticsPriceChart && window.Plotly) {
                Plotly.purge('analytics-price-chart');
              }
              if (analyticsEquityChart && window.Plotly) {
                Plotly.purge('analytics-equity-chart');
              }
              if (analyticsTradesBody) {
                analyticsTradesBody.innerHTML = '<tr><td colspan="6" class="analytics-empty">Simulation unavailable.</td></tr>';
              }
            } finally {
              if (analyticsRunButton) {
                analyticsRunButton.disabled = false;
              }
            }
          }

          function getForecastMinutes() {
            const rawValue = forecastStepsInput ? Number(forecastStepsInput.value) : 15;
            const bounded = Math.max(1, Math.min(Math.round(rawValue) || 15, 120));
            if (forecastStepsInput) {
              forecastStepsInput.value = bounded;
            }
            return bounded;
          }

          async function runForecast(auto = false) {
            if (!forecastStatus) return;
            const bounded = getForecastMinutes();
            const mode = forecastToggle && forecastToggle.checked ? 'latest' : 'random';
            forecastStatus.textContent = mode === 'latest' ? 'Generating forecast from latest candle…' : 'Generating historical forecast sample…';
            if (forecastRunButton) {
              forecastRunButton.disabled = true;
            }
            renderTradeBotReport(null, tradeUi.logregForecast);
            try {
              const response = await fetch(`/api/logreg/forecast?minutes=${bounded}&mode=${mode}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderForecast(payload);
            } catch (error) {
              console.error(error);
              forecastStatus.textContent = error.message || 'Forecast unavailable.';
              if (!auto && window.Plotly) {
                Plotly.purge('forecast-chart');
              }
            } finally {
              if (forecastRunButton) {
                forecastRunButton.disabled = false;
              }
            }
          }

          function renderForecast(data) {
            if (!forecastStatus) return;
            const forecast = Array.isArray(data.forecast) ? data.forecast : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : Array.isArray(data.ground_truth) ? data.ground_truth : [];
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            if (!forecast.length) {
              forecastStatus.textContent = 'Forecast unavailable.';
              if (window.Plotly) {
                Plotly.purge('forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.logregForecast);
              return;
            }

            const anchorDate = data.anchor && data.anchor.datetime ? new Date(data.anchor.datetime) : null;
            const statusParts = [];
            if (anchorDate && !Number.isNaN(anchorDate.getTime())) {
              statusParts.push(`Anchor candle: ${anchorDate.toLocaleString()}`);
            }
            statusParts.push(data.mode === 'latest' ? 'Mode: latest forecast' : 'Mode: historical evaluation');
            if (typeof data.horizon_minutes === 'number') {
              statusParts.push(`Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}`);
            }
            if (groundTruth.length === forecast.length && groundTruth.length) {
              statusParts.push('Ground truth overlay available.');
            } else if (groundTruth.length) {
              statusParts.push('Partial ground truth available.');
            } else {
              statusParts.push('Ground truth unavailable for this horizon.');
            }
            const lastForecast = forecast[forecast.length - 1];
            if (lastForecast && typeof lastForecast.probability === 'number') {
              statusParts.push(`Last step up probability: ${(lastForecast.probability * 100).toFixed(1)}%`);
            }
            forecastStatus.textContent = statusParts.join(' · ');
            renderTradeBotReport(data.trade_bot, tradeUi.logregForecast);

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.5,
              });
            }

            const forecastTrace = {
              x: forecast.map(c => c.datetime),
              open: forecast.map(c => c.open),
              high: forecast.map(c => c.high),
              low: forecast.map(c => c.low),
              close: forecast.map(c => c.close),
              type: 'candlestick',
              name: 'Forecast',
              increasing: { line: { color: '#38bdf8' } },
              decreasing: { line: { color: '#f59e0b' } },
              opacity: 0.65,
            };
            traces.push(forecastTrace);

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 50 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            if (window.Plotly) {
              Plotly.react('forecast-chart', traces, layout, { responsive: true, displaylogo: false });
            }
          }

          async function loadLogregSample() {
            if (!logregStatus) return;
            const useLatest = logregToggle && logregToggle.checked;
            const mode = useLatest ? 'latest' : 'random';
            logregStatus.textContent = useLatest ? 'Fetching latest candle prediction…' : 'Fetching random test sample…';
            logregCard.hidden = true;
            renderTradeBotReport(null, tradeUi.logregSample);
            try {
              const minutes = getForecastMinutes();
              const response = await fetch(`/api/logreg/sample?mode=${mode}&minutes=${minutes}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderLogregSample(payload);
            } catch (error) {
              console.error(error);
              logregStatus.textContent = error.message || 'Logistic regression data unavailable.';
              if (window.Plotly) {
                Plotly.purge('logreg-sample-chart');
              }
              renderTradeBotReport(null, tradeUi.logregSample);
            }
          }

          function renderLogregSample(data) {
            const mode = data.mode || (logregToggle && logregToggle.checked ? 'latest' : 'random');
            if (mode === 'random') {
              if (data.ground_truth === null || data.ground_truth === undefined) {
                logregStatus.textContent = 'Random test sample. Ground truth unavailable.';
              } else if (Number(data.ground_truth) === Number(data.prediction)) {
                logregStatus.textContent = 'Random test sample · Prediction matched the ground truth.';
              } else {
                logregStatus.textContent = 'Random test sample · Prediction differed from the ground truth.';
              }
            } else {
              logregStatus.textContent = 'Latest candle prediction (ground truth may not yet exist).';
            }

            if (typeof data.horizon_minutes === 'number') {
              logregStatus.textContent += ` Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}.`;
            }

            const ts = data.datetime;
            const date = new Date(ts);
            const humanTime = Number.isNaN(date.getTime()) ? ts : date.toLocaleString();
            const closePrice = typeof data.close === 'number' ? data.close : Number(data.close);
            const priceText = Number.isFinite(closePrice) ? closePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A';
            logregTime.textContent = `Candle time: ${humanTime} · Close: ${priceText} USDT`;

            applyOutcome(logregGroundTruth, data.ground_truth);
            applyOutcome(logregPrediction, data.prediction);

            if (typeof data.probability === 'number' && Number.isFinite(data.probability)) {
              const percentage = (data.probability * 100).toFixed(1);
              logregProbability.textContent = `${percentage}% up`;
            } else {
              logregProbability.textContent = 'N/A';
            }

            logregFeatureList.innerHTML = '';
            const features = data.features || {};
            Object.entries(features).forEach(([name, value]) => {
              const li = document.createElement('li');
              let display = Number(value);
              if (!Number.isFinite(display)) {
                li.textContent = `${name}: ${value}`;
              } else {
                li.textContent = `${name}: ${display.toFixed(4)}`;
              }
              logregFeatureList.appendChild(li);
            });

            renderLogregSampleChart(data);
            renderTradeBotReport(data.trade_bot, tradeUi.logregSample);
            logregCard.hidden = false;
          }

          function renderLogregSampleChart(data) {
            if (!logregSampleChart || !window.Plotly) {
              return;
            }
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            const forecast = Array.isArray(data.forecast_candles) ? data.forecast_candles : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : [];

            if (!history.length && !forecast.length) {
              Plotly.purge('logreg-sample-chart');
              return;
            }

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.55,
              });
            }

            if (forecast.length) {
              traces.push({
                x: forecast.map(c => c.datetime),
                open: forecast.map(c => c.open),
                high: forecast.map(c => c.high),
                low: forecast.map(c => c.low),
                close: forecast.map(c => c.close),
                type: 'candlestick',
                name: 'Forecast',
                increasing: { line: { color: '#38bdf8' } },
                decreasing: { line: { color: '#f97316' } },
                opacity: 0.75,
              });
            }

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 40 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            Plotly.react('logreg-sample-chart', traces, layout, { responsive: true, displaylogo: false });
          }

          function getXgbForecastMinutes() {
            const rawValue = xgbForecastStepsInput ? Number(xgbForecastStepsInput.value) : 15;
            const bounded = Math.max(1, Math.min(Math.round(rawValue) || 15, 120));
            if (xgbForecastStepsInput) {
              xgbForecastStepsInput.value = bounded;
            }
            return bounded;
          }

          async function runXgbForecast(auto = false) {
            if (!xgbForecastStatus) return;
            const bounded = getXgbForecastMinutes();
            const mode = xgbForecastToggle && xgbForecastToggle.checked ? 'latest' : 'random';
            xgbForecastStatus.textContent = mode === 'latest' ? 'Generating forecast from latest candle…' : 'Generating historical forecast sample…';
            if (xgbForecastRunButton) {
              xgbForecastRunButton.disabled = true;
            }
            renderTradeBotReport(null, tradeUi.xgbForecast);
            try {
              const response = await fetch(`/api/xgb/forecast?minutes=${bounded}&mode=${mode}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderXgbForecast(payload);
            } catch (error) {
              console.error(error);
              xgbForecastStatus.textContent = error.message || 'Forecast unavailable.';
              if (!auto && window.Plotly) {
                Plotly.purge('xgb-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.xgbForecast);
            } finally {
              if (xgbForecastRunButton) {
                xgbForecastRunButton.disabled = false;
              }
            }
          }

          function renderXgbForecast(data) {
            if (!xgbForecastStatus) return;
            const forecast = Array.isArray(data.forecast) ? data.forecast : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : Array.isArray(data.ground_truth) ? data.ground_truth : [];
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            if (!forecast.length) {
              xgbForecastStatus.textContent = 'Forecast unavailable.';
              if (window.Plotly) {
                Plotly.purge('xgb-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.xgbForecast);
              return;
            }

            const anchorDate = data.anchor && data.anchor.datetime ? new Date(data.anchor.datetime) : null;
            const statusParts = [];
            if (anchorDate && !Number.isNaN(anchorDate.getTime())) {
              statusParts.push(`Anchor candle: ${anchorDate.toLocaleString()}`);
            }
            statusParts.push(data.mode === 'latest' ? 'Mode: latest forecast' : 'Mode: historical evaluation');
            if (typeof data.horizon_minutes === 'number') {
              statusParts.push(`Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}`);
            }
            if (groundTruth.length === forecast.length && groundTruth.length) {
              statusParts.push('Ground truth overlay available.');
            } else if (groundTruth.length) {
              statusParts.push('Partial ground truth available.');
            } else {
              statusParts.push('Ground truth unavailable for this horizon.');
            }
            const lastForecast = forecast[forecast.length - 1];
            if (lastForecast && typeof lastForecast.probability === 'number') {
              statusParts.push(`Last step up probability: ${(lastForecast.probability * 100).toFixed(1)}%`);
            }
            xgbForecastStatus.textContent = statusParts.join(' · ');
            renderTradeBotReport(data.trade_bot, tradeUi.xgbForecast);

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.5,
              });
            }

            const forecastTrace = {
              x: forecast.map(c => c.datetime),
              open: forecast.map(c => c.open),
              high: forecast.map(c => c.high),
              low: forecast.map(c => c.low),
              close: forecast.map(c => c.close),
              type: 'candlestick',
              name: 'Forecast',
              increasing: { line: { color: '#a855f7' } },
              decreasing: { line: { color: '#f59e0b' } },
              opacity: 0.65,
            };
            traces.push(forecastTrace);

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 50 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            if (window.Plotly) {
              Plotly.react('xgb-forecast-chart', traces, layout, { responsive: true, displaylogo: false });
            }
          }

          async function loadXgbSample() {
            if (!xgbStatus) return;
            const useLatest = xgbToggle && xgbToggle.checked;
            const mode = useLatest ? 'latest' : 'random';
            xgbStatus.textContent = useLatest ? 'Fetching latest candle prediction…' : 'Fetching random test sample…';
            xgbCard.hidden = true;
            renderTradeBotReport(null, tradeUi.xgbSample);
            try {
              const minutes = getXgbForecastMinutes();
              const response = await fetch(`/api/xgb/sample?mode=${mode}&minutes=${minutes}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderXgbSample(payload);
            } catch (error) {
              console.error(error);
              xgbStatus.textContent = error.message || 'XGBoost data unavailable.';
              if (window.Plotly) {
                Plotly.purge('xgb-sample-chart');
              }
              renderTradeBotReport(null, tradeUi.xgbSample);
            }
          }

          function renderXgbSample(data) {
            const mode = data.mode || (xgbToggle && xgbToggle.checked ? 'latest' : 'random');
            if (mode === 'random') {
              if (data.ground_truth === null || data.ground_truth === undefined) {
                xgbStatus.textContent = 'Random test sample. Ground truth unavailable.';
              } else if (Number(data.ground_truth) === Number(data.prediction)) {
                xgbStatus.textContent = 'Random test sample · Prediction matched the ground truth.';
              } else {
                xgbStatus.textContent = 'Random test sample · Prediction differed from the ground truth.';
              }
            } else {
              xgbStatus.textContent = 'Latest candle prediction (ground truth may not yet exist).';
            }

            if (typeof data.horizon_minutes === 'number') {
              xgbStatus.textContent += ` Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}.`;
            }

            const ts = data.datetime;
            const date = new Date(ts);
            const humanTime = Number.isNaN(date.getTime()) ? ts : date.toLocaleString();
            const closePrice = typeof data.close === 'number' ? data.close : Number(data.close);
            const priceText = Number.isFinite(closePrice) ? closePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A';
            xgbTime.textContent = `Candle time: ${humanTime} · Close: ${priceText} USDT`;

            applyOutcome(xgbGroundTruth, data.ground_truth);
            applyOutcome(xgbPrediction, data.prediction);

            if (typeof data.probability === 'number' && Number.isFinite(data.probability)) {
              const percentage = (data.probability * 100).toFixed(1);
              xgbProbability.textContent = `${percentage}% up`;
            } else {
              xgbProbability.textContent = 'N/A';
            }

            xgbFeatureList.innerHTML = '';
            const features = data.features || {};
            Object.entries(features).forEach(([name, value]) => {
              const li = document.createElement('li');
              let display = Number(value);
              if (!Number.isFinite(display)) {
                li.textContent = `${name}: ${value}`;
              } else {
                li.textContent = `${name}: ${display.toFixed(4)}`;
              }
              xgbFeatureList.appendChild(li);
            });

            renderXgbSampleChart(data);
            renderTradeBotReport(data.trade_bot, tradeUi.xgbSample);
            xgbCard.hidden = false;
          }

          function renderXgbSampleChart(data) {
            if (!xgbSampleChart || !window.Plotly) {
              return;
            }
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            const forecast = Array.isArray(data.forecast_candles) ? data.forecast_candles : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : [];

            if (!history.length && !forecast.length) {
              Plotly.purge('xgb-sample-chart');
              return;
            }

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.55,
              });
            }

            if (forecast.length) {
              traces.push({
                x: forecast.map(c => c.datetime),
                open: forecast.map(c => c.open),
                high: forecast.map(c => c.high),
                low: forecast.map(c => c.low),
                close: forecast.map(c => c.close),
                type: 'candlestick',
                name: 'Forecast',
                increasing: { line: { color: '#a855f7' } },
                decreasing: { line: { color: '#f97316' } },
                opacity: 0.75,
              });
            }

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 40 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            Plotly.react('xgb-sample-chart', traces, layout, { responsive: true, displaylogo: false });
          }

          function getLstmForecastMinutes() {
            const rawValue = lstmForecastStepsInput ? Number(lstmForecastStepsInput.value) : 15;
            const bounded = Math.max(1, Math.min(Math.round(rawValue) || 15, 120));
            if (lstmForecastStepsInput) {
              lstmForecastStepsInput.value = bounded;
            }
            return bounded;
          }

          async function runLstmForecast(auto = false) {
            if (!lstmForecastStatus) return;
            const bounded = getLstmForecastMinutes();
            const mode = lstmForecastToggle && lstmForecastToggle.checked ? 'latest' : 'random';
            lstmForecastStatus.textContent = mode === 'latest' ? 'Generating forecast from latest candle…' : 'Generating historical forecast sample…';
            if (lstmForecastRunButton) {
              lstmForecastRunButton.disabled = true;
            }
            renderTradeBotReport(null, tradeUi.lstmForecast);
            try {
              const response = await fetch(`/api/lstm/forecast?minutes=${bounded}&mode=${mode}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderLstmForecast(payload);
            } catch (error) {
              console.error(error);
              lstmForecastStatus.textContent = error.message || 'Forecast unavailable.';
              if (!auto && window.Plotly) {
                Plotly.purge('lstm-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.lstmForecast);
            } finally {
              if (lstmForecastRunButton) {
                lstmForecastRunButton.disabled = false;
              }
            }
          }

          function renderLstmForecast(data) {
            if (!lstmForecastStatus) return;
            const forecast = Array.isArray(data.forecast) ? data.forecast : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : Array.isArray(data.ground_truth) ? data.ground_truth : [];
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            if (!forecast.length) {
              lstmForecastStatus.textContent = 'Forecast unavailable.';
              if (window.Plotly) {
                Plotly.purge('lstm-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.lstmForecast);
              return;
            }

            const anchorDate = data.anchor && data.anchor.datetime ? new Date(data.anchor.datetime) : null;
            const statusParts = [];
            if (anchorDate && !Number.isNaN(anchorDate.getTime())) {
              statusParts.push(`Anchor candle: ${anchorDate.toLocaleString()}`);
            }
            statusParts.push(data.mode === 'latest' ? 'Mode: latest forecast' : 'Mode: historical evaluation');
            if (typeof data.horizon_minutes === 'number') {
              statusParts.push(`Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}`);
            }
            if (groundTruth.length === forecast.length && groundTruth.length) {
              statusParts.push('Ground truth overlay available.');
            } else if (groundTruth.length) {
              statusParts.push('Partial ground truth available.');
            } else {
              statusParts.push('Ground truth unavailable for this horizon.');
            }
            const lastForecast = forecast[forecast.length - 1];
            if (lastForecast && typeof lastForecast.probability === 'number') {
              statusParts.push(`Last step up probability: ${(lastForecast.probability * 100).toFixed(1)}%`);
            }
            lstmForecastStatus.textContent = statusParts.join(' · ');
            renderTradeBotReport(data.trade_bot, tradeUi.lstmForecast);

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.5,
              });
            }

            const forecastTrace = {
              x: forecast.map(c => c.datetime),
              open: forecast.map(c => c.open),
              high: forecast.map(c => c.high),
              low: forecast.map(c => c.low),
              close: forecast.map(c => c.close),
              type: 'candlestick',
              name: 'Forecast',
              increasing: { line: { color: '#38bdf8' } },
              decreasing: { line: { color: '#f59e0b' } },
              opacity: 0.65,
            };
            traces.push(forecastTrace);

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 50 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            if (window.Plotly) {
              Plotly.react('lstm-forecast-chart', traces, layout, { responsive: true, displaylogo: false });
            }
          }

          async function loadLstmSample() {
            if (!lstmStatus) return;
            const useLatest = lstmToggle && lstmToggle.checked;
            const mode = useLatest ? 'latest' : 'random';
            lstmStatus.textContent = useLatest ? 'Fetching latest candle prediction…' : 'Fetching random test sample…';
            lstmCard.hidden = true;
            renderTradeBotReport(null, tradeUi.lstmSample);
            try {
              const minutes = getLstmForecastMinutes();
              const response = await fetch(`/api/lstm/sample?mode=${mode}&minutes=${minutes}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderLstmSample(payload);
            } catch (error) {
              console.error(error);
              lstmStatus.textContent = error.message || 'LSTM data unavailable.';
              if (window.Plotly) {
                Plotly.purge('lstm-sample-chart');
              }
              renderTradeBotReport(null, tradeUi.lstmSample);
            }
          }

          function renderLstmSample(data) {
            const mode = data.mode || (lstmToggle && lstmToggle.checked ? 'latest' : 'random');
            if (mode === 'random') {
              if (data.ground_truth === null || data.ground_truth === undefined) {
                lstmStatus.textContent = 'Random test sample. Ground truth unavailable.';
              } else if (Number(data.ground_truth) === Number(data.prediction)) {
                lstmStatus.textContent = 'Random test sample · Prediction matched the ground truth.';
              } else {
                lstmStatus.textContent = 'Random test sample · Prediction differed from the ground truth.';
              }
            } else {
              lstmStatus.textContent = 'Latest candle prediction (ground truth may not yet exist).';
            }

            if (typeof data.horizon_minutes === 'number') {
              lstmStatus.textContent += ` Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}.`;
            }

            const ts = data.datetime;
            const date = new Date(ts);
            const humanTime = Number.isNaN(date.getTime()) ? ts : date.toLocaleString();
            const closePrice = typeof data.close === 'number' ? data.close : Number(data.close);
            const priceText = Number.isFinite(closePrice) ? closePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A';
            lstmTime.textContent = `Candle time: ${humanTime} · Close: ${priceText} USDT`;

            applyOutcome(lstmGroundTruth, data.ground_truth);
            applyOutcome(lstmPrediction, data.prediction);

            if (typeof data.probability === 'number' && Number.isFinite(data.probability)) {
              const percentage = (data.probability * 100).toFixed(1);
              lstmProbability.textContent = `${percentage}% up`;
            } else {
              lstmProbability.textContent = 'N/A';
            }

            lstmFeatureList.innerHTML = '';
            const features = data.features || {};
            Object.entries(features).forEach(([name, value]) => {
              const li = document.createElement('li');
              let display = Number(value);
              if (!Number.isFinite(display)) {
                li.textContent = `${name}: ${value}`;
              } else {
                li.textContent = `${name}: ${display.toFixed(4)}`;
              }
              lstmFeatureList.appendChild(li);
            });

            renderLstmSampleChart(data);
            renderTradeBotReport(data.trade_bot, tradeUi.lstmSample);
            lstmCard.hidden = false;
          }

          function renderLstmSampleChart(data) {
            if (!lstmSampleChart || !window.Plotly) {
              return;
            }
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            const forecast = Array.isArray(data.forecast_candles) ? data.forecast_candles : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : [];

            if (!history.length && !forecast.length) {
              Plotly.purge('lstm-sample-chart');
              return;
            }

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.55,
              });
            }

            if (forecast.length) {
              traces.push({
                x: forecast.map(c => c.datetime),
                open: forecast.map(c => c.open),
                high: forecast.map(c => c.high),
                low: forecast.map(c => c.low),
                close: forecast.map(c => c.close),
                type: 'candlestick',
                name: 'Forecast',
                increasing: { line: { color: '#38bdf8' } },
                decreasing: { line: { color: '#f97316' } },
                opacity: 0.75,
              });
            }

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 40 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            Plotly.react('lstm-sample-chart', traces, layout, { responsive: true, displaylogo: false });
          }

          function getRfForecastMinutes() {
            const rawValue = rfForecastStepsInput ? Number(rfForecastStepsInput.value) : 15;
            const bounded = Math.max(1, Math.min(Math.round(rawValue) || 15, 120));
            if (rfForecastStepsInput) {
              rfForecastStepsInput.value = bounded;
            }
            return bounded;
          }

          async function runRfForecast(auto = false) {
            if (!rfForecastStatus) return;
            const bounded = getRfForecastMinutes();
            const mode = rfForecastToggle && rfForecastToggle.checked ? 'latest' : 'random';
            rfForecastStatus.textContent = mode === 'latest' ? 'Generating forecast from latest candle…' : 'Generating historical forecast sample…';
            if (rfForecastRunButton) {
              rfForecastRunButton.disabled = true;
            }
            renderTradeBotReport(null, tradeUi.rfForecast);
            try {
              const response = await fetch(`/api/rf/forecast?minutes=${bounded}&mode=${mode}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderRfForecast(payload);
            } catch (error) {
              console.error(error);
              rfForecastStatus.textContent = error.message || 'Forecast unavailable.';
              if (!auto && window.Plotly) {
                Plotly.purge('rf-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.rfForecast);
            } finally {
              if (rfForecastRunButton) {
                rfForecastRunButton.disabled = false;
              }
            }
          }

          function renderRfForecast(data) {
            if (!rfForecastStatus) return;
            const forecast = Array.isArray(data.forecast) ? data.forecast : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : Array.isArray(data.ground_truth) ? data.ground_truth : [];
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            if (!forecast.length) {
              rfForecastStatus.textContent = 'Forecast unavailable.';
              if (window.Plotly) {
                Plotly.purge('rf-forecast-chart');
              }
              renderTradeBotReport(null, tradeUi.rfForecast);
              return;
            }

            const anchorDate = data.anchor && data.anchor.datetime ? new Date(data.anchor.datetime) : null;
            const statusParts = [];
            if (anchorDate && !Number.isNaN(anchorDate.getTime())) {
              statusParts.push(`Anchor candle: ${anchorDate.toLocaleString()}`);
            }
            statusParts.push(data.mode === 'latest' ? 'Mode: latest forecast' : 'Mode: historical evaluation');
            if (typeof data.horizon_minutes === 'number') {
              statusParts.push(`Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}`);
            }
            if (groundTruth.length === forecast.length && groundTruth.length) {
              statusParts.push('Ground truth overlay available.');
            } else if (groundTruth.length) {
              statusParts.push('Partial ground truth available.');
            } else {
              statusParts.push('Ground truth unavailable for this horizon.');
            }
            const lastForecast = forecast[forecast.length - 1];
            if (lastForecast && typeof lastForecast.probability === 'number') {
              statusParts.push(`Last step up probability: ${(lastForecast.probability * 100).toFixed(1)}%`);
            }
            rfForecastStatus.textContent = statusParts.join(' · ');
            renderTradeBotReport(data.trade_bot, tradeUi.rfForecast);

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.5,
              });
            }

            const forecastTrace = {
              x: forecast.map(c => c.datetime),
              open: forecast.map(c => c.open),
              high: forecast.map(c => c.high),
              low: forecast.map(c => c.low),
              close: forecast.map(c => c.close),
              type: 'candlestick',
              name: 'Forecast',
              increasing: { line: { color: '#38bdf8' } },
              decreasing: { line: { color: '#f59e0b' } },
              opacity: 0.65,
            };
            traces.push(forecastTrace);

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 50 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            if (window.Plotly) {
              Plotly.react('rf-forecast-chart', traces, layout, { responsive: true, displaylogo: false });
            }
          }

          async function loadRfSample() {
            if (!rfStatus) return;
            const useLatest = rfToggle && rfToggle.checked;
            const mode = useLatest ? 'latest' : 'random';
            rfStatus.textContent = useLatest ? 'Fetching latest candle prediction…' : 'Fetching random test sample…';
            rfCard.hidden = true;
            renderTradeBotReport(null, tradeUi.rfSample);
            try {
              const minutes = getRfForecastMinutes();
              const response = await fetch(`/api/rf/sample?mode=${mode}&minutes=${minutes}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderRfSample(payload);
            } catch (error) {
              console.error(error);
              rfStatus.textContent = error.message || 'Random forest data unavailable.';
              if (window.Plotly) {
                Plotly.purge('rf-sample-chart');
              }
              renderTradeBotReport(null, tradeUi.rfSample);
            }
          }

          function renderRfSample(data) {
            const mode = data.mode || (rfToggle && rfToggle.checked ? 'latest' : 'random');
            if (mode === 'random') {
              if (data.ground_truth === null || data.ground_truth === undefined) {
                rfStatus.textContent = 'Random test sample. Ground truth unavailable.';
              } else if (Number(data.ground_truth) === Number(data.prediction)) {
                rfStatus.textContent = 'Random test sample · Prediction matched the ground truth.';
              } else {
                rfStatus.textContent = 'Random test sample · Prediction differed from the ground truth.';
              }
            } else {
              rfStatus.textContent = 'Latest candle prediction (ground truth may not yet exist).';
            }

            if (typeof data.horizon_minutes === 'number') {
              rfStatus.textContent += ` Horizon: ${data.horizon_minutes} minute${data.horizon_minutes === 1 ? '' : 's'}.`;
            }

            const ts = data.datetime;
            const date = new Date(ts);
            const humanTime = Number.isNaN(date.getTime()) ? ts : date.toLocaleString();
            const closePrice = typeof data.close === 'number' ? data.close : Number(data.close);
            const priceText = Number.isFinite(closePrice) ? closePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A';
            rfTime.textContent = `Candle time: ${humanTime} · Close: ${priceText} USDT`;

            applyOutcome(rfGroundTruth, data.ground_truth);
            applyOutcome(rfPrediction, data.prediction);

            if (typeof data.probability === 'number' && Number.isFinite(data.probability)) {
              const percentage = (data.probability * 100).toFixed(1);
              rfProbability.textContent = `${percentage}% up`;
            } else {
              rfProbability.textContent = 'N/A';
            }

            rfFeatureList.innerHTML = '';
            const features = data.features || {};
            Object.entries(features).forEach(([name, value]) => {
              const li = document.createElement('li');
              let display = Number(value);
              if (!Number.isFinite(display)) {
                li.textContent = `${name}: ${value}`;
              } else {
                li.textContent = `${name}: ${display.toFixed(4)}`;
              }
              rfFeatureList.appendChild(li);
            });

            renderRfSampleChart(data);
            renderTradeBotReport(data.trade_bot, tradeUi.rfSample);
            rfCard.hidden = false;
          }

          function renderRfSampleChart(data) {
            if (!rfSampleChart || !window.Plotly) {
              return;
            }
            const history = Array.isArray(data.history_candles) ? data.history_candles : [];
            const forecast = Array.isArray(data.forecast_candles) ? data.forecast_candles : [];
            const groundTruth = Array.isArray(data.ground_truth_candles) ? data.ground_truth_candles : [];

            if (!history.length && !forecast.length) {
              Plotly.purge('rf-sample-chart');
              return;
            }

            const traces = [];

            if (history.length) {
              traces.push({
                x: history.map(c => c.datetime),
                open: history.map(c => c.open),
                high: history.map(c => c.high),
                low: history.map(c => c.low),
                close: history.map(c => c.close),
                type: 'candlestick',
                name: 'History',
                increasing: { line: { color: '#94a3b8' } },
                decreasing: { line: { color: '#64748b' } },
                opacity: 0.55,
              });
            }

            if (forecast.length) {
              traces.push({
                x: forecast.map(c => c.datetime),
                open: forecast.map(c => c.open),
                high: forecast.map(c => c.high),
                low: forecast.map(c => c.low),
                close: forecast.map(c => c.close),
                type: 'candlestick',
                name: 'Forecast',
                increasing: { line: { color: '#38bdf8' } },
                decreasing: { line: { color: '#f97316' } },
                opacity: 0.75,
              });
            }

            if (groundTruth.length) {
              traces.push({
                x: groundTruth.map(c => c.datetime),
                open: groundTruth.map(c => c.open),
                high: groundTruth.map(c => c.high),
                low: groundTruth.map(c => c.low),
                close: groundTruth.map(c => c.close),
                type: 'candlestick',
                name: 'Ground truth',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                opacity: 0.85,
              });
            }

            const layout = {
              paper_bgcolor: '#0f172a',
              plot_bgcolor: '#0f172a',
              font: { color: '#e2e8f0' },
              margin: { l: 60, r: 30, t: 30, b: 40 },
              showlegend: true,
              legend: { orientation: 'h' },
              xaxis: { rangeslider: { visible: false } },
              yaxis: { fixedrange: false, title: 'Price (USDT)' },
            };

            Plotly.react('rf-sample-chart', traces, layout, { responsive: true, displaylogo: false });
          }

          async function fetchAndRender() {
            const statusEl = document.getElementById('status');
            statusEl.textContent = 'Refreshing data…';
            try {
              const response = await fetch(`/api/candles?interval=${currentInterval}&limit=200`);
              if (!response.ok) {
                throw new Error('API error ' + response.status);
              }
              const payload = await response.json();
              if (!payload.candles || !payload.candles.length) {
                statusEl.textContent = 'No data available yet.';
                return;
              }

              const candles = payload.candles;
              const x = candles.map(c => c.datetime);
              const open = candles.map(c => c.open);
              const high = candles.map(c => c.high);
              const low = candles.map(c => c.low);
              const close = candles.map(c => c.close);

              const trace = {
                x, open, high, low, close,
                type: 'candlestick',
                increasing: { line: { color: '#22c55e' } },
                decreasing: { line: { color: '#ef4444' } },
                name: 'BTC/USDT'
              };

              const layout = {
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0f172a',
                font: { color: '#e2e8f0' },
                margin: { l: 60, r: 30, t: 30, b: 50 },
                xaxis: { rangeslider: { visible: false } },
                yaxis: { fixedrange: false }
              };

              Plotly.newPlot('chart', [trace], layout, { responsive: true, displaylogo: false });
              statusEl.textContent = `Last update: ${new Date(payload.last_update).toLocaleTimeString()} (Interval: ${payload.interval})`;
            } catch (error) {
              console.error(error);
              statusEl.textContent = 'Failed to load data. Check the server logs.';
            }
          }

          fetchAndRender();
          setInterval(fetchAndRender, 60000);
        </script>
      </body>
    </html>
    """
    return render_template_string(template)


# Load the dataset immediately so the first request is responsive.
ensure_dataset_loaded()

if __name__ == "__main__":  # pragma: no cover - manual execution
    app.run(debug=True)
