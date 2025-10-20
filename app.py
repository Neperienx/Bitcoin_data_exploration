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

APP_UPDATE_INTERVAL = 60  # seconds
DATA_FILE = Path("data/btc_usdt_1m_all.parquet")
LOGREG_MODEL_PATH = Path("models/artifacts/logreg.pkl")
RF_MODEL_PATH = Path("models/artifacts/rf.pkl")
DEFAULT_LOOKBACK = timedelta(days=7)
BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_KLINES = 1000
LOGREG_CONTEXT_WINDOW = 60
RF_CONTEXT_WINDOW = 60

app = Flask(__name__)

data_lock = threading.Lock()
background_started = False
candles_df: pd.DataFrame | None = None
logreg_model = None
rf_model = None
logreg_samples: List[Dict[str, object]] = []
logreg_latest_sample: Dict[str, object] | None = None
rf_samples: List[Dict[str, object]] = []
rf_latest_sample: Dict[str, object] | None = None
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
        latest_index = feature_view.index[-1]
        latest_features = feature_view.loc[[latest_index]]
        try:
            proba = float(model.predict_proba(latest_features)[:, 1][0])
        except Exception:  # pragma: no cover - fallback for unexpected models
            proba = float(model.predict(latest_features)[0])
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
    return True


def ensure_dataset_loaded() -> None:
    global candles_df
    df = _read_local_dataset()
    with data_lock:
        candles_df = df
    _recompute_logreg_cache(df.copy())
    _recompute_rf_cache(df.copy())
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

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))

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
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        ground_truth = _serialise_candles(gt_slice.to_dict("records"))

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(LOGREG_CONTEXT_WINDOW)

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
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

    payload["horizon_minutes"] = len(forecasts)
    payload["history_candles"] = _serialise_candles(history_preview.to_dict("records"))
    payload["forecast_candles"] = _serialise_forecast(forecasts)
    payload["ground_truth_candles"] = _serialise_candles(gt_slice.to_dict("records"))

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
    if mode == "random":
        gt_slice = prepared.iloc[anchor_idx + 1 : anchor_idx + 1 + len(forecasts)]
        ground_truth = _serialise_candles(gt_slice.to_dict("records"))

    anchor_payload = {
        "timestamp": int(anchor_row["timestamp"]),
        "datetime": pd.to_datetime(anchor_row["datetime"]).isoformat(),
        "close": float(anchor_row["close"]),
    }

    history_preview = history.tail(RF_CONTEXT_WINDOW)

    return jsonify(
        {
            "mode": mode,
            "steps": len(forecasts),
            "anchor": anchor_payload,
            "forecast": _serialise_forecast(forecasts),
            "ground_truth_candles": ground_truth,
            "history_candles": _serialise_candles(history_preview.to_dict("records")),
            "horizon_minutes": len(forecasts),
        }
    )


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
          .toggle { display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; user-select: none; }
          .toggle input { width: 1.1rem; height: 1.1rem; }
          .logreg-forecast { margin-top: 2rem; }
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
            <button class="tab-button" data-tab="rf">Random forest</button>
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
              <div id="forecast-chart"></div>
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
          const forecastRunButton = document.getElementById('forecast-run');
          const forecastStepsInput = document.getElementById('forecast-steps');
          const forecastStatus = document.getElementById('forecast-status');
          const forecastToggle = document.getElementById('forecast-latest-toggle');
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
          let currentInterval = '1m';
          let logregHasLoaded = false;
          let rfHasLoaded = false;

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
              } else if (target === 'rf' && !rfHasLoaded) {
                updateRfControls();
                loadRfSample();
                runRfForecast(true);
                rfHasLoaded = true;
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
