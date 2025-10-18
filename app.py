"""Flask application that keeps BTC/USDT minute data current and serves
an interactive candlestick dashboard."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
import random
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

from mlops.features import make_basic_features, make_label
from mlops.split import time_train_test_split
from models.logistic_regression import FEATURE_ORDER, load_model, predict_logreg

APP_UPDATE_INTERVAL = 60  # seconds
DATA_FILE = Path("data/btc_usdt_1m_all.parquet")
LOGREG_MODEL_PATH = Path("models/artifacts/logreg.pkl")
DEFAULT_LOOKBACK = timedelta(days=7)
BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_KLINES = 1000

app = Flask(__name__)

data_lock = threading.Lock()
background_started = False
candles_df: pd.DataFrame | None = None
logreg_model = None
logreg_samples: List[Dict[str, object]] = []
logreg_latest_sample: Dict[str, object] | None = None


def _ensure_logreg_model_loaded() -> None:
    """Lazy-load the logistic regression model if the artifact exists."""

    global logreg_model
    if logreg_model is not None:
        return
    if not LOGREG_MODEL_PATH.exists():
        return
    try:
        logreg_model = load_model(LOGREG_MODEL_PATH)
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
    dataset = dataset.dropna(subset=FEATURE_ORDER + ["label"])

    samples: List[Dict[str, object]] = []
    if not dataset.empty:
        X = dataset.loc[:, FEATURE_ORDER]
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
            for feature in FEATURE_ORDER:
                meta[feature] = dataset.loc[X_test.index, feature]
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
                        feature: float(row[feature]) for feature in FEATURE_ORDER
                    },
                }
                for _, row in meta.iterrows()
            ]

    latest_record: Dict[str, object] | None = None
    feature_view = features.loc[:, FEATURE_ORDER].dropna()
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
                feature: float(latest_row.iloc[0][feature]) for feature in FEATURE_ORDER
            },
        }

    with data_lock:
        logreg_samples = samples
        logreg_latest_sample = latest_record


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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])
    df = pd.read_parquet(DATA_FILE)
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
    return True


def ensure_dataset_loaded() -> None:
    global candles_df
    df = _read_local_dataset()
    with data_lock:
        candles_df = df
    _recompute_logreg_cache(df.copy())
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


@app.route("/api/logreg/sample")
def api_logreg_sample():
    mode = request.args.get("mode", "random").lower()

    with data_lock:
        model_available = logreg_model is not None
        samples = [sample.copy() for sample in logreg_samples]
        latest = logreg_latest_sample.copy() if logreg_latest_sample else None

    if not model_available:
        return jsonify({"error": "Logistic regression model is unavailable."}), 503

    if mode == "latest":
        if latest is None:
            return jsonify({"error": "Latest sample unavailable."}), 404
        payload = latest.copy()
        payload["mode"] = "latest"
        return jsonify(payload)

    if not samples:
        return jsonify({"error": "No evaluation samples available."}), 404

    payload = random.choice(samples)
    payload["mode"] = "random"
    return jsonify(payload)


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
          .toggle { display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; user-select: none; }
          .toggle input { width: 1.1rem; height: 1.1rem; }
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
          let currentInterval = '1m';
          let logregHasLoaded = false;

          tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
              const target = btn.dataset.tab;
              tabButtons.forEach(b => b.classList.toggle('active', b === btn));
              tabContents.forEach(section => section.classList.toggle('active', section.id === `${target}-tab`));
              if (target === 'logreg' && !logregHasLoaded) {
                updateLogregControls();
                loadLogregSample();
                logregHasLoaded = true;
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

          async function loadLogregSample() {
            if (!logregStatus) return;
            const useLatest = logregToggle && logregToggle.checked;
            const mode = useLatest ? 'latest' : 'random';
            logregStatus.textContent = useLatest ? 'Fetching latest candle prediction…' : 'Fetching random test sample…';
            logregCard.hidden = true;
            try {
              const response = await fetch(`/api/logreg/sample?mode=${mode}`);
              const payload = await response.json();
              if (!response.ok || payload.error) {
                throw new Error(payload.error || `API error ${response.status}`);
              }
              renderLogregSample(payload);
            } catch (error) {
              console.error(error);
              logregStatus.textContent = error.message || 'Logistic regression data unavailable.';
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

            logregCard.hidden = false;
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
