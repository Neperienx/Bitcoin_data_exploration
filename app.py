"""Flask application that keeps BTC/USDT minute data current and serves
an interactive candlestick dashboard."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

APP_UPDATE_INTERVAL = 60  # seconds
DATA_FILE = Path("data/btc_usdt_1m_all.parquet")
DEFAULT_LOOKBACK = timedelta(days=7)
BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_KLINES = 1000

app = Flask(__name__)

data_lock = threading.Lock()
background_started = False
candles_df: pd.DataFrame | None = None


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
    return True


def ensure_dataset_loaded() -> None:
    global candles_df
    df = _read_local_dataset()
    with data_lock:
        candles_df = df
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
          body { font-family: 'Inter', sans-serif; margin: 0; background: #0f172a; color: #f1f5f9; }
          header { padding: 1.5rem 2rem; background: #111827; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
          h1 { margin: 0; font-size: 1.6rem; }
          main { padding: 1rem 2rem 2rem; }
          .controls { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
          button { background: #2563eb; border: none; border-radius: 999px; padding: 0.6rem 1.2rem; color: #f8fafc; font-weight: 600; cursor: pointer; transition: background 0.2s ease; }
          button:hover { background: #1d4ed8; }
          button.active { background: #22c55e; }
          #chart { width: 100%; height: 70vh; }
          .status { margin-top: 0.75rem; font-size: 0.9rem; color: #cbd5f5; }
        </style>
      </head>
      <body>
        <header>
          <h1>Bitcoin Candle Explorer</h1>
          <p>Up-to-date BTC/USDT candles aggregated from Binance.</p>
        </header>
        <main>
          <div class="controls">
            <button data-interval="1m" class="active">1 Minute</button>
            <button data-interval="5m">5 Minutes</button>
            <button data-interval="15m">15 Minutes</button>
            <button data-interval="1h">1 Hour</button>
            <button data-interval="4h">4 Hours</button>
          </div>
          <div id="chart"></div>
          <div class="status" id="status">Loading...</div>
        </main>
        <script>
          const buttons = document.querySelectorAll('button[data-interval]');
          let currentInterval = '1m';

          buttons.forEach(btn => {
            btn.addEventListener('click', () => {
              currentInterval = btn.dataset.interval;
              buttons.forEach(b => b.classList.toggle('active', b === btn));
              fetchAndRender();
            });
          });

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
