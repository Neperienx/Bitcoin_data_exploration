# Bitcoin Data Exploration

This repository hosts tooling to download, maintain, and visualise minute level BTC/USDT market data sourced from Binance. The long-term goal of the project is to explore different modelling strategies for Bitcoin price evolution. To make that research repeatable we provide scripts for collecting the underlying candles and a lightweight web application to visualise and monitor the dataset.

## Project scope

* **Data collection** – `data_extraction.py` downloads 1-minute OHLCV candles from Binance using [ccxt](https://github.com/ccxt/ccxt) and stores them in chunked Parquet files. The process is resumable, so you can keep incrementally extending the historical dataset.
* **Unified dataset** – the canonical working file is `data/btc_usdt_1m_all.parquet`. This Parquet file contains the cumulative BTC/USDT minute candles that the exploration tools rely on.
* **Visual exploration** – the new Flask application (`app.py`) keeps the dataset up-to-date and serves an interactive candlestick dashboard. Different time horizons can be selected to understand intraday and higher-timeframe structure before moving to modelling experiments.

As the project evolves, additional notebooks, feature engineering utilities, and model training pipelines can be layered on top of this foundation.

## Getting started

### 1. Environment setup

Create and activate a Python environment (3.9 or later is recommended) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you intend to use the resumable downloader you will also need valid network access to Binance's public API.

### 2. Make sure the canonical dataset exists

Place the aggregated minute-level file at `data/btc_usdt_1m_all.parquet`. If you already ran `data_extraction.py` this file should be present. Otherwise, you can create it by concatenating the chunked Parquet files under `data/btc_usdt_1m/` or by downloading the data afresh.

### 3. Launch the visualisation app

The Flask application automatically keeps the dataset current and exposes a real-time dashboard:

```bash
flask --app app.py --debug run
```

The explicit `--app` flag works the same way on macOS, Linux, and Windows so you
do not need to worry about shell-specific environment variable syntax. If you
prefer setting an environment variable instead, use one of:

* `export FLASK_APP=app.py` (macOS/Linux bash/zsh)
* `$env:FLASK_APP="app.py"` (Windows PowerShell)
* `set FLASK_APP=app.py` (Windows Command Prompt)

Then visit http://127.0.0.1:5000/ in your browser.

### 4. Interacting with the dashboard

* The app updates the Parquet dataset on start-up and then every minute by requesting fresh candles from Binance.
* The front-end polls the API every minute and redraws the candlestick chart with the newest data.
* Use the time horizon buttons (1m, 5m, 15m, 1h, 4h) to switch aggregation levels.

## Development notes

* The update routine is idempotent – it only downloads candles that are newer than the most recent entry already saved locally and appends them to `btc_usdt_1m_all.parquet`.
* Candle aggregation is computed server-side with pandas resampling to ensure accurate OHLCV roll-ups.
* The API returns compact JSON payloads that the browser renders with Plotly.

## Preparing machine-learning datasets

Running ``python split_dataset.py`` transforms the consolidated Parquet file into
a machine-learning friendly folder structure:

```
data/
├── btc_usdt_1m_all.parquet  # canonical consolidated dataset
└── ml/
    ├── train.parquet        # chronological 80% training split
    └── test.parquet         # chronological 20% test split
```

The split is chronological to avoid leaking future information into the
training set.  You can customise the train/test ratio or the output directory
via ``--train-ratio`` and ``--output-dir`` command-line flags.

## Next steps

With consistent data and a baseline visualisation in place you can begin experimenting with:

* Feature engineering pipelines that derive indicators or statistical signals from the minute candles.
* Modelling notebooks that evaluate different prediction techniques.
* Automated backtesting utilities that reuse the continuously updated dataset.

