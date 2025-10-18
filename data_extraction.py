# -*- coding: utf-8 -*-
"""
Resumable 1-minute OHLCV downloader (Binance via ccxt)
- Saves data in chunked Parquet files under data/btc_usdt_1m/
- Maintains data/checkpoint.json with last saved timestamp (ms since epoch)
- Safe to stop with Ctrl+C; next run resumes automatically
"""

import os
import json
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

# ------------------ CONFIG ------------------
SYMBOL = "BTC/USDT"        # Binance pair
TIMEFRAME = "1m"           # 1-minute candles
START_ISO = "2017-01-01T00:00:00Z"  # initial start if no data yet
LIMIT = 1000               # Binance max candles per request
CHUNK_ROWS = 100_000       # how many rows before writing a new parquet
SAVE_DIR = "data/btc_usdt_1m"
CHECKPOINT = "data/checkpoint.json"
# --------------------------------------------

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

def utc_from_ms(ms):
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            doc = json.load(f)
            return int(doc.get("last_ts_ms"))
    return None

def save_checkpoint(ts_ms):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"last_ts_ms": int(ts_ms)}, f)
    os.replace(tmp, CHECKPOINT)

def find_last_ts_from_files():
    """
    If checkpoint is missing, scan existing Parquet files and return the max timestamp.
    This reads just the 'ts' column for each file (fast for moderate number of files).
    """
    files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".parquet")]
    if not files:
        return None
    max_ts = None
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=["ts"])
        except Exception:
            # If columns option fails due to metadata issues, fall back to full read
            df = pd.read_parquet(fp)
        if not df.empty:
            m = int(df["ts"].max())
            max_ts = m if (max_ts is None or m > max_ts) else max_ts
    return max_ts

def next_cursor_ms(existing_last_ts_ms):
    if existing_last_ts_ms is None:
        return ccxt.binance().parse8601(START_ISO)
    # next candle starts one minute after last candle
    return int(existing_last_ts_ms) + 60_000

def write_chunk(df_chunk):
    if df_chunk.empty:
        return None
    start_ts = int(df_chunk["ts"].min())
    end_ts = int(df_chunk["ts"].max())
    start_str = utc_from_ms(start_ts).strftime("%Y%m%d_%H%M")
    end_str = utc_from_ms(end_ts).strftime("%Y%m%d_%H%M")
    fname = f"btc_usdt_1m_{start_str}_{end_str}.parquet"
    out_path = os.path.join(SAVE_DIR, fname)
    # Write atomically
    tmp_path = out_path + ".tmp"
    df_chunk.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    return out_path

def fetch_resumable():
    ensure_dirs()

    exchange = ccxt.binance({"enableRateLimit": True})

    # figure out where to start: checkpoint -> files -> START_ISO
    last_ts_ms = load_checkpoint()
    if last_ts_ms is None:
        last_ts_ms = find_last_ts_from_files()

    cursor = next_cursor_ms(last_ts_ms)
    now_ms = exchange.milliseconds()

    print(f"Starting from: {utc_from_ms(cursor)}  (ms={cursor})")
    all_rows = []

    try:
        while cursor < now_ms:
            print(f"Fetching from {utc_from_ms(cursor)} ...")
            try:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=cursor, limit=LIMIT)
            except Exception as e:
                print("Fetch error:", e)
                time.sleep(5)
                continue

            if not ohlcv:
                print("No more data returned; stopping.")
                break

            # Convert to DataFrame & clean quickly
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

            all_rows.append(df)

            # advance the cursor
            last_batch_ts = int(df["ts"].iloc[-1])
            cursor = last_batch_ts + 60_000  # next minute

            # periodic write
            total_rows = sum(len(x) for x in all_rows)
            if total_rows >= CHUNK_ROWS:
                big = pd.concat(all_rows, ignore_index=True)
                # de-dup by ts in case of overlaps
                big = big.drop_duplicates(subset="ts").sort_values("ts")
                path = write_chunk(big)
                if path:
                    print(f"Wrote chunk: {path}  rows={len(big)}")
                    # update checkpoint atomically
                    save_checkpoint(int(big["ts"].max()))
                all_rows = []  # reset buffer

            # polite backoff (enableRateLimit already throttles)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial progress...")

    finally:
        # write any remaining rows
        if all_rows:
            big = pd.concat(all_rows, ignore_index=True)
            big = big.drop_duplicates(subset="ts").sort_values("ts")
            path = write_chunk(big)
            if path:
                print(f"Wrote final chunk: {path}  rows={len(big)}")
                save_checkpoint(int(big["ts"].max()))

        # Final message
        ck = load_checkpoint()
        if ck:
            print(f"Checkpoint saved at: {utc_from_ms(ck)}  (ms={ck})")
        print("Done.")

if __name__ == "__main__":
    fetch_resumable()
