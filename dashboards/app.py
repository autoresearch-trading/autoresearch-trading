from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pacifica Live Dashboard", layout="wide")


def _parse_symbols(raw: str) -> List[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def _collect_files(
    root: Path, dataset: str, symbol: str, per_partition: int
) -> List[Path]:
    files: List[Path] = []
    symbol_dir = root / dataset / f"symbol={symbol}"
    if not symbol_dir.exists():
        return files
    partitions = sorted(symbol_dir.glob("date=*"))
    for partition in partitions[-2:]:
        files.extend(sorted(partition.glob("*.parquet"))[-per_partition:])
    return files


def load_latest(
    root: Path,
    dataset: str,
    symbols: Sequence[str],
    *,
    per_partition: int = 5,
    tail: int | None = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        for parquet_file in _collect_files(root, dataset, symbol, per_partition):
            try:
                frame = pd.read_parquet(parquet_file)
                frames.append(frame)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("ts_ms")
    if tail:
        data = data.tail(tail)
    return data


sidebar = st.sidebar
sidebar.header("Collector Settings")

data_root = Path(sidebar.text_input("Data root", "./data")).expanduser()
symbols_raw = sidebar.text_input("Symbols", "BTC,ETH")
refresh_seconds = sidebar.slider(
    "Auto-refresh (seconds)", min_value=1, max_value=30, value=5
)
trade_points = sidebar.slider(
    "Trade samples", min_value=100, max_value=5000, value=1000, step=100
)
book_depth = sidebar.slider(
    "Order book depth", min_value=5, max_value=50, value=25, step=5
)

symbols = _parse_symbols(symbols_raw)

if not data_root.exists():
    st.warning(
        f"Data root '{data_root}' not found yet. Waiting for live collector to write Parquet files."
    )
    st.stop()

if not symbols:
    st.warning("Enter at least one symbol to begin monitoring.")
    st.stop()

st.caption(
    f"Streaming parquet files from `{data_root}` for symbols {', '.join(symbols)}"
)

upper_cols = st.columns(2)

prices = load_latest(data_root, "prices", symbols, tail=5000)
if not prices.empty:
    prices["ts"] = pd.to_datetime(prices["ts_ms"], unit="ms")
    fig_prices = px.line(prices, x="ts", y="price", color="symbol", title="Spot Prices")
    upper_cols[0].plotly_chart(fig_prices, use_container_width=True)
else:
    upper_cols[0].info("No price data written yet.")

trades = load_latest(data_root, "trades", symbols, tail=trade_points)
if not trades.empty:
    trades["ts"] = pd.to_datetime(trades["ts_ms"], unit="ms")
    fig_trades = px.scatter(
        trades,
        x="ts",
        y="price",
        size="qty",
        color="side",
        facet_row="symbol",
        title="Recent Trades",
    )
    upper_cols[1].plotly_chart(fig_trades, use_container_width=True)
else:
    upper_cols[1].info("No trade data written yet.")

lower_cols = st.columns(2)

orderbook = load_latest(data_root, "orderbook", symbols, per_partition=3)
if not orderbook.empty:
    for symbol in symbols:
        latest_rows = orderbook[orderbook["symbol"] == symbol]
        if latest_rows.empty:
            continue
        latest = latest_rows.iloc[-1]
        bids = pd.DataFrame(latest["bids"][:book_depth])
        asks = pd.DataFrame(latest["asks"][:book_depth])
        with lower_cols[0].expander(f"{symbol} Order Book", expanded=False):
            if not bids.empty:
                bids_fig = px.bar(
                    bids,
                    x="qty",
                    y="price",
                    orientation="h",
                    title=f"{symbol} Bids",
                )
                bids_fig.update_yaxes(autorange="reversed")
                st.plotly_chart(bids_fig, use_container_width=True)
            if not asks.empty:
                asks_fig = px.bar(
                    asks,
                    x="qty",
                    y="price",
                    orientation="h",
                    title=f"{symbol} Asks",
                )
                st.plotly_chart(asks_fig, use_container_width=True)
else:
    lower_cols[0].info("No order book snapshots yet.")

funding = load_latest(data_root, "funding", symbols, tail=5000)
if not funding.empty:
    funding["ts"] = pd.to_datetime(funding["ts_ms"], unit="ms")
    fig_funding = px.bar(
        funding, x="ts", y="rate", color="symbol", title="Funding Rates"
    )
    lower_cols[1].plotly_chart(fig_funding, use_container_width=True)
else:
    lower_cols[1].info("No funding data written yet.")

time.sleep(refresh_seconds)
st.rerun()
