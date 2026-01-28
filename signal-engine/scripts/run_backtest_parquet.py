#!/usr/bin/env python3
"""Run backtest directly from Parquet files without QuestDB."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from backtest import BacktestConfig, BacktestEngine, BacktestReporter
from regime.detectors import ATRRegimeDetector
from signals.base import (
    MarketRegime,
    OrderbookSnapshot,
    Signal,
    SignalDirection,
    SignalType,
    Trade,
)
from signals.cvd import CVDCalculator
from signals.ofi import OFICalculator
from signals.tfi import TFICalculator

from config import Settings

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH"],
        help="Symbols to backtest (default: BTC ETH).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback window in days (default: 90 for 3 months).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Initial virtual capital.",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Fraction of capital per trade (0-1).",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage (0-1).",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.04,
        help="Take profit percentage (0-1). Default aligned with settings.py.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum signal confidence for entry. Default aligned with settings.py.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root path.",
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip regime filtering (useful when orderbook data unavailable).",
    )
    parser.add_argument(
        "--tfi-window",
        type=int,
        default=60,
        help="TFI rolling window in seconds (default: 60).",
    )
    parser.add_argument(
        "--tfi-threshold",
        type=float,
        default=0.5,
        help="TFI signal threshold. Default aligned with settings.py.",
    )
    parser.add_argument(
        "--cvd-lookback",
        type=int,
        default=100,
        help="CVD lookback periods for divergence. Default 100 (~2min at 46 trades/min).",
    )
    parser.add_argument(
        "--cvd-divergence",
        type=float,
        default=0.1,
        help="CVD divergence threshold. Default aligned with settings.py.",
    )
    parser.add_argument(
        "--cvd-min-denom",
        type=float,
        default=1.0,
        help="CVD minimum divergence denominator to prevent edge cases near zero.",
    )
    parser.add_argument(
        "--cvd-price-threshold",
        type=float,
        default=0.0005,
        help="CVD price change threshold for HH/LL detection (0.0005=0.05%%, ~$43 at $87k).",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1.0,
        help="Minimum orderbook depth (BTC) for trading (default: 1.0).",
    )
    parser.add_argument(
        "--spread-threshold",
        type=int,
        default=15,
        help="Max spread in bps before LOW_LIQUIDITY regime (default: 15).",
    )
    return parser.parse_args()


def discover_parquet_files(
    data_root: Path,
    data_type: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> List[Path]:
    """Find all Parquet files for a symbol within date range."""
    base = data_root / data_type / f"symbol={symbol}"
    if not base.exists():
        return []

    files = []
    for date_dir in sorted(base.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        try:
            dir_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if start_date <= dir_date <= end_date:
            files.extend(sorted(date_dir.glob("*.parquet")))

    return files


def load_orderbook_from_parquet(files: List[Path]) -> List[OrderbookSnapshot]:
    """Load and parse orderbook snapshots from Parquet files."""
    if not files:
        return []

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            console.print(f"[yellow]Failed to read orderbook {f}: {e}[/yellow]")

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_ms")

    snapshots = []
    for _, row in combined.iterrows():
        ts_ms = row.get("ts_ms")
        if ts_ms is None:
            continue

        ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)

        # Parse bids and asks
        raw_bids = row.get("bids", [])
        raw_asks = row.get("asks", [])

        bids = []
        for b in raw_bids[:5]:
            if isinstance(b, dict) and "price" in b and "qty" in b:
                try:
                    bids.append((float(b["price"]), float(b["qty"])))
                except (ValueError, TypeError):
                    continue

        asks = []
        for a in raw_asks[:5]:
            if isinstance(a, dict) and "price" in a and "qty" in a:
                try:
                    asks.append((float(a["price"]), float(a["qty"])))
                except (ValueError, TypeError):
                    continue

        if not bids or not asks:
            continue

        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        spread_bps = int((spread / mid_price) * 10_000) if mid_price else 0

        snapshot = OrderbookSnapshot(
            ts=ts,
            symbol=str(row.get("symbol", "")),
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread_bps=spread_bps,
        )
        snapshots.append(snapshot)

    return snapshots


def normalize_side(raw_side: str) -> str:
    """Convert perpetual trade sides to buy/sell for signal processing.

    Perpetual sides indicate position intent:
    - open_long/close_short = buying pressure (lifting asks)
    - open_short/close_long = selling pressure (hitting bids)
    """
    raw = raw_side.lower()
    if raw in ("buy", "open_long", "close_short"):
        return "buy"
    elif raw in ("sell", "open_short", "close_long"):
        return "sell"
    # Fallback: try to infer from partial match
    if "long" in raw and "close" not in raw:
        return "buy"
    if "short" in raw and "close" not in raw:
        return "sell"
    return "sell"  # Default to sell if unknown


def load_trades_from_parquet(files: List[Path]) -> List[Trade]:
    """Load and parse trades from Parquet files."""
    if not files:
        return []

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            console.print(f"[yellow]Failed to read {f}: {e}[/yellow]")

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_ms" if "ts_ms" in combined.columns else "ts")

    trades = []
    for _, row in combined.iterrows():
        ts_ms = row.get("ts_ms") or row.get("ts")
        if ts_ms is None:
            continue

        ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        raw_side = str(row.get("side", ""))
        trade = Trade(
            ts=ts,
            recv_ts=ts,
            symbol=str(row.get("symbol", "")),
            trade_id=str(row.get("trade_id", "")),
            side=normalize_side(raw_side),
            price=float(row.get("price", 0)),
            qty=float(row.get("qty", 0)),
            is_large=bool(row.get("is_large", False)),
        )
        trades.append(trade)

    return trades


def compute_signals(
    trades: List[Trade],
    orderbooks: List[OrderbookSnapshot],
    settings: Settings,
    tfi_window: int = 60,
    tfi_threshold: float = 0.5,
    cvd_lookback: int = 100,
    cvd_divergence: float = 0.1,
    cvd_min_denom: float = 1.0,
    cvd_price_threshold: float = 0.0005,
    min_depth: float = 1.0,
    spread_threshold: int = 15,
) -> tuple[List[Signal], List[MarketRegime]]:
    """Compute CVD, TFI, OFI signals and market regimes from trades and orderbook."""
    if not trades:
        return [], []

    symbol = trades[0].symbol

    # CVD with configurable parameters (tuned for tick data density)
    cvd_calc = CVDCalculator(
        symbol=symbol,
        lookback_periods=cvd_lookback,
        divergence_threshold=cvd_divergence,
        min_divergence_denom=cvd_min_denom,
        price_change_threshold=cvd_price_threshold,
    )

    # TFI with tuned parameters (60s window per strategy doc)
    tfi_calc = TFICalculator(
        symbol=symbol,
        window_seconds=tfi_window,
        signal_threshold=tfi_threshold,
    )

    # OFI calculator for orderbook signals
    ofi_calc = OFICalculator(symbol=symbol, **settings.ofi_config())

    # ATR regime detector with configurable thresholds
    atr_config = settings.atr_config()
    atr_detector = ATRRegimeDetector(
        symbol=symbol,
        atr_period=atr_config.get("period", 14),
        atr_threshold_multiplier=atr_config.get("threshold_multiplier", 1.5),
        min_depth_threshold=min_depth,
        spread_threshold_bps=spread_threshold,
    )

    signals: List[Signal] = []
    regimes: List[MarketRegime] = []

    # Pre-compute OFI signals from orderbook snapshots (sorted by time)
    ofi_signals_raw: List[Signal] = []
    for ob in orderbooks:
        ofi_signal = ofi_calc.process_snapshot(ob)
        if ofi_signal:
            ofi_signals_raw.append(ofi_signal)
    ofi_signals_raw.sort(key=lambda s: s.ts)

    # Build index for finding most recent OFI signal
    ofi_timestamps = [s.ts for s in ofi_signals_raw]
    current_ofi_idx = 0

    # Build orderbook lookup by timestamp (rounded to second)
    ob_by_ts: dict[datetime, OrderbookSnapshot] = {}
    for ob in orderbooks:
        ob_ts = ob.ts.replace(microsecond=0)
        ob_by_ts[ob_ts] = ob

    current_minute: Optional[datetime] = None
    minute_high = minute_low = minute_close = 0.0
    last_ob: Optional[OrderbookSnapshot] = None

    for trade in trades:
        # Find nearest orderbook snapshot
        trade_ts_sec = trade.ts.replace(microsecond=0)
        if trade_ts_sec in ob_by_ts:
            last_ob = ob_by_ts[trade_ts_sec]

        # Advance OFI index to find most recent OFI signal before this trade
        while (
            current_ofi_idx < len(ofi_timestamps) - 1
            and ofi_timestamps[current_ofi_idx + 1] <= trade.ts
        ):
            current_ofi_idx += 1

        # Get current OFI signal (if any)
        current_ofi: Optional[Signal] = None
        if ofi_signals_raw and ofi_timestamps[current_ofi_idx] <= trade.ts:
            current_ofi = ofi_signals_raw[current_ofi_idx]

        # Compute CVD and TFI signals
        cvd_signal = cvd_calc.process_trade(trade)
        tfi_signal = tfi_calc.process_trade(trade)

        if cvd_signal:
            signals.append(cvd_signal)
        if tfi_signal:
            signals.append(tfi_signal)

        # Forward-fill OFI: emit ONCE per trade when CVD or TFI fires (not twice)
        # This ensures OFI is aligned with trade-based signals without duplication
        if (cvd_signal or tfi_signal) and current_ofi:
            ofi_at_trade = Signal(
                ts=trade.ts,  # Use trade timestamp for alignment
                recv_ts=trade.ts,  # Use trade timestamp for consistency
                symbol=current_ofi.symbol,
                signal_type=current_ofi.signal_type,
                value=current_ofi.value,
                confidence=current_ofi.confidence,
                direction=current_ofi.direction,
                price=trade.price,  # Use current trade price
                spread_bps=current_ofi.spread_bps,
                bid_depth=current_ofi.bid_depth,
                ask_depth=current_ofi.ask_depth,
                metadata=current_ofi.metadata,
            )
            signals.append(ofi_at_trade)

        # Build minute candles for regime detection
        minute_bucket = trade.ts.replace(second=0, microsecond=0)
        if current_minute is None:
            current_minute = minute_bucket
            minute_high = minute_low = minute_close = trade.price
        elif minute_bucket == current_minute:
            minute_high = max(minute_high, trade.price)
            minute_low = min(minute_low, trade.price)
            minute_close = trade.price
        else:
            # New minute - update regime detector with orderbook context
            if last_ob:
                bid_depth = sum(qty for _, qty in last_ob.bids)
                ask_depth = sum(qty for _, qty in last_ob.asks)
                atr_detector.update_orderbook_context(
                    last_ob.spread_bps, bid_depth, ask_depth
                )

            atr_detector.update_price(
                current_minute, minute_high, minute_low, minute_close
            )
            regime = atr_detector.detect_regime(current_minute)
            if regime:
                regimes.append(regime)

            current_minute = minute_bucket
            minute_high = minute_low = minute_close = trade.price

    # Final candle
    if current_minute is not None:
        if last_ob:
            bid_depth = sum(qty for _, qty in last_ob.bids)
            ask_depth = sum(qty for _, qty in last_ob.asks)
            atr_detector.update_orderbook_context(
                last_ob.spread_bps, bid_depth, ask_depth
            )
        atr_detector.update_price(current_minute, minute_high, minute_low, minute_close)
        regime = atr_detector.detect_regime(current_minute)
        if regime:
            regimes.append(regime)

    # Sort signals by timestamp
    signals.sort(key=lambda s: s.ts)

    return signals, regimes


def build_price_map(trades: List[Trade]) -> dict[datetime, float]:
    """Build timestamp->price mapping from trades."""
    return {trade.ts: trade.price for trade in trades}


def main() -> None:
    args = parse_args()
    settings = Settings()

    if args.data_root:
        settings.data_root = args.data_root

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_confidence=args.min_confidence,
        min_signals_agree=2,
        require_cvd=False,  # CVD as supplementary context, not mandatory
        require_tfi=True,  # TFI as primary signal
        require_ofi=True,  # OFI as confirmation
    )

    reporter = BacktestReporter(console)
    symbols = [s.upper() for s in args.symbols]

    console.print(
        f"[bold blue]Running Parquet backtest[/bold blue] "
        f"symbols={', '.join(symbols)} window={start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}"
    )

    processed = 0
    skipped: List[str] = []

    for idx, symbol in enumerate(symbols, start=1):
        if len(symbols) > 1:
            console.rule(f"{idx}. {symbol}")

        console.print(f"[bold]Loading trades for {symbol}...[/bold]")
        trade_files = discover_parquet_files(
            settings.data_root, "trades", symbol, start_dt, end_dt
        )

        if not trade_files:
            console.print(
                f"[yellow]No trade files found for {symbol}. Skipping.[/yellow]"
            )
            skipped.append(symbol)
            continue

        console.print(f"Found {len(trade_files)} Parquet files")
        trades = load_trades_from_parquet(trade_files)

        if not trades:
            console.print(f"[yellow]No trades loaded for {symbol}. Skipping.[/yellow]")
            skipped.append(symbol)
            continue

        console.print(f"Loaded {len(trades):,} trades")

        # Data diagnostics
        if trades:
            console.print(f"  First trade: {trades[0].ts} price={trades[0].price:.2f}")
            console.print(f"  Last trade: {trades[-1].ts} price={trades[-1].price:.2f}")
            buys = sum(1 for t in trades if t.side == "buy")
            sells = len(trades) - buys
            console.print(
                f"  Buy/Sell ratio: {buys:,}/{sells:,} ({100*buys/len(trades):.1f}%/{100*sells/len(trades):.1f}%)"
            )

        # Load orderbook data
        console.print("[bold]Loading orderbook data...[/bold]")
        orderbook_files = discover_parquet_files(
            settings.data_root, "orderbook", symbol, start_dt, end_dt
        )
        if orderbook_files:
            console.print(f"Found {len(orderbook_files)} orderbook files")
            orderbooks = load_orderbook_from_parquet(orderbook_files)
            console.print(f"Loaded {len(orderbooks):,} orderbook snapshots")
        else:
            console.print(
                "[yellow]No orderbook data found - regime detection may be limited[/yellow]"
            )
            orderbooks = []

        console.print("Computing signals...")

        signals, regimes = compute_signals(
            trades,
            orderbooks,
            settings,
            tfi_window=args.tfi_window,
            tfi_threshold=args.tfi_threshold,
            cvd_lookback=args.cvd_lookback,
            cvd_divergence=args.cvd_divergence,
            cvd_min_denom=args.cvd_min_denom,
            cvd_price_threshold=args.cvd_price_threshold,
            min_depth=args.min_depth,
            spread_threshold=args.spread_threshold,
        )
        console.print(
            f"Generated {len(signals):,} signals, {len(regimes):,} regime updates"
        )

        # Signal diagnostics
        cvd_signals = [s for s in signals if s.signal_type == "cvd"]
        tfi_signals = [s for s in signals if s.signal_type == "tfi"]
        ofi_signals = [s for s in signals if s.signal_type == "ofi"]
        console.print(
            f"  CVD: {len(cvd_signals):,}, TFI: {len(tfi_signals):,}, OFI: {len(ofi_signals):,}"
        )

        if cvd_signals:
            cvd_bullish = sum(1 for s in cvd_signals if s.direction == "bullish")
            cvd_bearish = sum(1 for s in cvd_signals if s.direction == "bearish")
            avg_conf = sum(s.confidence for s in cvd_signals) / len(cvd_signals)
            console.print(
                f"    CVD: {cvd_bullish} bullish, {cvd_bearish} bearish, avg conf: {avg_conf:.2f}"
            )

        if tfi_signals:
            tfi_bullish = sum(1 for s in tfi_signals if s.direction == "bullish")
            tfi_bearish = sum(1 for s in tfi_signals if s.direction == "bearish")
            avg_conf = sum(s.confidence for s in tfi_signals) / len(tfi_signals)
            console.print(
                f"    TFI: {tfi_bullish} bullish, {tfi_bearish} bearish, avg conf: {avg_conf:.2f}"
            )

        if ofi_signals:
            ofi_bullish = sum(1 for s in ofi_signals if s.direction == "bullish")
            ofi_bearish = sum(1 for s in ofi_signals if s.direction == "bearish")
            avg_conf = sum(s.confidence for s in ofi_signals) / len(ofi_signals)
            console.print(
                f"    OFI: {ofi_bullish} bullish, {ofi_bearish} bearish, avg conf: {avg_conf:.2f}"
            )

        if regimes:
            should_trade_count = sum(1 for r in regimes if r.should_trade)
            regime_breakdown = {}
            for r in regimes:
                regime_breakdown[r.regime] = regime_breakdown.get(r.regime, 0) + 1
            console.print(
                f"  Regimes allowing trade: {should_trade_count}/{len(regimes)} ({100*should_trade_count/len(regimes):.1f}%)"
            )
            console.print(f"    Breakdown: {dict(regime_breakdown)}")

        if not signals:
            console.print(
                f"[yellow]No signals generated for {symbol}. Skipping.[/yellow]"
            )
            skipped.append(symbol)
            continue

        price_map = build_price_map(trades)
        console.print(f"Built price map with {len(price_map):,} points")

        console.print("[green]Running backtest...[/green]")
        engine = BacktestEngine(config)

        # If skipping regime, pass empty regimes so trades aren't blocked
        backtest_regimes = [] if args.skip_regime else regimes
        if args.skip_regime:
            console.print("[yellow]Regime filtering disabled[/yellow]")

        results = engine.run(
            signals=signals,
            regimes=backtest_regimes,
            price_data={symbol: price_map},
        )

        reporter.display(results)
        processed += 1

    if not processed:
        console.print("[red]No symbols produced results.[/red]")
        raise SystemExit(1)

    if skipped:
        console.print(f"[yellow]Skipped: {', '.join(skipped)}[/yellow]")


if __name__ == "__main__":
    main()
