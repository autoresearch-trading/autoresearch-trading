"""
measure_base_rate_stationarity.py
==================================
Measures H500 base-rate stationarity across the pre-April training window and
validates three volume-related spec claims.

Deliverables:
  1. Per-symbol total return, drift-implied H500 base rate, observed H500 base rate,
     30-day rolling H500 base rate, max intra-period swing (pp)
  2. Events/day distribution (median, p10, p90) for all 25 symbols
  3. Total window count at stride=50 across all 25 symbols / pre-April dates
  4. 200-event window duration in calendar time per symbol

Hard constraints enforced here:
  - Pre-April data ONLY (max_date = 2026-03-25, local data cutoff)
  - Correct dedup: (ts_ms, qty, price) WITHOUT side for pre-April
  - H500 label uses event-VWAP, not last-fill price
  - H500 = 500 order events ahead in post-dedup/grouping index
  - No windows crossing day boundaries
  - All rolling statistics are causal
"""

import json
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "docs" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "2Z",
    "AAVE",
    "ASTER",
    "AVAX",
    "BNB",
    "BTC",
    "CRV",
    "DOGE",
    "ENA",
    "ETH",
    "FARTCOIN",
    "HYPE",
    "KBONK",
    "KPEPE",
    "LDO",
    "LINK",
    "LTC",
    "PENGU",
    "PUMP",
    "SOL",
    "SUI",
    "UNI",
    "WLFI",
    "XPL",
    "XRP",
]

# 8 focus symbols for rolling H500 analysis
FOCUS_SYMBOLS = ["BTC", "ETH", "SOL", "CRV", "2Z", "AVAX", "UNI", "LDO"]

# Pre-April only — local data ends 2026-03-25
MAX_DATE = date(2026, 3, 25)
MIN_DATE = date(2025, 10, 16)

H500 = 500
STRIDE = 50
WINDOW_SIZE = 200

ESCALATION_SWING_BTC_ETH = 8.0  # pp — stop if BTC/ETH MATURE swing exceeds this (warm-up artifact can cause ~11pp on all-windows)
ESCALATION_SWING_ANY = 3.0  # pp — gate recommendation threshold (mature windows)
ESCALATION_RATE_BOUNDS = (35.0, 65.0)  # pp — H500 base rate must stay in [35%, 65%]
ESCALATION_WINDOW_MIN = 500_000
ESCALATION_WINDOW_MAX = 10_000_000
MIN_DAYS_IN_WINDOW = 20  # require at least this many days in the rolling window (avoids warm-up artifacts)


def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET home_directory = '{ROOT}'")
    return con


def list_dates(
    symbol: str, min_date: date = MIN_DATE, max_date: date = MAX_DATE
) -> list[str]:
    sym_dir = DATA_DIR / "trades" / f"symbol={symbol}"
    if not sym_dir.exists():
        return []
    dates = []
    for date_dir in sorted(sym_dir.iterdir()):
        date_str = date_dir.name.replace("date=", "")
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            continue
        if min_date <= d <= max_date:
            dates.append(date_str)
    return sorted(dates)


def load_day_events(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    """
    Load one symbol-day: dedup (ts_ms, qty, price), group by ts_ms,
    return DataFrame with columns [ts_ms, vwap, total_qty, n_fills].
    Pre-April data only → no event_type filter needed.
    Returns None if no files or insufficient rows.
    """
    sym_path = DATA_DIR / "trades" / f"symbol={symbol}" / f"date={date_str}"
    files = list(sym_path.glob("*.parquet"))
    if not files:
        return None

    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    con = get_con()
    try:
        df = con.execute(
            f"""
            WITH deduped AS (
                SELECT ts_ms, qty, price,
                       ROW_NUMBER() OVER (PARTITION BY ts_ms, qty, price ORDER BY recv_ms) as rn
                FROM read_parquet({file_list})
            ),
            dr AS (SELECT ts_ms, qty, price FROM deduped WHERE rn = 1),
            events AS (
                SELECT ts_ms,
                       SUM(qty * price) / NULLIF(SUM(qty), 0) AS vwap,
                       SUM(qty) AS total_qty,
                       COUNT(*) AS n_fills
                FROM dr
                GROUP BY ts_ms
                ORDER BY ts_ms
            )
            SELECT ts_ms, vwap, total_qty, n_fills FROM events
        """
        ).fetchdf()
    except Exception as e:
        return None
    finally:
        con.close()

    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Section A: Events/day distribution for all 25 symbols
# ---------------------------------------------------------------------------
def compute_events_per_day_all_symbols() -> dict:
    """
    For each symbol, compute events/day across all pre-April dates.
    Returns dict: symbol -> {median, p10, p90, n_days, all_counts}
    """
    print("\n=== Section A: Events/day distribution (all 25 symbols) ===")
    result = {}
    for sym in SYMBOLS:
        dates = list_dates(sym)
        if not dates:
            result[sym] = {"error": "no_dates"}
            continue
        counts = []
        for ds in dates:
            sym_path = DATA_DIR / "trades" / f"symbol={sym}" / f"date={ds}"
            files = list(sym_path.glob("*.parquet"))
            if not files:
                continue
            file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
            con = get_con()
            try:
                count = con.execute(
                    f"""
                    WITH deduped AS (
                        SELECT ts_ms,
                               ROW_NUMBER() OVER (PARTITION BY ts_ms, qty, price ORDER BY recv_ms) as rn
                        FROM read_parquet({file_list})
                    )
                    SELECT COUNT(DISTINCT ts_ms) FROM deduped WHERE rn = 1
                """
                ).fetchone()[0]
                counts.append(int(count))
            except Exception:
                pass
            finally:
                con.close()

        if not counts:
            result[sym] = {"error": "no_counts"}
            continue

        arr = np.array(counts, dtype=float)
        result[sym] = {
            "n_days": len(arr),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "total_events": int(arr.sum()),
        }
        print(
            f"  {sym}: {len(arr)} days, median={np.median(arr):.0f}, p10={np.percentile(arr,10):.0f}, p90={np.percentile(arr,90):.0f}"
        )

    return result


# ---------------------------------------------------------------------------
# Section B: Total window count at stride=50 (all 25 symbols, pre-April)
# ---------------------------------------------------------------------------
def compute_total_window_count(events_per_day: dict) -> dict:
    """
    For each symbol-day with N events, windows = max(0, (N - WINDOW_SIZE) // STRIDE + 1).
    No windows crossing day boundaries.
    Returns total count, per-symbol count.
    """
    print("\n=== Section B: Total window count at stride=50 ===")
    symbol_windows = {}
    grand_total = 0

    for sym in SYMBOLS:
        sym_dir = DATA_DIR / "trades" / f"symbol={sym}"
        if not sym_dir.exists():
            symbol_windows[sym] = 0
            continue

        sym_total = 0
        dates = list_dates(sym)
        for ds in dates:
            sym_path = DATA_DIR / "trades" / f"symbol={sym}" / f"date={ds}"
            files = list(sym_path.glob("*.parquet"))
            if not files:
                continue
            file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
            con = get_con()
            try:
                count = con.execute(
                    f"""
                    WITH deduped AS (
                        SELECT ts_ms,
                               ROW_NUMBER() OVER (PARTITION BY ts_ms, qty, price ORDER BY recv_ms) as rn
                        FROM read_parquet({file_list})
                    )
                    SELECT COUNT(DISTINCT ts_ms) FROM deduped WHERE rn = 1
                """
                ).fetchone()[0]
                n = int(count)
                if n >= WINDOW_SIZE:
                    windows = (n - WINDOW_SIZE) // STRIDE + 1
                    sym_total += windows
            except Exception:
                pass
            finally:
                con.close()

        symbol_windows[sym] = sym_total
        grand_total += sym_total

    print(f"  Grand total windows (stride={STRIDE}): {grand_total:,}")
    print(
        f"  Spec claim: ~3.5M — {'OK' if 500_000 <= grand_total <= 10_000_000 else 'ESCALATION'}"
    )

    if grand_total < ESCALATION_WINDOW_MIN or grand_total > ESCALATION_WINDOW_MAX:
        print(
            f"  ESCALATION: window count {grand_total} outside [{ESCALATION_WINDOW_MIN}, {ESCALATION_WINDOW_MAX}]"
        )
        raise RuntimeError(
            f"STOP: total window count {grand_total:,} is outside escalation bounds "
            f"[{ESCALATION_WINDOW_MIN:,}, {ESCALATION_WINDOW_MAX:,}]. "
            "Stride/sampling logic may be wrong."
        )

    return {"grand_total": grand_total, "per_symbol": symbol_windows}


# ---------------------------------------------------------------------------
# Section C: 200-event window duration in calendar time
# ---------------------------------------------------------------------------
def compute_window_duration(events_per_day: dict) -> dict:
    """
    For each symbol, derive median inter-event gap (ms) from a sample of dates,
    then multiply by WINDOW_SIZE to get window duration.
    Uses first available date + a mid-period date for robustness.
    """
    print("\n=== Section C: 200-event window duration ===")
    result = {}

    for sym in SYMBOLS:
        dates = list_dates(sym)
        if not dates:
            result[sym] = {"error": "no_dates"}
            continue

        # Sample 5 dates spread across the window
        n = len(dates)
        sample_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        sample_dates = [dates[i] for i in sample_indices if i < n]

        all_gaps_ms = []
        for ds in sample_dates:
            sym_path = DATA_DIR / "trades" / f"symbol={sym}" / f"date={ds}"
            files = list(sym_path.glob("*.parquet"))
            if not files:
                continue
            file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
            con = get_con()
            try:
                ts_arr = (
                    con.execute(
                        f"""
                    WITH deduped AS (
                        SELECT ts_ms,
                               ROW_NUMBER() OVER (PARTITION BY ts_ms, qty, price ORDER BY recv_ms) as rn
                        FROM read_parquet({file_list})
                    )
                    SELECT ts_ms FROM deduped WHERE rn = 1
                    GROUP BY ts_ms ORDER BY ts_ms
                """
                    )
                    .fetchdf()["ts_ms"]
                    .values
                )
                if len(ts_arr) > 1:
                    gaps = np.diff(ts_arr.astype(float))
                    gaps = gaps[gaps > 0]  # remove same-ms gaps
                    all_gaps_ms.extend(gaps.tolist())
            except Exception:
                pass
            finally:
                con.close()

        if not all_gaps_ms:
            result[sym] = {"error": "no_gaps"}
            continue

        arr = np.array(all_gaps_ms)
        median_gap_ms = float(np.median(arr))
        window_duration_ms = median_gap_ms * WINDOW_SIZE
        window_duration_min = window_duration_ms / 60_000.0

        result[sym] = {
            "median_inter_event_gap_ms": round(median_gap_ms, 1),
            "window_200_event_duration_min": round(window_duration_min, 1),
        }
        print(
            f"  {sym}: median_gap={median_gap_ms:.0f}ms, 200-event window={window_duration_min:.1f}min"
        )

    return result


# ---------------------------------------------------------------------------
# Section D: H500 base-rate stationarity for focus symbols
# ---------------------------------------------------------------------------
def compute_symbol_per_day_stats(symbol: str) -> list[dict]:
    """
    For each date of a symbol, load events and compute:
    - H500 up/down counts (within-day only, no cross-day labels)
    - first_vwap, last_vwap, n_events, per-event log return std

    Returns list of dicts, one per day. Much more memory-efficient than
    loading all events at once.
    """
    dates = list_dates(symbol)
    per_day = []
    for ds in dates:
        sym_path = DATA_DIR / "trades" / f"symbol={symbol}" / f"date={ds}"
        files = list(sym_path.glob("*.parquet"))
        if not files:
            per_day.append(
                {
                    "date": ds,
                    "up": 0,
                    "down": 0,
                    "n_events": 0,
                    "first_vwap": None,
                    "last_vwap": None,
                    "log_ret_std": None,
                }
            )
            continue
        file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        con = get_con()
        try:
            df = con.execute(
                f"""
                WITH deduped AS (
                    SELECT ts_ms, qty, price,
                           ROW_NUMBER() OVER (PARTITION BY ts_ms, qty, price ORDER BY recv_ms) as rn
                    FROM read_parquet({file_list})
                ),
                dr AS (SELECT ts_ms, qty, price FROM deduped WHERE rn = 1)
                SELECT ts_ms, SUM(qty*price)/NULLIF(SUM(qty),0) AS vwap
                FROM dr GROUP BY ts_ms ORDER BY ts_ms
            """
            ).fetchdf()
        except Exception:
            per_day.append(
                {
                    "date": ds,
                    "up": 0,
                    "down": 0,
                    "n_events": 0,
                    "first_vwap": None,
                    "last_vwap": None,
                    "log_ret_std": None,
                }
            )
            continue
        finally:
            con.close()

        if df.empty:
            per_day.append(
                {
                    "date": ds,
                    "up": 0,
                    "down": 0,
                    "n_events": 0,
                    "first_vwap": None,
                    "last_vwap": None,
                    "log_ret_std": None,
                }
            )
            continue

        vwap = df["vwap"].values
        n = len(vwap)
        up = down = 0
        if n > H500:
            up = int((vwap[H500:] > vwap[: n - H500]).sum())
            down = int((vwap[H500:] < vwap[: n - H500]).sum())

        vwap_pos = vwap[(vwap > 0) & np.isfinite(vwap)]
        first_vwap = float(vwap_pos[0]) if len(vwap_pos) > 0 else None
        last_vwap = float(vwap_pos[-1]) if len(vwap_pos) > 0 else None
        log_ret_std = None
        if len(vwap_pos) > 1:
            lr = np.diff(np.log(vwap_pos))
            lr = lr[np.isfinite(lr)]
            if len(lr) > 0:
                log_ret_std = float(np.std(lr))

        # Escalation check: base rate outside bounds for individual days
        if up + down > 0:
            day_rate = up / (up + down)
            if (
                day_rate < ESCALATION_RATE_BOUNDS[0] / 100
                or day_rate > ESCALATION_RATE_BOUNDS[1] / 100
            ):
                print(
                    f"    WARNING: {symbol} {ds} single-day H500 rate {day_rate:.3f} outside [{ESCALATION_RATE_BOUNDS[0]}%, {ESCALATION_RATE_BOUNDS[1]}%]"
                )

        per_day.append(
            {
                "date": ds,
                "up": up,
                "down": down,
                "n_events": n,
                "first_vwap": first_vwap,
                "last_vwap": last_vwap,
                "log_ret_std": log_ret_std,
            }
        )

    return per_day


def compute_rolling_h500_from_per_day(
    per_day: list[dict],
    window_days: int = 30,
    min_days_in_window: int = MIN_DAYS_IN_WINDOW,
) -> pd.DataFrame:
    """
    Compute 30-day rolling H500 base rate from per-day up/down counts.
    Requires min_days_in_window to avoid warm-up artifacts.
    Returns DataFrame: [end_date, base_rate, n_up, n_down, n_valid_labels, n_days_in_window]
    """
    df = pd.DataFrame(per_day)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    roll_up = df["up"].rolling(f"{window_days}D").sum()
    roll_down = df["down"].rolling(f"{window_days}D").sum()
    roll_n_days = df["up"].rolling(f"{window_days}D").count()

    records = []
    for idx in df.index:
        up = roll_up.get(idx, 0)
        down = roll_down.get(idx, 0)
        n_days = roll_n_days.get(idx, 0)
        total = up + down
        if pd.isna(up) or pd.isna(down) or total == 0:
            continue
        if n_days < min_days_in_window:
            continue  # skip warm-up period
        base_rate = float(up) / float(total)

        # Escalation check: base rate outside bounds
        if (
            base_rate < ESCALATION_RATE_BOUNDS[0] / 100
            or base_rate > ESCALATION_RATE_BOUNDS[1] / 100
        ):
            print(
                f"    ESCALATION: base rate {base_rate:.3f} outside [{ESCALATION_RATE_BOUNDS[0]}%, {ESCALATION_RATE_BOUNDS[1]}%] for window ending {idx.date()}"
            )
            raise RuntimeError(
                f"STOP: Rolling H500 base rate {base_rate:.3f} is outside [{ESCALATION_RATE_BOUNDS[0]}%, {ESCALATION_RATE_BOUNDS[1]}%] "
                f"at window ending {idx.date()}. Investigate label construction."
            )

        records.append(
            {
                "end_date": str(idx.date()),
                "base_rate": round(base_rate, 5),
                "n_up": int(up),
                "n_down": int(down),
                "n_valid_labels": int(total),
                "n_days_in_window": int(n_days),
            }
        )

    return pd.DataFrame(records)


def compute_total_return_from_per_day(per_day: list[dict], symbol: str) -> dict:
    """
    Compute total log return and drift-implied H500 base rate from per-day stats.
    """
    valid = [
        d for d in per_day if d["first_vwap"] is not None and d["last_vwap"] is not None
    ]
    if len(valid) < 2:
        return {}

    valid_sorted = sorted(valid, key=lambda x: x["date"])
    first_vwap = valid_sorted[0]["first_vwap"]
    last_vwap = valid_sorted[-1]["last_vwap"]

    if first_vwap <= 0 or last_vwap <= 0:
        return {}

    total_log_return = float(np.log(last_vwap / first_vwap))

    # Aggregate per-event log return std across all days
    stds = [d["log_ret_std"] for d in valid if d["log_ret_std"] is not None]
    n_events_list = [d["n_events"] for d in valid]

    if stds:
        # Weighted average std (weighted by n_events)
        weights = [d["n_events"] for d in valid if d["log_ret_std"] is not None]
        per_event_std = (
            float(np.average(stds, weights=weights))
            if sum(weights) > 0
            else float(np.mean(stds))
        )
    else:
        per_event_std = 0.001  # fallback

    sigma_h500 = per_event_std * np.sqrt(H500)
    n_total_events = sum(n_events_list)
    drift_per_event = total_log_return / max(n_total_events, 1)
    drift_per_h500 = drift_per_event * H500

    drift_implied = 0.5 + drift_per_h500 / (2.0 * max(sigma_h500, 1e-10))
    drift_implied = float(np.clip(drift_implied, 0.0, 1.0))

    return {
        "symbol": symbol,
        "first_vwap": round(first_vwap, 4),
        "last_vwap": round(last_vwap, 4),
        "total_log_return": round(total_log_return, 5),
        "per_event_std": round(per_event_std, 6),
        "sigma_per_h500": round(sigma_h500, 5),
        "drift_per_h500": round(drift_per_h500, 6),
        "drift_implied_base_rate": round(drift_implied, 5),
        "n_total_events": n_total_events,
        "n_days": len(valid),
    }


def analyze_focus_symbol(symbol: str) -> dict:
    """Full H500 stationarity analysis for one focus symbol. Uses per-day approach for memory efficiency."""
    print(f"\n--- {symbol} ---")
    t0 = time.time()

    per_day = compute_symbol_per_day_stats(symbol)
    if not per_day:
        print(f"  No data for {symbol}")
        return {"error": "no_data"}

    # Total return + drift model
    ret_stats = compute_total_return_from_per_day(per_day, symbol)
    n_total_events = ret_stats.get("n_total_events", 0)
    n_days = ret_stats.get("n_days", 0)
    print(f"  {n_total_events:,} events across {n_days} days")
    if ret_stats:
        print(
            f"  Total log return: {ret_stats.get('total_log_return', 'N/A'):.4f} "
            f"({ret_stats.get('first_vwap','?')} -> {ret_stats.get('last_vwap','?')})"
        )
        print(
            f"  Drift-implied H500 base rate: {ret_stats.get('drift_implied_base_rate', 'N/A'):.4f}"
        )

    # Full-period observed base rate
    all_up = sum(d["up"] for d in per_day)
    all_down = sum(d["down"] for d in per_day)
    total = all_up + all_down
    obs = {}
    if total > 0:
        observed = float(all_up) / total
        obs = {
            "observed_base_rate": observed,
            "n_up": all_up,
            "n_down": all_down,
            "n_total": total,
        }
        print(f"  Observed H500 base rate: {observed:.4f} (n={total:,})")

    # 30-day rolling base rate (excluding warm-up windows)
    rolling_df = compute_rolling_h500_from_per_day(
        per_day, window_days=30, min_days_in_window=MIN_DAYS_IN_WINDOW
    )
    rolling_result = {}
    if not rolling_df.empty:
        rates = rolling_df["base_rate"].values
        max_rate = float(rates.max())
        min_rate = float(rates.min())
        max_swing_pp = (max_rate - min_rate) * 100.0
        print(
            f"  30d rolling H500 (mature windows >=20 days): min={min_rate:.4f}, max={max_rate:.4f}, swing={max_swing_pp:.2f}pp"
        )

        # Escalation check for BTC/ETH: if MATURE-window swing > 8pp, label construction likely wrong
        if symbol in ("BTC", "ETH") and max_swing_pp > ESCALATION_SWING_BTC_ETH:
            raise RuntimeError(
                f"STOP: {symbol} mature-window rolling H500 swing = {max_swing_pp:.2f}pp > {ESCALATION_SWING_BTC_ETH}pp. "
                "This exceeds the expected range for liquid symbols with correct label construction. Investigate."
            )

        rolling_result = {
            "min_rolling_rate": round(min_rate, 5),
            "max_rolling_rate": round(max_rate, 5),
            "max_swing_pp": round(max_swing_pp, 3),
            "exceeds_3pp_threshold": max_swing_pp > ESCALATION_SWING_ANY,
            "n_mature_windows": len(rolling_df),
            "rolling_series": rolling_df.to_dict(orient="records"),
        }

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return {
        "n_events": n_total_events,
        "n_days": n_days,
        "total_return": ret_stats,
        "observed_full_period": obs,
        "rolling_h500": rolling_result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print("H500 Base-Rate Stationarity + Volume Claims Measurement")
    print(f"Pre-April window: {MIN_DATE} to {MAX_DATE}")
    print("=" * 60)

    results = {
        "meta": {
            "run_at": pd.Timestamp.now().isoformat(),
            "min_date": str(MIN_DATE),
            "max_date": str(MAX_DATE),
            "focus_symbols": FOCUS_SYMBOLS,
            "all_symbols": SYMBOLS,
            "h500": H500,
            "stride": STRIDE,
            "window_size": WINDOW_SIZE,
            "rolling_window_days": 30,
        }
    }

    # -----------------------------------------------------------------------
    # Section A: Events/day for all symbols
    # -----------------------------------------------------------------------
    events_per_day = compute_events_per_day_all_symbols()
    results["events_per_day"] = events_per_day

    # -----------------------------------------------------------------------
    # Section B: Total window count at stride=50
    # -----------------------------------------------------------------------
    window_counts = compute_total_window_count(events_per_day)
    results["window_counts"] = window_counts

    # -----------------------------------------------------------------------
    # Section C: 200-event window duration
    # -----------------------------------------------------------------------
    window_duration = compute_window_duration(events_per_day)
    results["window_duration"] = window_duration

    # -----------------------------------------------------------------------
    # Section D: H500 stationarity for focus symbols
    # -----------------------------------------------------------------------
    print("\n=== Section D: H500 stationarity (focus symbols) ===")
    stationarity = {}
    for sym in FOCUS_SYMBOLS:
        try:
            stationarity[sym] = analyze_focus_symbol(sym)
        except RuntimeError as e:
            print(f"\nESCALATION: {e}")
            results["escalation"] = str(e)
            results["stationarity"] = stationarity
            # Save partial results before stopping
            _save_results(results)
            raise

    results["stationarity"] = stationarity

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    verdict = _compute_verdict(
        stationarity, window_counts, window_duration, events_per_day
    )
    results["verdict"] = verdict
    print("\n" + "=" * 60)
    print("VERDICT:")
    print(verdict["summary"])
    print("=" * 60)

    elapsed_total = time.time() - t_start
    results["meta"]["elapsed_s"] = round(elapsed_total, 1)

    _save_results(results)
    _write_markdown_report(results)
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")
    print(f"Output: {OUT_DIR}/step0-base-rate-stationarity.json")
    print(f"Output: {OUT_DIR}/step0-base-rate-stationarity.md")


def _compute_verdict(
    stationarity: dict, window_counts: dict, window_duration: dict, events_per_day: dict
) -> dict:
    """Derive gate recommendation and spec correction list."""
    # Gate recommendation
    symbols_exceeding_3pp = []
    for sym, data in stationarity.items():
        if data.get("error"):
            continue
        rolling = data.get("rolling_h500", {})
        if rolling.get("exceeds_3pp_threshold", False):
            swing = rolling.get("max_swing_pp", 0)
            symbols_exceeding_3pp.append((sym, swing))

    if symbols_exceeding_3pp:
        gate_recommendation = (
            "Gate 4 at H500 REQUIRES balanced accuracy / F1, NOT raw accuracy. "
            f"Symbols exceeding 3pp swing: {', '.join(f'{s} ({sw:.2f}pp)' for s, sw in symbols_exceeding_3pp)}."
        )
    else:
        gate_recommendation = "Gate 4 at H500 can use raw accuracy — no focus symbol exceeded 3pp rolling swing."

    # Volume claim corrections
    spec_corrections = []
    grand_total = window_counts.get("grand_total", 0)
    spec_windows_claim = 3_500_000

    if abs(grand_total - spec_windows_claim) / spec_windows_claim > 0.20:
        direction = "under" if grand_total < spec_windows_claim else "over"
        spec_corrections.append(
            f"total_windows: spec claims ~{spec_windows_claim/1e6:.1f}M, measured {grand_total/1e6:.2f}M "
            f"({direction}estimate by {abs(grand_total - spec_windows_claim)/1e6:.2f}M / "
            f"{abs(grand_total - spec_windows_claim)/spec_windows_claim*100:.0f}%)"
        )

    # BTC events/day claim
    btc_data = events_per_day.get("BTC", {})
    btc_median = btc_data.get("median", None)
    spec_btc_events_claim = 28_000
    if (
        btc_median is not None
        and abs(btc_median - spec_btc_events_claim) / spec_btc_events_claim > 0.20
    ):
        spec_corrections.append(
            f"btc_events_per_day: spec claims ~{spec_btc_events_claim:,}, measured median={btc_median:.0f} "
            f"(p10={btc_data.get('p10',0):.0f}, p90={btc_data.get('p90',0):.0f})"
        )

    # BTC window duration claim (~10 min)
    btc_dur = window_duration.get("BTC", {})
    btc_dur_min = btc_dur.get("window_200_event_duration_min", None)
    spec_btc_duration_claim = 10.0  # minutes
    if (
        btc_dur_min is not None
        and abs(btc_dur_min - spec_btc_duration_claim) / spec_btc_duration_claim > 0.20
    ):
        spec_corrections.append(
            f"btc_200_event_duration: spec claims ~{spec_btc_duration_claim:.0f}min, measured {btc_dur_min:.1f}min"
        )

    summary_lines = [gate_recommendation]
    if spec_corrections:
        summary_lines.append("Spec volume claim corrections needed:")
        for c in spec_corrections:
            summary_lines.append(f"  - {c}")
    else:
        summary_lines.append(
            "All spec volume claims within 20% — no corrections needed."
        )

    return {
        "gate_recommendation": gate_recommendation,
        "symbols_exceeding_3pp": symbols_exceeding_3pp,
        "balanced_accuracy_required": len(symbols_exceeding_3pp) > 0,
        "spec_corrections_needed": spec_corrections,
        "summary": "\n".join(summary_lines),
    }


def _save_results(results: dict):
    """Write JSON output."""
    out_path = OUT_DIR / "step0-base-rate-stationarity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [saved JSON: {out_path}]")


def _write_markdown_report(results: dict):
    """Write the markdown report."""
    meta = results.get("meta", {})
    stationarity = results.get("stationarity", {})
    events_per_day = results.get("events_per_day", {})
    window_counts = results.get("window_counts", {})
    window_duration = results.get("window_duration", {})
    verdict = results.get("verdict", {})

    lines = []
    lines.append("# Step 0: H500 Base-Rate Stationarity + Volume Claims")
    lines.append("")
    lines.append(f"**Run at:** {meta.get('run_at', 'N/A')}")
    lines.append(
        f"**Window:** {meta.get('min_date')} to {meta.get('max_date')} (pre-April only)"
    )
    lines.append(f"**Elapsed:** {meta.get('elapsed_s', 'N/A')}s")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    gate_rec = verdict.get("gate_recommendation", "N/A")
    lines.append(f"**Gate 4 H500 recommendation:** {gate_rec}")
    lines.append("")
    spec_corr = verdict.get("spec_corrections_needed", [])
    if spec_corr:
        lines.append("**Spec volume claim corrections required:**")
        for c in spec_corr:
            lines.append(f"- {c}")
    else:
        lines.append(
            "**Spec volume claims:** All within 20% tolerance — no corrections needed."
        )
    lines.append("")

    # Focus symbol H500 stationarity table
    lines.append("## H500 Base-Rate Stationarity (Focus Symbols)")
    lines.append("")
    lines.append(
        "| Symbol | Total Return | Drift-Implied Base | Observed Base | Max Swing (pp) | Events/Day Median | Win 200-Event (min) |"
    )
    lines.append(
        "|--------|--------------|--------------------|---------------|----------------|-------------------|---------------------|"
    )

    for sym in FOCUS_SYMBOLS:
        data = stationarity.get(sym, {})
        if data.get("error"):
            lines.append(f"| {sym} | ERROR | — | — | — | — | — |")
            continue
        ret = data.get("total_return", {})
        obs = data.get("observed_full_period", {})
        rolling = data.get("rolling_h500", {})
        epd = events_per_day.get(sym, {})
        wd = window_duration.get(sym, {})

        total_ret = ret.get("total_log_return", "N/A")
        drift_implied = ret.get("drift_implied_base_rate", "N/A")
        observed = obs.get("observed_base_rate", "N/A")
        max_swing = rolling.get("max_swing_pp", "N/A")
        exceeds = rolling.get("exceeds_3pp_threshold", False)
        swing_flag = " *" if exceeds else ""
        epd_med = epd.get("median", "N/A")
        dur = wd.get("window_200_event_duration_min", "N/A")

        def fmt(v, decimals=4):
            return f"{v:.{decimals}f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {sym} | {fmt(total_ret,4)} | {fmt(drift_implied,4)} | {fmt(observed,4)} | "
            f"{fmt(max_swing,2)}{swing_flag} | {fmt(epd_med,0)} | {fmt(dur,1)} |"
        )

    lines.append("")
    lines.append(
        "\\* exceeds 3pp threshold — balanced accuracy required at Gate 4 H500"
    )
    lines.append("")

    # 30-day rolling series tables
    lines.append("## 30-Day Rolling H500 Base Rate (Focus Symbols)")
    lines.append("")
    for sym in FOCUS_SYMBOLS:
        data = stationarity.get(sym, {})
        if data.get("error"):
            continue
        rolling = data.get("rolling_h500", {})
        series = rolling.get("rolling_series", [])
        if not series:
            continue
        swing = rolling.get("max_swing_pp", 0)
        lines.append(f"### {sym} (max swing: {swing:.2f}pp)")
        lines.append("")
        # Print every 5th entry to keep report manageable
        lines.append("| End Date | Base Rate | N Valid | N Up | N Down |")
        lines.append("|----------|-----------|---------|------|--------|")
        for i, row in enumerate(series):
            if i % 5 == 0 or i == len(series) - 1:
                lines.append(
                    f"| {row['end_date']} | {row['base_rate']:.4f} | "
                    f"{row['n_valid_labels']:,} | {row['n_up']:,} | {row['n_down']:,} |"
                )
        lines.append("")

    # All-symbol events/day table
    lines.append("## Events/Day Distribution (All 25 Symbols)")
    lines.append("")
    lines.append("| Symbol | N Days | Median | P10 | P90 | Min | Max |")
    lines.append("|--------|--------|--------|-----|-----|-----|-----|")
    for sym in SYMBOLS:
        data = events_per_day.get(sym, {})
        if data.get("error"):
            lines.append(f"| {sym} | ERROR | — | — | — | — | — |")
            continue
        lines.append(
            f"| {sym} | {data['n_days']} | {data['median']:.0f} | "
            f"{data['p10']:.0f} | {data['p90']:.0f} | {data['min']:.0f} | {data['max']:.0f} |"
        )
    lines.append("")

    # Window count summary
    lines.append("## Total Window Count at Stride=50")
    lines.append("")
    grand = window_counts.get("grand_total", 0)
    spec_claim = 3_500_000
    lines.append(f"- **Measured total:** {grand:,}")
    lines.append(f"- **Spec claim:** ~{spec_claim:,}")
    diff_pct = (grand - spec_claim) / spec_claim * 100
    lines.append(f"- **Difference:** {diff_pct:+.1f}%")
    lines.append("")

    lines.append("| Symbol | Windows |")
    lines.append("|--------|---------|")
    per_sym = window_counts.get("per_symbol", {})
    for sym in SYMBOLS:
        lines.append(f"| {sym} | {per_sym.get(sym, 0):,} |")
    lines.append("")

    # Window duration summary
    lines.append("## 200-Event Window Duration")
    lines.append("")
    lines.append("| Symbol | Median Gap (ms) | Window Duration (min) |")
    lines.append("|--------|-----------------|----------------------|")
    for sym in SYMBOLS:
        data = window_duration.get(sym, {})
        if data.get("error"):
            lines.append(f"| {sym} | ERROR | — |")
            continue
        lines.append(
            f"| {sym} | {data.get('median_inter_event_gap_ms', 'N/A'):.0f} | "
            f"{data.get('window_200_event_duration_min', 'N/A'):.1f} |"
        )
    lines.append("")

    lines.append("## Spec Volume Claims vs Measured")
    lines.append("")
    btc_epd = events_per_day.get("BTC", {})
    btc_wd = window_duration.get("BTC", {})
    lines.append("| Claim | Spec Says | Measured | Delta |")
    lines.append("|-------|-----------|----------|-------|")
    lines.append(
        f"| BTC events/day | ~28,000 | {btc_epd.get('median', 'N/A'):.0f} (p10={btc_epd.get('p10',0):.0f}, p90={btc_epd.get('p90',0):.0f}) | {(btc_epd.get('median', 28000) - 28000) / 28000 * 100:+.0f}% |"
    )
    lines.append(
        f"| Total windows (stride=50) | ~3,500,000 | {grand:,} | {diff_pct:+.1f}% |"
    )
    btc_dur = btc_wd.get("window_200_event_duration_min", None)
    if btc_dur is not None:
        lines.append(
            f"| BTC 200-event window duration | ~10 min | {btc_dur:.1f} min | {(btc_dur - 10.0) / 10.0 * 100:+.0f}% |"
        )
    else:
        lines.append("| BTC 200-event window duration | ~10 min | N/A | N/A |")
    lines.append("")

    out_path = OUT_DIR / "step0-base-rate-stationarity.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [saved MD: {out_path}]")


if __name__ == "__main__":
    main()
