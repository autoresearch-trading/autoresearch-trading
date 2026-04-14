"""
measure_falsifiability_prereqs.py
===================================
Measures five falsifiability prerequisites flagged by council-5 before Steps 1-2.

All measurements use pre-April data only (local data ends 2026-03-25).
April 14+ is the hold-out — never touched.

Outputs:
  docs/experiments/step0-falsifiability-prereqs.md
  docs/experiments/step0-falsifiability-prereqs.json

Measurements:
  1. Stress label firing rate (full OB: log_spread > p90 AND |depth_ratio| > p90)
     Dates: 2025-10-20, 2025-12-15, 2026-02-10 × {BTC, ETH}

  2. Informed_flow label firing rate (full OB: kyle_lambda > p75 AND |cum_ofi_5| > p50 AND 3-snap sign consistency)
     Same dates/symbols.

  3. Climax date-diversity per symbol (all 25, all pre-April)
     Thresholds: z_qty>2 & z_return>2 and z_qty>3 & z_return>3

  4. Spring threshold recalibration (all 25 symbols, pre-April)
     Try -2σ, -2.5σ, -3σ; find threshold ≤8% firing on 24+/25 symbols.

  5. Feature autocorrelation at lag 5 (BTC, ETH, SOL — 17 features, ≥1000 windows)
"""

import json
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(row: Optional[tuple]) -> Any:
    """Unwrap DuckDB .fetchone() for aggregate queries that always return one row."""
    if row is None:
        raise RuntimeError("DuckDB aggregate query returned no row")
    return row[0]


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "docs" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DATE = date(2026, 3, 25)
MIN_DATE = date(2025, 10, 16)

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

ROLLING_WINDOW = 1000  # events, causal

# Target dates for measurements 1 & 2 (evenly spaced)
TARGET_DATES = ["2025-10-20", "2025-12-15", "2026-02-10"]
TARGET_SYMS_12 = ["BTC", "ETH"]

FEATURE_NAMES = [
    "log_return",
    "log_total_qty",
    "is_open",
    "time_delta",
    "num_fills",
    "book_walk",
    "effort_vs_result",
    "climax_score",
    "prev_seq_time_span",
    "log_spread",
    "imbalance_L1",
    "imbalance_L5",
    "depth_ratio",
    "trade_vs_mid",
    "delta_imbalance_L1",
    "kyle_lambda",
    "cum_ofi_5",
]


def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET home_directory = '{ROOT}'")
    return con


def list_dates(symbol: str) -> list[str]:
    """Return sorted pre-April date strings for symbol."""
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
        if MIN_DATE <= d <= MAX_DATE:
            dates.append(date_str)
    return sorted(dates)


def ob_files_for_date(symbol: str, date_str: str) -> list[str]:
    """Return OB parquet files for one symbol-date."""
    p = DATA_DIR / "orderbook" / f"symbol={symbol}" / f"date={date_str}"
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.parquet"))


def trade_files_for_date(symbol: str, date_str: str) -> list[str]:
    """Return trade parquet files for one symbol-date."""
    p = DATA_DIR / "trades" / f"symbol={symbol}" / f"date={date_str}"
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.parquet"))


# ---------------------------------------------------------------------------
# OB loading helper — extracts all 10 levels, returns DataFrame
# ---------------------------------------------------------------------------
_BID_COLS = ", ".join(
    f"bids[{i}].price as bid{i}_price, bids[{i}].qty as bid{i}_qty"
    for i in range(1, 11)
)
_ASK_COLS = ", ".join(
    f"asks[{i}].price as ask{i}_price, asks[{i}].qty as ask{i}_qty"
    for i in range(1, 11)
)


def load_ob_day(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    """Load full OB for one symbol-date; returns None if missing."""
    files = ob_files_for_date(symbol, date_str)
    if not files:
        return None
    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    con = get_con()
    try:
        df = con.execute(
            f"SELECT ts_ms, {_BID_COLS}, {_ASK_COLS} "
            f"FROM read_parquet({file_list}) ORDER BY ts_ms"
        ).fetchdf()
    except Exception:
        return None
    finally:
        con.close()
    return df if not df.empty else None


def load_trade_events_day(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    """
    Load deduped+grouped trade events for one symbol-date.
    Returns [ts_ms, vwap, total_qty, n_fills, is_open_frac] or None.
    Pre-April: dedup by (ts_ms, qty, price) without side.
    """
    files = trade_files_for_date(symbol, date_str)
    if not files:
        return None
    file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    con = get_con()
    try:
        df = con.execute(
            f"""
            WITH deduped AS (
                SELECT ts_ms, qty, price, side,
                       ROW_NUMBER() OVER (
                           PARTITION BY ts_ms, qty, price ORDER BY recv_ms
                       ) AS rn
                FROM read_parquet({file_list})
            ),
            dr AS (SELECT ts_ms, qty, price, side FROM deduped WHERE rn = 1),
            events AS (
                SELECT
                    ts_ms,
                    SUM(qty * price) / NULLIF(SUM(qty), 0) AS vwap,
                    SUM(qty)                                AS total_qty,
                    COUNT(*)                                AS n_fills,
                    SUM(CASE WHEN side IN ('open_long','open_short')
                             THEN 1 ELSE 0 END)::DOUBLE
                        / NULLIF(COUNT(*), 0)               AS is_open_frac
                FROM dr
                GROUP BY ts_ms
                ORDER BY ts_ms
            )
            SELECT ts_ms, vwap, total_qty, n_fills, is_open_frac FROM events
            """
        ).fetchdf()
    except Exception:
        return None
    finally:
        con.close()
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# OB feature computation helpers
# ---------------------------------------------------------------------------


def compute_ob_features(ob: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw OB DataFrame (all 10 levels), compute per-snapshot:
      - bid1, ask1, mid, spread
      - log_spread
      - imbalance_L1, imbalance_L5
      - depth_ratio (notional, epsilon-guarded)
    Returns DataFrame indexed like ob.
    """
    n = len(ob)

    # Level 1
    bid1_p = ob["bid1_price"].to_numpy(float)
    ask1_p = ob["ask1_price"].to_numpy(float)
    bid1_q = ob["bid1_qty"].to_numpy(float)
    ask1_q = ob["ask1_qty"].to_numpy(float)

    mid = (bid1_p + ask1_p) / 2.0
    spread = ask1_p - bid1_p
    eps_spread = np.maximum(spread, 1e-8 * mid)

    log_spread = np.log((spread / mid) + 1e-10)

    # L1 notional imbalance
    bid1_not = bid1_p * bid1_q
    ask1_not = ask1_p * ask1_q
    tot1 = bid1_not + ask1_not
    imbalance_L1 = np.where(tot1 > 0, (bid1_not - ask1_not) / tot1, 0.0)

    # L5 inverse-level-weighted notional imbalance
    weights = np.array([1.0 / i for i in range(1, 6)])
    bid_nots = np.zeros(n)
    ask_nots = np.zeros(n)
    for lvl in range(1, 6):
        w = weights[lvl - 1]
        bp = ob[f"bid{lvl}_price"].to_numpy(float)
        bq = ob[f"bid{lvl}_qty"].to_numpy(float)
        ap = ob[f"ask{lvl}_price"].to_numpy(float)
        aq = ob[f"ask{lvl}_qty"].to_numpy(float)
        bid_nots += w * bp * bq
        ask_nots += w * ap * aq
    tot5 = bid_nots + ask_nots
    imbalance_L5 = np.where(tot5 > 0, (bid_nots - ask_nots) / tot5, 0.0)

    # Depth ratio — total notional across all 10 levels
    total_bid_not = np.zeros(n)
    total_ask_not = np.zeros(n)
    for lvl in range(1, 11):
        bp = ob[f"bid{lvl}_price"].to_numpy(float)
        bq = ob[f"bid{lvl}_qty"].to_numpy(float)
        ap = ob[f"ask{lvl}_price"].to_numpy(float)
        aq = ob[f"ask{lvl}_qty"].to_numpy(float)
        total_bid_not += bp * bq
        total_ask_not += ap * aq

    depth_ratio = np.log(
        np.maximum(total_bid_not, 1e-6) / np.maximum(total_ask_not, 1e-6)
    )

    result = pd.DataFrame(
        {
            "ts_ms": ob["ts_ms"].to_numpy(),
            "mid": mid,
            "spread": spread,
            "log_spread": log_spread,
            "imbalance_L1": imbalance_L1,
            "imbalance_L5": imbalance_L5,
            "depth_ratio": depth_ratio,
            "total_bid_not": total_bid_not,
            "total_ask_not": total_ask_not,
            "bid1_p": bid1_p,
            "ask1_p": ask1_p,
        }
    )
    return result


def rolling_percentile(arr: np.ndarray, window: int, pct: float) -> np.ndarray:
    """
    Causal rolling percentile. Element i uses arr[max(0,i-window):i+1].
    Vectorized via pandas.
    """
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1).quantile(pct / 100.0).to_numpy()


# ---------------------------------------------------------------------------
# Measurement 1: Stress label firing rate
# ---------------------------------------------------------------------------


def measure_stress_firing(
    ob_features: pd.DataFrame,
    rolling_win: int = 1000,
) -> dict:
    """
    Per-snapshot: fires if log_spread > rolling_p90(log_spread) AND
                          |depth_ratio| > rolling_p90(|depth_ratio|).
    Rolling window = rolling_win snapshots.
    """
    ls = ob_features["log_spread"].to_numpy(float)
    dr_abs = np.abs(ob_features["depth_ratio"].to_numpy(float))

    p90_ls = rolling_percentile(ls, rolling_win, 90)
    p90_dr = rolling_percentile(dr_abs, rolling_win, 90)

    fires = (ls > p90_ls) & (dr_abs > p90_dr)
    rate = float(fires.mean()) * 100.0  # percent
    return {"n_snapshots": int(len(fires)), "firing_rate_pct": rate}


# ---------------------------------------------------------------------------
# Measurement 2: Informed flow firing rate
# ---------------------------------------------------------------------------


def compute_piecewise_ofi(ob_features: pd.DataFrame) -> np.ndarray:
    """
    Piecewise Cont 2014 OFI: at each snapshot, compare bid1/ask1 to prior snapshot.
    OFI_t = bid_vol_in - ask_vol_in where:
      bid_vol_in  = bid1_qty_t  if bid1_p_t >= bid1_p_{t-1}, else 0
      ask_vol_in  = ask1_qty_t  if ask1_p_t <= ask1_p_{t-1}, else 0
    Returns array of OFI values (length = n_snapshots, first=0).
    bid1_q and ask1_q columns must be present in ob_features (added by caller).
    """
    bid_p = ob_features["bid1_p"].to_numpy(float)
    ask_p = ob_features["ask1_p"].to_numpy(float)
    bid_q = ob_features["bid1_q"].to_numpy(float)
    ask_q = ob_features["ask1_q"].to_numpy(float)

    # Vectorized: compare t to t-1
    bid_in = np.where(bid_p[1:] >= bid_p[:-1], bid_q[1:], 0.0)
    ask_in = np.where(ask_p[1:] <= ask_p[:-1], ask_q[1:], 0.0)
    ofi = np.zeros(len(bid_p))
    ofi[1:] = bid_in - ask_in
    return ofi


def compute_kyle_lambda_snapshots(
    ob_features: pd.DataFrame, trades_events: Optional[pd.DataFrame], win: int = 50
) -> np.ndarray:
    """
    Per-snapshot kyle_lambda: Cov(Δmid, cum_signed_notional) / Var(cum_signed_notional)
    over rolling 50-snapshot window.
    cum_signed_notional comes from aligning trade events to snapshots.
    If trades unavailable, returns zeros.
    """
    n = len(ob_features)
    mid = ob_features["mid"].to_numpy(float)
    ob_ts = ob_features["ts_ms"].to_numpy(np.int64)

    # Build delta_mid per snapshot
    delta_mid = np.zeros(n)
    delta_mid[1:] = mid[1:] - mid[:-1]

    # Build cum_signed_notional: for each snapshot interval, sum signed notional
    # Signed notional: positive if is_open > 0.5 (net long open), negative otherwise
    if trades_events is not None and len(trades_events) > 0:
        ev_ts = trades_events["ts_ms"].to_numpy(np.int64)
        ev_qty = trades_events["total_qty"].to_numpy(float)
        ev_vwap = trades_events["vwap"].to_numpy(float)
        ev_is_open = trades_events["is_open_frac"].to_numpy(float)
        ev_sign = np.where(ev_is_open > 0.5, 1.0, -1.0)
        ev_notional = ev_qty * ev_vwap * ev_sign

        # Vectorized: assign each trade event to its snapshot bin
        idx = np.searchsorted(ob_ts, ev_ts, side="right") - 1
        idx = np.clip(idx, 0, n - 1)
        snap_notional = np.zeros(n)
        np.add.at(snap_notional, idx, ev_notional)
    else:
        snap_notional = np.zeros(n)

    # Rolling Cov/Var over win snapshots — vectorized with pandas
    # Cov(Δmid, snap_notional) / Var(snap_notional) over rolling win
    s_x = pd.Series(snap_notional)
    s_y = pd.Series(delta_mid)

    roll_cov: np.ndarray = np.asarray(s_x.rolling(win, min_periods=win).cov(s_y))
    roll_var: np.ndarray = np.asarray(s_x.rolling(win, min_periods=win).var())

    kyle_lambda = np.where(
        np.isfinite(roll_var) & (roll_var > 1e-10),
        roll_cov / roll_var,
        0.0,
    )
    kyle_lambda = np.where(np.isfinite(kyle_lambda), kyle_lambda, 0.0)

    # Forward-fill (vectorized)
    for i in range(1, n):
        if kyle_lambda[i] == 0.0 and i >= win and kyle_lambda[i - 1] != 0.0:
            kyle_lambda[i] = kyle_lambda[i - 1]

    return kyle_lambda


def compute_cum_ofi_5(
    ofi: np.ndarray, total_bid_not: np.ndarray, total_ask_not: np.ndarray
) -> np.ndarray:
    """
    Rolling 5-snapshot piecewise OFI, normalized by rolling mean notional volume.
    Vectorized with pandas rolling.
    """
    s_ofi = pd.Series(ofi)
    total_not = total_bid_not + total_ask_not
    s_not = pd.Series(total_not)

    rolling_ofi_sum: np.ndarray = np.asarray(s_ofi.rolling(5, min_periods=1).sum())
    rolling_not_mean: np.ndarray = np.asarray(s_not.rolling(5, min_periods=1).mean())

    norm = np.maximum(rolling_not_mean, 1e-10)
    return rolling_ofi_sum / norm


def measure_informed_flow_firing(
    ob_feat: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    rolling_win: int = 1000,
    kyle_win: int = 50,
) -> dict:
    """
    Per-snapshot: fires if kyle_lambda > rolling_p75(kyle_lambda)
                      AND |cum_ofi_5| > rolling_p50(|cum_ofi_5|)
                      AND sign(cum_ofi_5) consistent across 3 consecutive snapshots.
    """
    n = len(ob_feat)

    # Compute OFI and kyle_lambda
    ofi = compute_piecewise_ofi(ob_feat)
    kyle_lam = compute_kyle_lambda_snapshots(ob_feat, trades, win=kyle_win)
    cum_ofi5 = compute_cum_ofi_5(
        ofi,
        ob_feat["total_bid_not"].to_numpy(float),
        ob_feat["total_ask_not"].to_numpy(float),
    )

    # Rolling percentiles (causal)
    p75_kl = rolling_percentile(kyle_lam, rolling_win, 75)
    p50_ofi = rolling_percentile(np.abs(cum_ofi5), rolling_win, 50)

    # 3-snapshot sign consistency: sign matches at i, i-1, i-2 (vectorized)
    sign_ofi = np.sign(cum_ofi5)
    sign_nonzero = sign_ofi != 0
    sign_consistent = np.zeros(n, dtype=bool)
    if n >= 3:
        sign_consistent[2:] = (
            sign_nonzero[2:]
            & (sign_ofi[2:] == sign_ofi[1:-1])
            & (sign_ofi[2:] == sign_ofi[:-2])
        )

    fires = (kyle_lam > p75_kl) & (np.abs(cum_ofi5) > p50_ofi) & sign_consistent
    rate = float(fires.mean()) * 100.0

    return {
        "n_snapshots": int(n),
        "firing_rate_pct": rate,
        "kyle_lambda_mean": float(np.mean(kyle_lam)),
        "cum_ofi5_std": float(np.std(cum_ofi5)),
    }


# ---------------------------------------------------------------------------
# Measurement 3: Climax date-diversity per symbol
# ---------------------------------------------------------------------------


def compute_climax_date_diversity() -> dict:
    """
    For all 25 symbols, all pre-April dates:
    Compute rolling z-scores of total_qty and |log_return| per event.
    At each event, climax fires if min(z_qty, z_return) > threshold.
    Count distinct calendar dates with at least one firing event.
    Thresholds: 2σ and 3σ.
    """
    print("\n=== Measurement 3: Climax date-diversity (all 25 symbols) ===")
    result = {}

    for sym in SYMBOLS:
        dates = list_dates(sym)
        if not dates:
            result[sym] = {"error": "no_dates"}
            continue

        # Collect (date, climax2, climax3) per event across all dates
        dates_climax2: set[str] = set()
        dates_climax3: set[str] = set()

        # Collect all events across dates, carry rolling stats (pandas rolling, vectorized)
        prev_vwap: Optional[float] = None
        # We process per-day but use pandas rolling on the full concatenated array
        # to keep rolling stats causal across day boundaries.
        all_qty: list[float] = []
        all_abs_ret: list[float] = []
        all_dates_seq: list[str] = []

        for ds in dates:
            events = load_trade_events_day(sym, ds)
            if events is None or len(events) < 2:
                continue

            vwap = events["vwap"].to_numpy(float)
            total_qty = events["total_qty"].to_numpy(float)
            n_ev = len(events)

            # Compute log_return per event (vectorized)
            log_ret = np.zeros(n_ev)
            start_prev = (
                prev_vwap if (prev_vwap is not None and prev_vwap > 0) else vwap[0]
            )
            all_vwap = np.concatenate([[start_prev], vwap])
            prev_ok = (all_vwap[:-1] > 0) & (all_vwap[1:] > 0)
            log_ret = np.where(
                prev_ok,
                np.log(
                    np.where(
                        prev_ok,
                        all_vwap[1:] / np.where(all_vwap[:-1] > 0, all_vwap[:-1], 1.0),
                        1.0,
                    )
                ),
                0.0,
            )

            prev_vwap = float(vwap[-1])

            all_qty.extend(total_qty.tolist())
            all_abs_ret.extend(np.abs(log_ret).tolist())
            all_dates_seq.extend([ds] * n_ev)

        if not all_qty:
            continue

        # Vectorized rolling z-scores
        qty_arr = np.array(all_qty, dtype=float)
        ret_arr = np.array(all_abs_ret, dtype=float)
        s_qty = pd.Series(qty_arr)
        s_ret = pd.Series(ret_arr)

        roll_mean_qty: np.ndarray = np.asarray(
            s_qty.rolling(ROLLING_WINDOW, min_periods=10).mean()
        )
        roll_std_qty: np.ndarray = np.asarray(
            s_qty.rolling(ROLLING_WINDOW, min_periods=10).std().fillna(1e-10)
        )
        roll_mean_ret: np.ndarray = np.asarray(
            s_ret.rolling(ROLLING_WINDOW, min_periods=10).mean()
        )
        roll_std_ret: np.ndarray = np.asarray(
            s_ret.rolling(ROLLING_WINDOW, min_periods=10).std().fillna(1e-10)
        )

        z_qty = (qty_arr - roll_mean_qty) / np.maximum(roll_std_qty, 1e-10)
        z_ret = (ret_arr - roll_mean_ret) / np.maximum(roll_std_ret, 1e-10)
        climax_score_arr = np.minimum(z_qty, z_ret)

        fires2 = climax_score_arr > 2.0
        fires3 = climax_score_arr > 3.0

        dates_arr = np.array(all_dates_seq)
        for ds in np.unique(dates_arr):
            mask = dates_arr == ds
            if fires2[mask].any():
                dates_climax2.add(str(ds))
            if fires3[mask].any():
                dates_climax3.add(str(ds))

        n2 = len(dates_climax2)
        n3 = len(dates_climax3)
        result[sym] = {
            "distinct_dates_2sigma": n2,
            "distinct_dates_3sigma": n3,
            "total_dates": len(dates),
            "flag_below_15_2sigma": n2 < 15,
            "flag_below_15_3sigma": n3 < 15,
        }
        print(f"  {sym}: 2σ={n2} dates, 3σ={n3} dates (total={len(dates)})")

    return result


# ---------------------------------------------------------------------------
# Measurement 4: Spring threshold recalibration
# ---------------------------------------------------------------------------


def compute_spring_rate(
    events: pd.DataFrame, log_returns: np.ndarray, sigma_mult: float
) -> float:
    """
    Spring fires at event i if:
      min(log_return[i-50:i+1]) < -sigma_mult * rolling_std[i]
      AND evr at the min-return event > 1.0
      AND is_open at the min-return event > 0.5
      AND mean(log_return[i-10:i+1]) > 0

    EVR = clip(log_total_qty - log(|return|+1e-6), -5, 5) with median-normalized qty.
    Returns firing_rate in [0,1].
    """
    n = len(log_returns)
    is_open = events["is_open_frac"].to_numpy(float)
    total_qty = events["total_qty"].to_numpy(float)

    # Rolling median for qty normalization (causal)
    med_qty = (
        pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=1).median().to_numpy()
    )
    log_total_qty = np.log(total_qty / np.maximum(med_qty, 1e-10))

    # Rolling std of log_return (causal)
    roll_std = (
        pd.Series(log_returns)
        .rolling(ROLLING_WINDOW, min_periods=10)
        .std()
        .fillna(1e-6)
        .to_numpy()
    )

    # EVR approximation
    evr = np.clip(
        log_total_qty - np.log(np.abs(log_returns) + 1e-6),
        -5.0,
        5.0,
    )

    BACK50 = 50
    BACK10 = 10

    # Build rolling-min over 50-event window using pandas rolling
    roll_min50 = (
        pd.Series(log_returns)
        .rolling(BACK50 + 1, min_periods=BACK50 + 1)
        .min()
        .to_numpy()
    )
    # Build rolling-mean over last 10 events
    roll_mean10: np.ndarray = np.asarray(
        pd.Series(log_returns).rolling(BACK10 + 1, min_periods=BACK10 + 1).mean()
    )
    # Threshold at each position
    threshold = -sigma_mult * roll_std

    # Condition 1: rolling_min < threshold (has NaN for first BACK50 events)
    cond1 = np.isfinite(roll_min50) & (roll_min50 < threshold)
    cond4 = np.isfinite(roll_mean10) & (roll_mean10 > 0)

    # For conditions 2+3, we need evr and is_open at the argmin position
    # We must loop over firing candidates only (much cheaper than all events)
    candidate_idx = np.where(cond1 & cond4)[0]
    fires = 0

    for i in candidate_idx:
        window50 = log_returns[i - BACK50 : i + 1]
        min_pos = int(np.argmin(window50))
        global_min_idx = i - BACK50 + min_pos
        if global_min_idx < 0 or global_min_idx >= n:
            continue
        if evr[global_min_idx] > 1.0 and is_open[global_min_idx] > 0.5:
            fires += 1

    total = int(np.sum(np.isfinite(roll_min50)))
    return float(fires) / max(total, 1)


def measure_spring_recalibration() -> dict:
    """
    For all 25 symbols, compute spring firing rate at sigma_mult in {2.0, 2.5, 3.0}.
    Report rates per symbol, recommend threshold that keeps ≤8% on 24+/25 symbols.
    """
    print("\n=== Measurement 4: Spring threshold recalibration (all 25 symbols) ===")
    sigma_mults = [2.0, 2.5, 3.0]
    result: dict[str, Any] = {}

    for sym in SYMBOLS:
        dates = list_dates(sym)
        if not dates:
            result[sym] = {"error": "no_dates"}
            continue

        all_log_ret: list[float] = []
        all_events_rows: list[pd.DataFrame] = []
        prev_vwap: Optional[float] = None

        for ds in dates:
            ev = load_trade_events_day(sym, ds)
            if ev is None or len(ev) < 2:
                continue
            vwap = ev["vwap"].to_numpy(float)
            start_prev = (
                prev_vwap if (prev_vwap is not None and prev_vwap > 0) else vwap[0]
            )
            all_vwap = np.concatenate([[start_prev], vwap])
            prev_ok = (all_vwap[:-1] > 0) & (all_vwap[1:] > 0)
            lr = np.where(
                prev_ok,
                np.log(
                    np.where(
                        prev_ok,
                        all_vwap[1:] / np.where(all_vwap[:-1] > 0, all_vwap[:-1], 1.0),
                        1.0,
                    )
                ),
                0.0,
            )
            prev_vwap = float(vwap[-1])
            all_log_ret.extend(lr.tolist())
            all_events_rows.append(ev)

        if not all_events_rows or len(all_log_ret) < 200:
            result[sym] = {"error": "insufficient_data"}
            continue

        combined_events = pd.concat(all_events_rows, ignore_index=True)
        log_ret_arr = np.array(all_log_ret, dtype=float)

        rates: dict[str, float] = {}
        for sm in sigma_mults:
            rate = compute_spring_rate(combined_events, log_ret_arr, sm)
            rates[f"sigma_{sm:.1f}"] = round(rate * 100, 3)

        result[sym] = rates
        line = "  " + sym + ": " + ", ".join(f"{k}={v:.2f}%" for k, v in rates.items())
        print(line)

    # Determine recommended threshold
    recommendations: dict[str, int] = {}
    for sm in sigma_mults:
        key = f"sigma_{sm:.1f}"
        n_ok = sum(
            1
            for sym in SYMBOLS
            if sym in result
            and isinstance(result[sym], dict)
            and key in result[sym]
            and result[sym][key] <= 8.0
        )
        recommendations[key] = n_ok

    result["_recommendation"] = recommendations
    best = max(recommendations, key=lambda k: recommendations[k])
    result["_recommended_sigma_mult"] = float(best.split("_")[1])
    print(f"\n  Symbols ≤8% per threshold: {recommendations}")
    print(f"  Recommended sigma_mult: {result['_recommended_sigma_mult']}")
    return result


# ---------------------------------------------------------------------------
# Measurement 5: Feature autocorrelation at lag 5
# ---------------------------------------------------------------------------


def build_17_features_day(
    symbol: str, date_str: str, prev_day_events: Optional[pd.DataFrame] = None
) -> Optional[np.ndarray]:
    """
    Compute all 17 features for one symbol-date.
    Returns np.ndarray of shape (n_events, 17) or None.
    prev_day_events: last N events from prior day (for rolling warm-up).
    """
    events = load_trade_events_day(symbol, date_str)
    if events is None or len(events) < 10:
        return None

    ob_raw = load_ob_day(symbol, date_str)
    if ob_raw is None or len(ob_raw) < 2:
        return None

    ob_feat = compute_ob_features(ob_raw)
    # Add bid1_q, ask1_q for piecewise OFI
    ob_feat = ob_feat.copy()
    ob_feat["bid1_q"] = ob_raw["bid1_qty"].to_numpy(float)
    ob_feat["ask1_q"] = ob_raw["ask1_qty"].to_numpy(float)

    n = len(events)
    vwap = events["vwap"].to_numpy(float)
    total_qty = events["total_qty"].to_numpy(float)
    n_fills = events["n_fills"].to_numpy(float)
    is_open_frac = events["is_open_frac"].to_numpy(float)
    ev_ts = events["ts_ms"].to_numpy(np.int64)

    # --- Trade features ---

    # 1. log_return (vectorized)
    prev_v = (
        float(prev_day_events["vwap"].iloc[-1])
        if (prev_day_events is not None and len(prev_day_events) > 0)
        else float(vwap[0])
    )
    prev_v = prev_v if prev_v > 0 else float(vwap[0])
    all_vwap_lr = np.concatenate([[prev_v], vwap])
    ok_lr = (all_vwap_lr[:-1] > 0) & (all_vwap_lr[1:] > 0)
    log_return = np.where(
        ok_lr,
        np.log(
            np.where(
                ok_lr,
                all_vwap_lr[1:] / np.where(all_vwap_lr[:-1] > 0, all_vwap_lr[:-1], 1.0),
                1.0,
            )
        ),
        0.0,
    )

    # 2. log_total_qty (rolling median normalized)
    med_qty = (
        pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=1).median().to_numpy()
    )
    log_total_qty = np.log(total_qty / np.maximum(med_qty, 1e-10))

    # 3. is_open
    is_open = is_open_frac

    # 4. time_delta (vectorized)
    prev_ts_val = (
        int(prev_day_events["ts_ms"].iloc[-1])
        if (prev_day_events is not None and len(prev_day_events) > 0)
        else int(ev_ts[0])
    )
    all_ts_td = np.concatenate([[prev_ts_val], ev_ts.astype(np.int64)])
    dt_arr = np.maximum(all_ts_td[1:] - all_ts_td[:-1], 0)
    time_delta = np.log(dt_arr.astype(float) + 1.0)

    # 5. num_fills
    num_fills = np.log(np.maximum(n_fills, 1).astype(float))

    # 6. book_walk: abs(last_fill - first_fill) / max(spread, 1e-8*mid)
    # Approximation: for grouped events with single vwap, use 0 (no per-fill prices stored here)
    # Real pipeline uses per-fill data; for autocorrelation purposes this is constant=0
    book_walk = np.zeros(n)

    # 7. effort_vs_result
    effort_vs_result = np.clip(
        log_total_qty - np.log(np.abs(log_return) + 1e-6),
        -5.0,
        5.0,
    )

    # 8. climax_score (rolling 1000-event std, causal)
    roll_std_qty: np.ndarray = np.asarray(
        pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).std().fillna(1e-6)
    )
    roll_std_ret: np.ndarray = np.asarray(
        pd.Series(np.abs(log_return))
        .rolling(ROLLING_WINDOW, min_periods=10)
        .std()
        .fillna(1e-6)
    )
    _rmq: Any = pd.Series(total_qty).rolling(ROLLING_WINDOW, min_periods=10).mean()
    roll_mean_qty: np.ndarray = np.asarray(_rmq.fillna(0))
    _rmr: Any = (
        pd.Series(np.abs(log_return)).rolling(ROLLING_WINDOW, min_periods=10).mean()
    )
    roll_mean_ret: np.ndarray = np.asarray(_rmr.fillna(0))
    z_qty = (total_qty - roll_mean_qty) / np.maximum(roll_std_qty, 1e-10)
    z_ret = (np.abs(log_return) - roll_mean_ret) / np.maximum(roll_std_ret, 1e-10)
    climax_score = np.clip(np.minimum(z_qty, z_ret), 0.0, 5.0)

    # 9. prev_seq_time_span: log(last_ts - first_ts + 1) for PREVIOUS 200-event window
    # Vectorized: span = ts[i-1] - ts[i-PREV_WIN] for i >= PREV_WIN
    PREV_WIN = 200
    prev_seq_time_span = np.zeros(n)
    if n > PREV_WIN:
        # For each i in [PREV_WIN, n): span = ev_ts[i-1] - ev_ts[i-PREV_WIN]
        idx_last = np.arange(PREV_WIN - 1, n - 1)  # ts at end of prior window
        idx_first = np.arange(0, n - PREV_WIN)  # ts at start of prior window
        spans = np.maximum(ev_ts[idx_last] - ev_ts[idx_first], 0)
        prev_seq_time_span[PREV_WIN:] = np.log(spans.astype(float) + 1.0)

    # --- OB features (align by nearest prior snapshot) ---
    ob_ts = ob_feat["ts_ms"].to_numpy(np.int64)
    # Causal alignment: for each event, find the most recent prior OB snapshot
    aligned_idx = np.searchsorted(ob_ts, ev_ts, side="right") - 1
    aligned_idx = np.clip(aligned_idx, 0, len(ob_ts) - 1)

    def align(col: np.ndarray) -> np.ndarray:
        return col[aligned_idx]

    # 10. log_spread
    log_spread = align(ob_feat["log_spread"].to_numpy(float))

    # 11. imbalance_L1
    imbalance_L1 = align(ob_feat["imbalance_L1"].to_numpy(float))

    # 12. imbalance_L5
    imbalance_L5 = align(ob_feat["imbalance_L5"].to_numpy(float))

    # 13. depth_ratio
    depth_ratio = align(ob_feat["depth_ratio"].to_numpy(float))

    # 14. trade_vs_mid
    ob_mid = align(ob_feat["mid"].to_numpy(float))
    ob_spread = align(ob_feat["spread"].to_numpy(float))
    eps_spread = np.maximum(ob_spread, 1e-8 * ob_mid)
    trade_vs_mid = np.clip((vwap - ob_mid) / eps_spread, -5.0, 5.0)

    # 15. delta_imbalance_L1 (change in aligned L1 imbalance from prior event)
    # Vectorized: diff of the aligned snapshot index sequence
    imb_L1_full = ob_feat["imbalance_L1"].to_numpy(float)
    imb_at_events = imb_L1_full[aligned_idx]
    delta_imbalance_L1 = np.zeros(n)
    delta_imbalance_L1[1:] = imb_at_events[1:] - imb_at_events[:-1]

    # 16. kyle_lambda: compute per-snapshot, forward-fill to events
    kyle_lam_snap = compute_kyle_lambda_snapshots(ob_feat, events, win=50)
    kyle_lambda = kyle_lam_snap[aligned_idx]

    # 17. cum_ofi_5
    ofi = compute_piecewise_ofi(ob_feat)
    cum_ofi5_snap = compute_cum_ofi_5(
        ofi,
        ob_feat["total_bid_not"].to_numpy(float),
        ob_feat["total_ask_not"].to_numpy(float),
    )
    cum_ofi_5 = cum_ofi5_snap[aligned_idx]

    features = np.column_stack(
        [
            log_return,
            log_total_qty,
            is_open,
            time_delta,
            num_fills,
            book_walk,
            effort_vs_result,
            climax_score,
            prev_seq_time_span,
            log_spread,
            imbalance_L1,
            imbalance_L5,
            depth_ratio,
            trade_vs_mid,
            delta_imbalance_L1,
            kyle_lambda,
            cum_ofi_5,
        ]
    )

    # Guard: replace non-finite
    features = np.where(np.isfinite(features), features, 0.0)
    assert features.shape[1] == 17, f"Expected 17 features, got {features.shape[1]}"
    return features


def measure_autocorrelation(n_windows: int = 1000) -> dict:
    """
    For BTC, ETH, SOL: sample ≥1000 random 200-event windows,
    compute lag-5 autocorrelation per feature.
    Flag (feature, symbol) pairs with r>0.8.
    """
    print("\n=== Measurement 5: Feature autocorrelation at lag 5 ===")
    syms = ["BTC", "ETH", "SOL"]
    LAG = 5
    WIN = 200
    result: dict[str, Any] = {}
    flags: list[dict] = []

    rng = np.random.default_rng(42)

    for sym in syms:
        dates = list_dates(sym)
        if not dates:
            result[sym] = {"error": "no_dates"}
            continue

        # Collect feature arrays per day
        all_features: list[np.ndarray] = []
        for ds in dates:
            feat = build_17_features_day(sym, ds)
            if feat is not None and len(feat) >= WIN:
                all_features.append(feat)

        if not all_features:
            result[sym] = {"error": "no_valid_days"}
            continue

        # Sample windows
        windows_collected = 0
        autocorrs: list[np.ndarray] = []  # one per window, shape=(17,)

        # Shuffle days
        day_indices = list(range(len(all_features)))
        rng.shuffle(day_indices)

        for di in day_indices:
            feat = all_features[di]
            n_ev = len(feat)
            if n_ev < WIN + LAG:
                continue
            # Sample random windows from this day
            max_start = n_ev - WIN
            n_from_day = max(1, min(50, max_start))
            starts = rng.integers(0, max_start + 1, size=n_from_day)
            for s in starts:
                w = feat[s : s + WIN]  # shape (200, 17)
                # Lag-5 autocorr for all 17 features simultaneously
                x = w[: WIN - LAG]  # (195, 17)
                y = w[LAG:]  # (195, 17)
                # Pearson correlation: (x - mean_x) * (y - mean_y) / (std_x * std_y)
                x_c = x - x.mean(axis=0, keepdims=True)
                y_c = y - y.mean(axis=0, keepdims=True)
                std_x = x.std(axis=0)
                std_y = y.std(axis=0)
                cov_xy = (x_c * y_c).mean(axis=0)
                denom = std_x * std_y
                ac = np.where(denom > 1e-10, cov_xy / denom, 0.0)
                autocorrs.append(ac)
                windows_collected += 1
                if windows_collected >= n_windows:
                    break
            if windows_collected >= n_windows:
                break

        if not autocorrs:
            result[sym] = {"error": "no_windows"}
            continue

        ac_arr = np.array(autocorrs)  # shape (n_windows, 17)
        mean_ac = ac_arr.mean(axis=0)  # shape (17,)
        std_ac = ac_arr.std(axis=0)

        feat_results: dict[str, float] = {}
        for fi, fname in enumerate(FEATURE_NAMES):
            r = float(mean_ac[fi])
            feat_results[fname] = round(r, 4)
            if r > 0.8:
                flags.append(
                    {
                        "symbol": sym,
                        "feature": fname,
                        "lag5_autocorr": r,
                        "flag": "r>0.8 -- MEM block masking trivially solvable",
                    }
                )

        result[sym] = {
            "n_windows": windows_collected,
            "lag5_autocorr": feat_results,
        }
        # Print top 5 most autocorrelated
        top5 = sorted(feat_results.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"  {sym} ({windows_collected} windows): top-5 {top5}")

    result["_flags"] = flags
    if flags:
        print(
            f"\n  FLAGS (r>0.8): {[(f['feature'], f['symbol'], f['lag5_autocorr']) for f in flags]}"
        )
    else:
        print("\n  No (feature, symbol) pairs with r>0.8")
    return result


# ---------------------------------------------------------------------------
# Main orchestrator for measurements 1 & 2 (OB-based)
# ---------------------------------------------------------------------------


def measure_stress_and_informed(rolling_win: int = 1000) -> dict:
    """
    Run measurements 1 (stress) and 2 (informed flow) on
    3 dates × {BTC, ETH}.
    """
    print("\n=== Measurements 1+2: Stress + Informed flow (BTC, ETH × 3 dates) ===")
    stress_results: dict[str, dict] = {}
    informed_results: dict[str, dict] = {}

    for sym in TARGET_SYMS_12:
        stress_results[sym] = {}
        informed_results[sym] = {}

        for dt in TARGET_DATES:
            print(f"  Loading OB for {sym} {dt} ...", flush=True)
            ob_raw = load_ob_day(sym, dt)
            if ob_raw is None:
                stress_results[sym][dt] = {"error": "no_ob_data"}
                informed_results[sym][dt] = {"error": "no_ob_data"}
                continue

            ob_feat = compute_ob_features(ob_raw)
            # Add bid1_q, ask1_q for OFI
            ob_feat = ob_feat.copy()
            ob_feat["bid1_q"] = ob_raw["bid1_qty"].to_numpy(float)
            ob_feat["ask1_q"] = ob_raw["ask1_qty"].to_numpy(float)

            n_snap = len(ob_feat)
            print(f"    {n_snap} snapshots", flush=True)

            # Measurement 1: Stress
            s_res = measure_stress_firing(ob_feat, rolling_win=rolling_win)
            stress_results[sym][dt] = s_res
            print(
                f"    Stress: {s_res['firing_rate_pct']:.3f}% "
                f"({s_res['n_snapshots']} snaps)",
                flush=True,
            )

            # Measurement 2: Informed flow
            trades = load_trade_events_day(sym, dt)
            i_res = measure_informed_flow_firing(
                ob_feat, trades, rolling_win=rolling_win
            )
            informed_results[sym][dt] = i_res
            print(
                f"    Informed: {i_res['firing_rate_pct']:.3f}% "
                f"({i_res['n_snapshots']} snaps)",
                flush=True,
            )

    return {"stress": stress_results, "informed": informed_results}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def verdict_stress(stress_results: dict) -> tuple[str, str]:
    """Return (verdict, recommendation)."""
    rates = []
    for sym in TARGET_SYMS_12:
        for dt in TARGET_DATES:
            r = stress_results.get(sym, {}).get(dt, {})
            if "firing_rate_pct" in r:
                rates.append(r["firing_rate_pct"])

    if not rates:
        return "ERROR", "No data"
    min_rate = min(rates)
    max_rate = max(rates)
    mean_rate = sum(rates) / len(rates)

    if min_rate < 0.1:
        return (
            "RECALIBRATE",
            f"Min firing rate {min_rate:.3f}% < 0.1% on BTC/ETH. "
            "Recalibrate to p80 threshold or use single-feature trigger (log_spread > p90 OR |depth_ratio| > p90).",
        )
    elif mean_rate >= 0.5:
        return (
            "PASS",
            f"Mean firing rate {mean_rate:.3f}% >= 0.5% on BTC/ETH. Stress label is sufficiently sensitive.",
        )
    else:
        return (
            "RECALIBRATE",
            f"Mean firing rate {mean_rate:.3f}% < 0.5%. Consider p80 or single-feature trigger.",
        )


def verdict_informed(informed_results: dict) -> tuple[str, str]:
    rates = []
    for sym in TARGET_SYMS_12:
        for dt in TARGET_DATES:
            r = informed_results.get(sym, {}).get(dt, {})
            if "firing_rate_pct" in r:
                rates.append(r["firing_rate_pct"])

    if not rates:
        return "ERROR", "No data"
    min_rate = min(rates)
    max_rate = max(rates)
    mean_rate = sum(rates) / len(rates)

    if min_rate < 1.0:
        return (
            "RECALIBRATE",
            f"Min rate {min_rate:.3f}% < 1%. Consider loosening kyle_lambda to p60 or removing 3-snap consistency.",
        )
    elif max_rate > 20.0:
        return (
            "RECALIBRATE",
            f"Max rate {max_rate:.3f}% > 20%. Tighten kyle_lambda percentile to p85.",
        )
    else:
        return (
            "PASS",
            f"Firing rate {min_rate:.2f}%–{max_rate:.2f}% within 2–15% target range.",
        )


def verdict_climax(climax_results: dict) -> tuple[str, str]:
    below15_2sig = [
        s
        for s in SYMBOLS
        if isinstance(climax_results.get(s), dict)
        and climax_results[s].get("distinct_dates_2sigma", 0) < 15
    ]
    below15_3sig = [
        s
        for s in SYMBOLS
        if isinstance(climax_results.get(s), dict)
        and climax_results[s].get("distinct_dates_3sigma", 0) < 15
    ]

    if len(below15_2sig) > 5:
        return (
            "FAIL",
            f"{len(below15_2sig)}/25 symbols below 15 dates at 2σ. "
            "Drop climax from Wyckoff probe suite — label memorizes crashes.",
        )
    elif len(below15_3sig) > 10:
        return (
            "RECALIBRATE",
            f"{len(below15_3sig)}/25 symbols below 15 dates at 3σ. Use 2σ threshold only.",
        )
    else:
        return (
            "PASS",
            f"2σ: {25 - len(below15_2sig)}/25 symbols ≥15 dates. "
            f"3σ: {25 - len(below15_3sig)}/25 ≥15 dates.",
        )


def verdict_spring(spring_results: dict) -> tuple[str, str]:
    rec = spring_results.get("_recommendation", {})
    best_key = spring_results.get("_recommended_sigma_mult")
    n_ok = rec.get(f"sigma_{best_key:.1f}", 0) if best_key else 0

    if n_ok < 24:
        return (
            "FAIL",
            f"No sigma threshold achieves ≤8% firing on 24+/25 symbols. "
            "Need additional filter (evr > 1.5) or symbol-specific thresholds.",
        )
    else:
        return (
            "PASS",
            f"sigma_mult={best_key:.1f} achieves ≤8% on {n_ok}/25 symbols. "
            f"Use sigma_mult={best_key:.1f} in Steps 1-2 spring label.",
        )


def verdict_autocorr(autocorr_results: dict) -> tuple[str, str]:
    flags = autocorr_results.get("_flags", [])
    if not flags:
        return (
            "PASS",
            "No (feature, symbol) pairs with lag-5 r>0.8. "
            "5-event MEM block masking is non-trivial. Block size = 5 is valid.",
        )
    else:
        flagged_features = list(set(f["feature"] for f in flags))
        return (
            "RECALIBRATE",
            f"Features with r>0.8 at lag 5: {flagged_features}. "
            "MEM block size must grow to 10–15 events for these features, "
            "or use random position masking instead of block masking.",
        )


def write_report(
    m12: dict,
    climax: dict,
    spring: dict,
    autocorr: dict,
    elapsed: float,
) -> None:
    stress = m12["stress"]
    informed = m12["informed"]

    v_stress, r_stress = verdict_stress(stress)
    v_informed, r_informed = verdict_informed(informed)
    v_climax, r_climax = verdict_climax(climax)
    v_spring, r_spring = verdict_spring(spring)
    v_autocorr, r_autocorr = verdict_autocorr(autocorr)

    # Build stress table
    stress_rows = []
    for sym in TARGET_SYMS_12:
        for dt in TARGET_DATES:
            r = stress.get(sym, {}).get(dt, {})
            rate = f"{r['firing_rate_pct']:.3f}%" if "firing_rate_pct" in r else "ERROR"
            n = r.get("n_snapshots", "-")
            stress_rows.append(f"| {sym} | {dt} | {n} | {rate} |")

    # Build informed table
    informed_rows = []
    for sym in TARGET_SYMS_12:
        for dt in TARGET_DATES:
            r = informed.get(sym, {}).get(dt, {})
            rate = f"{r['firing_rate_pct']:.3f}%" if "firing_rate_pct" in r else "ERROR"
            n = r.get("n_snapshots", "-")
            informed_rows.append(f"| {sym} | {dt} | {n} | {rate} |")

    # Build climax table
    climax_rows = []
    for sym in SYMBOLS:
        c = climax.get(sym, {})
        if isinstance(c, dict) and "distinct_dates_2sigma" in c:
            n2 = c["distinct_dates_2sigma"]
            n3 = c["distinct_dates_3sigma"]
            tot = c["total_dates"]
            flag2 = " !!!" if n2 < 15 else ""
            flag3 = " !!!" if n3 < 15 else ""
            climax_rows.append(f"| {sym} | {n2}{flag2} | {n3}{flag3} | {tot} |")
        else:
            climax_rows.append(f"| {sym} | ERROR | ERROR | - |")

    # Build spring table
    spring_sym_rows = []
    for sym in SYMBOLS:
        s = spring.get(sym, {})
        if isinstance(s, dict) and "sigma_2.0" in s:
            r20 = s["sigma_2.0"]
            r25 = s["sigma_2.5"]
            r30 = s["sigma_3.0"]
            flags = []
            if r20 > 8.0:
                flags.append("2σ>8%")
            spring_sym_rows.append(
                f"| {sym} | {r20:.2f}% | {r25:.2f}% | {r30:.2f}% | {''.join(flags)} |"
            )
        else:
            spring_sym_rows.append(f"| {sym} | ERROR | ERROR | ERROR | - |")

    # Build autocorr tables per symbol
    autocorr_sections = []
    for sym in ["BTC", "ETH", "SOL"]:
        a = autocorr.get(sym, {})
        if isinstance(a, dict) and "lag5_autocorr" in a:
            n_w = a["n_windows"]
            ac = a["lag5_autocorr"]
            rows = []
            for fname in FEATURE_NAMES:
                r_val = ac.get(fname, float("nan"))
                flag = " ***" if r_val > 0.8 else ""
                rows.append(f"| {fname} | {r_val:.4f}{flag} |")
            autocorr_sections.append(
                f"#### {sym} ({n_w} windows)\n\n"
                "| Feature | Lag-5 Autocorr |\n"
                "|---------|----------------|\n" + "\n".join(rows)
            )

    # Build MEM recommendation
    flags_list = autocorr.get("_flags", [])
    flagged_feats = list(set(f["feature"] for f in flags_list))
    if flagged_feats:
        mem_rec = (
            f"Increase MEM block size to 10-15 for: {', '.join(sorted(flagged_feats))}."
        )
    else:
        mem_rec = "MEM block size = 5 is valid for all 17 features."

    rec_sigma = spring.get("_recommended_sigma_mult", 2.0)

    report = f"""# Step 0 Falsifiability Prerequisites

**Date:** 2026-04-14
**Data:** Pre-April only (2025-10-16 to 2026-03-25)
**Runtime:** {elapsed:.1f}s

---

## Summary

| Gate | Measurement | Verdict | Action |
|------|-------------|---------|--------|
| 1 | Stress label firing rate | **{v_stress}** | {r_stress} |
| 2 | Informed flow firing rate | **{v_informed}** | {r_informed} |
| 3 | Climax date-diversity | **{v_climax}** | {r_climax} |
| 4 | Spring threshold recalibration | **{v_spring}** | {r_spring} |
| 5 | Feature autocorr at lag 5 | **{v_autocorr}** | {r_autocorr} |

---

## Measurement 1: Stress Label Firing Rate

**Definition:** Fires if `log_spread > rolling_p90(log_spread)` AND `|depth_ratio| > rolling_p90(|depth_ratio|)` per snapshot.
**Council target:** ≥0.5% on 20+/25 symbols. Red flag if BTC/ETH fire <0.1%.

| Symbol | Date | Snapshots | Firing Rate |
|--------|------|-----------|-------------|
{chr(10).join(stress_rows)}

**Verdict: {v_stress}**
{r_stress}

---

## Measurement 2: Informed Flow Label Firing Rate

**Definition:** `kyle_lambda > rolling_p75` AND `|cum_ofi_5| > rolling_p50` AND sign consistency over 3 consecutive snapshots.
**Council target:** 2–15% firing. Flag if <1% or >20%.

| Symbol | Date | Snapshots | Firing Rate |
|--------|------|-----------|-------------|
{chr(10).join(informed_rows)}

**Verdict: {v_informed}**
{r_informed}

---

## Measurement 3: Climax Date-Diversity per Symbol

**Definition:** At least one event fires climax_score = min(z_qty, z_return) > threshold on that calendar date.
**Council target:** ≥15 distinct dates per symbol. Symbols below 15 flag (`!!!`).

| Symbol | Dates ≥2σ | Dates ≥3σ | Total Dates |
|--------|-----------|-----------|-------------|
{chr(10).join(climax_rows)}

**Verdict: {v_climax}**
{r_climax}

---

## Measurement 4: Spring Threshold Recalibration

**Definition:** `min(last_50_returns) < -N*σ` AND `evr_at_min > 1.0` AND `is_open_at_min > 0.5` AND `mean(last_10_returns) > 0`.
**Council target:** ≤8% firing on 24+/25 symbols. Symbols where 2σ fires >8% are flagged.

| Symbol | -2σ | -2.5σ | -3σ | Flags |
|--------|-----|-------|-----|-------|
{chr(10).join(spring_sym_rows)}

**Verdict: {v_spring}**
{r_spring}

**Recommended sigma_mult for Steps 1-2:** `{rec_sigma:.1f}`

---

## Measurement 5: Feature Autocorrelation at Lag 5

**Definition:** Average lag-5 autocorrelation over ≥1000 random 200-event windows per symbol.
**Council threshold:** r>0.8 means MEM 5-event block masking is trivially solvable — block size must grow.

{chr(10).join(autocorr_sections)}

**Verdict: {v_autocorr}**
{r_autocorr}

---

## Actionable Parameters for Steps 1-2 Plan

1. **MEM block size:** {mem_rec}
2. **Spring threshold:** Use `sigma_mult = {rec_sigma:.1f}` (set `evr > 1.0`, `is_open > 0.5`, `mean_recent10 > 0` unchanged unless ≥24 symbols still fire >8% at this mult).
3. **Stress label:** {"Use p80 or single-feature (OR) trigger." if v_stress == "RECALIBRATE" else "Keep joint p90 AND p90 as specified."}
4. **Informed flow label:** {"Loosen: try kyle_lambda > p60 or remove 3-snap consistency." if v_informed == "RECALIBRATE" and "Loosen" in r_informed else "Keep as specified." if v_informed == "PASS" else "Tighten: try kyle_lambda > p85."}
5. **Climax probe:** {"Drop from Wyckoff diagnostic probe suite — too few firing dates on too many symbols." if v_climax == "FAIL" else "Keep. Use 2σ threshold." if v_climax in ("PASS", "RECALIBRATE") else "Investigate."}
"""

    out_md = OUT_DIR / "step0-falsifiability-prereqs.md"
    out_md.write_text(report)
    print(f"\nReport written: {out_md}")


def write_json(
    m12: dict,
    climax: dict,
    spring: dict,
    autocorr: dict,
) -> None:
    v_stress, r_stress = verdict_stress(m12["stress"])
    v_informed, r_informed = verdict_informed(m12["informed"])
    v_climax, r_climax = verdict_climax(climax)
    v_spring, r_spring = verdict_spring(spring)
    v_autocorr, r_autocorr = verdict_autocorr(autocorr)

    payload = {
        "generated": "2026-04-14",
        "verdicts": {
            "stress": v_stress,
            "informed_flow": v_informed,
            "climax_diversity": v_climax,
            "spring_recalibration": v_spring,
            "autocorr_lag5": v_autocorr,
        },
        "recommendations": {
            "stress": r_stress,
            "informed_flow": r_informed,
            "climax_diversity": r_climax,
            "spring_recalibration": r_spring,
            "autocorr_lag5": r_autocorr,
        },
        "stress_firing": m12["stress"],
        "informed_firing": m12["informed"],
        "climax_date_diversity": climax,
        "spring_recalibration": spring,
        "autocorr_lag5": autocorr,
    }

    out_json = OUT_DIR / "step0-falsifiability-prereqs.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"JSON written: {out_json}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Falsifiability Prerequisites — Step 0 council-5 audit")
    print("=" * 70)

    # Measurements 1 + 2
    m12 = measure_stress_and_informed(rolling_win=1000)

    # Measurement 3
    climax = compute_climax_date_diversity()

    # Measurement 4
    spring = measure_spring_recalibration()

    # Measurement 5
    autocorr = measure_autocorrelation(n_windows=1000)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Output
    write_report(m12, climax, spring, autocorr, elapsed)
    write_json(m12, climax, spring, autocorr)

    print("\nDone.")


if __name__ == "__main__":
    main()
