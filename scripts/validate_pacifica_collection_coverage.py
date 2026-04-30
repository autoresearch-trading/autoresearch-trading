# scripts/validate_pacifica_collection_coverage.py
"""Validate local Pacifica raw-data collection coverage against public API docs.

This is a data audit, not an alpha experiment.  It separates:

- local raw trade/orderbook partition coverage;
- schema fields stored locally;
- currently listed Pacifica symbols from REST `/info` and `/info/prices`;
- public websocket market-data streams documented by Pacifica.

The audit intentionally avoids account/private streams.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import duckdb
import pandas as pd

REST_BASE = "https://api.pacifica.fi/api/v1"

# Public websocket subscription docs checked on 2026-04-30.
PUBLIC_WS_STREAMS: dict[str, dict[str, Any]] = {
    "prices": {
        "captured": False,
        "fields": [
            "funding",
            "mark",
            "mid",
            "next_funding",
            "open_interest",
            "oracle",
            "symbol",
            "timestamp",
            "volume_24h",
            "yesterday_price",
        ],
        "note": "No local data/prices partitions found.",
    },
    "book": {
        "captured": True,
        "fields": ["l", "a", "n", "p", "s", "t", "li"],
        "note": "Local data/orderbook stores ts_ms, symbol, bids/asks price+qty, recv_ms, agg_level; it does not preserve documented order-count n or exchange nonce li.",
    },
    "bbo": {
        "captured": False,
        "fields": ["s", "i", "li", "t", "b", "B", "a", "A"],
        "note": "Best bid/ask can be derived from local orderbook snapshots, but the raw BBO stream fields order id i and last order id li are not stored.",
    },
    "trades": {
        "captured": True,
        "fields": ["h", "s", "a", "p", "d", "tc", "t", "li"],
        "note": "Local data/trades stores ts_ms, symbol, trade_id, side, qty, price, recv_ms, cause, event_type, date; April+ cause/event_type are present, but documented history id h and exchange nonce li are not preserved.",
    },
    "candle": {
        "captured": False,
        "fields": ["t", "T", "s", "i", "o", "c", "h", "l", "v", "n"],
        "note": "Trade candles are derivable from trades but raw candle stream is not stored.",
    },
    "mark_price_candle": {
        "captured": False,
        "fields": ["t", "T", "s", "i", "o", "c", "h", "l", "v", "n"],
        "note": "Not derivable from local trades/orderbook without mark-price data; no local mark-price candle partitions found.",
    },
}


@dataclass(frozen=True)
class PartitionStats:
    dataset: str
    symbols: int
    partitions: int
    min_date: str
    max_date: str
    counts_by_first_date: dict[str, int]
    symbols_ge_100_days: int
    symbols_le_30_days: int
    by_symbol: pd.DataFrame
    by_date: pd.DataFrame


def _partition_pairs(root: Path) -> dict[str, list[str]]:
    by_symbol: dict[str, list[str]] = defaultdict(list)
    for sdir in root.glob("symbol=*"):
        symbol = sdir.name.split("=", 1)[1]
        for ddir in sdir.glob("date=*"):
            if any(ddir.glob("*.parquet")):
                by_symbol[symbol].append(ddir.name.split("=", 1)[1])
    return by_symbol


def partition_stats(repo: Path, dataset: str) -> PartitionStats:
    by_symbol_raw = _partition_pairs(repo / "data" / dataset)
    rows = []
    for symbol, dates in sorted(by_symbol_raw.items()):
        unique_dates = sorted(set(dates))
        rows.append(
            {
                "symbol": symbol,
                "first_date": unique_dates[0],
                "last_date": unique_dates[-1],
                "n_dates": len(unique_dates),
            }
        )
    by_symbol = pd.DataFrame(rows)
    if by_symbol.empty:
        return PartitionStats(
            dataset, 0, 0, "", "", {}, 0, 0, by_symbol, pd.DataFrame()
        )

    by_date_counts: Counter[str] = Counter()
    for dates in by_symbol_raw.values():
        by_date_counts.update(set(dates))
    by_date = pd.DataFrame(
        {"date": d, "n_symbols": n} for d, n in sorted(by_date_counts.items())
    )
    first_counts = Counter(by_symbol["first_date"].astype(str).tolist())
    return PartitionStats(
        dataset=dataset,
        symbols=int(by_symbol["symbol"].nunique()),
        partitions=int(by_symbol["n_dates"].sum()),
        min_date=str(by_symbol["first_date"].min()),
        max_date=str(by_symbol["last_date"].max()),
        counts_by_first_date=dict(sorted(first_counts.items())),
        symbols_ge_100_days=int((by_symbol["n_dates"] >= 100).sum()),
        symbols_le_30_days=int((by_symbol["n_dates"] <= 30).sum()),
        by_symbol=by_symbol,
        by_date=by_date,
    )


def pair_mismatches(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def pairs(dataset: str) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        root = repo / "data" / dataset
        for ddir in root.glob("symbol=*/date=*"):
            if any(ddir.glob("*.parquet")):
                symbol = ddir.parent.name.split("=", 1)[1]
                date = ddir.name.split("=", 1)[1]
                out.add((symbol, date))
        return out

    tr = pairs("trades")
    ob = pairs("orderbook")
    tr_not_ob = pd.DataFrame(sorted(tr - ob), columns=["symbol", "date"])
    ob_not_tr = pd.DataFrame(sorted(ob - tr), columns=["symbol", "date"])
    return tr_not_ob, ob_not_tr


def sample_schema(repo: Path, dataset: str) -> pd.DataFrame:
    files: list[Path] = []
    for pattern in [
        f"data/{dataset}/symbol=BTC/date=2026-04-13/*.parquet",
        f"data/{dataset}/symbol=ADA/date=2026-04-13/*.parquet",
        f"data/{dataset}/symbol=BTC/date=2025-11-01/*.parquet",
    ]:
        files.extend(sorted(repo.glob(pattern))[:1])
    rows = []
    for f in files:
        desc = duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{f}')").to_df()
        rows.append(
            {
                "file": str(f.relative_to(repo)),
                "columns": ", ".join(desc["column_name"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def april_cause_counts(repo: Path, dates: list[str]) -> pd.DataFrame:
    rows = []
    for date in dates:
        files = list((repo / "data" / "trades").glob(f"symbol=*/date={date}/*.parquet"))
        if not files:
            continue
        q = f"""
        SELECT cause, count(*) AS n, count(distinct symbol) AS symbols
        FROM read_parquet('{repo}/data/trades/symbol=*/date={date}/*.parquet', hive_partitioning=1, union_by_name=1)
        GROUP BY cause ORDER BY n DESC
        """
        try:
            df = duckdb.query(q).to_df()
        except Exception as exc:  # pragma: no cover - filesystem dependent
            rows.append({"date": date, "cause": f"ERROR: {exc}", "n": 0, "symbols": 0})
            continue
        for row in df.to_dict("records"):
            rows.append({"date": date, **row})
    return pd.DataFrame(rows)


def local_symbols(repo: Path) -> set[str]:
    out: set[str] = set()
    for dataset in ["trades", "orderbook"]:
        for sdir in (repo / "data" / dataset).glob("symbol=*"):
            out.add(sdir.name.split("=", 1)[1])
    return out


def fetch_live_json(path: str) -> Any:
    with urlopen(
        f"{REST_BASE}{path}", timeout=30
    ) as response:  # noqa: S310 - public API audit
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("success") is True:
        return payload.get("data")
    return payload


def live_symbol_tables(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = fetch_live_json("/info")
    prices = fetch_live_json("/info/prices")
    info_df = pd.DataFrame(info)
    prices_df = pd.DataFrame(prices)
    local = local_symbols(repo)
    live_info = (
        set(info_df["symbol"].astype(str).tolist()) if "symbol" in info_df else set()
    )
    live_prices = (
        set(prices_df["symbol"].astype(str).tolist())
        if "symbol" in prices_df
        else set()
    )
    union_live = live_info | live_prices
    comparison = pd.DataFrame(
        [
            {
                "symbol": s,
                "local_raw": s in local,
                "live_info": s in live_info,
                "live_prices": s in live_prices,
            }
            for s in sorted(local | union_live)
        ]
    )
    return info_df, prices_df, comparison


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    out = ["| " + " | ".join(map(str, df.columns)) + " |"]
    out.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_report(repo: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = partition_stats(repo, "trades")
    orderbook = partition_stats(repo, "orderbook")
    tr_not_ob, ob_not_tr = pair_mismatches(repo)
    trade_schema = sample_schema(repo, "trades")
    ob_schema = sample_schema(repo, "orderbook")
    cause_counts = april_cause_counts(
        repo, ["2026-04-06", "2026-04-13", "2026-04-14", "2026-04-26", "2026-04-27"]
    )
    info_df, prices_df, live_cmp = live_symbol_tables(repo)

    # Persist machine-readable artifacts.
    trades.by_symbol.to_csv(out_dir / "trades_by_symbol.csv", index=False)
    orderbook.by_symbol.to_csv(out_dir / "orderbook_by_symbol.csv", index=False)
    trades.by_date.to_csv(out_dir / "trades_by_date.csv", index=False)
    orderbook.by_date.to_csv(out_dir / "orderbook_by_date.csv", index=False)
    live_cmp.to_csv(out_dir / "live_symbol_comparison.csv", index=False)
    cause_counts.to_csv(out_dir / "april_cause_counts.csv", index=False)
    (out_dir / "public_ws_streams.json").write_text(
        json.dumps(PUBLIC_WS_STREAMS, indent=2)
    )

    local_symbol_count = int(live_cmp["local_raw"].sum()) if not live_cmp.empty else 0
    live_info_count = int(live_cmp["live_info"].sum()) if not live_cmp.empty else 0
    live_prices_count = int(live_cmp["live_prices"].sum()) if not live_cmp.empty else 0
    missing_live = live_cmp[
        (~live_cmp["local_raw"]) & (live_cmp["live_info"] | live_cmp["live_prices"])
    ]
    stale_local = live_cmp[
        live_cmp["local_raw"] & ~(live_cmp["live_info"] | live_cmp["live_prices"])
    ]

    stream_rows = pd.DataFrame(
        {
            "stream": name,
            "captured_locally": spec["captured"],
            "fields": ", ".join(spec["fields"]),
            "local_status": spec["note"],
        }
        for name, spec in PUBLIC_WS_STREAMS.items()
    )

    report = f"""# Pacifica Collection Coverage Audit

**Purpose:** validate whether this repo is collecting all market-data datapoints Pacifica publicly provides.  
**Scope:** public market-data REST/WebSocket docs and local raw `data/trades` / `data/orderbook` partitions. Private account streams are intentionally out of scope.

## Executive verdict

The repo is collecting the two core high-frequency streams needed for tape research — trades and orderbook — across the expanded raw universe with near-complete symbol/date coverage through full days ending 2026-04-26.

It is **not** collecting every public Pacifica datapoint/field. Missing or lossy relative to public docs:

- `prices` stream / `/info/prices`: funding, next funding, mark, oracle, open interest, 24h volume, yesterday price.
- `bbo` stream raw fields: best bid/ask can be derived from orderbook, but order id `i` and exchange nonce `li` are not stored.
- `trades` stream raw fields: local trades do not preserve documented history id `h` or exchange-wide nonce `li`; local `trade_id` appears blank in sampled files.
- `book` stream raw fields: local orderbook does not preserve documented level order-count `n` or exchange-wide nonce `li`; `agg_level` is present but sampled values are null.
- `candle` stream: derivable from trades, but not stored as raw candles.
- `mark_price_candle` stream: not stored and not derivable without mark-price data.
- REST funding history is available in API docs but no local funding-history partition was found.

So: **good raw tape capture, not complete Pacifica market-data capture.**

## Local partition coverage

| dataset | symbols | symbol/date partitions | min_date | max_date | >=100-day symbols | <=30-day symbols | first-date cohorts |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| trades | {trades.symbols} | {trades.partitions} | {trades.min_date} | {trades.max_date} | {trades.symbols_ge_100_days} | {trades.symbols_le_30_days} | {trades.counts_by_first_date} |
| orderbook | {orderbook.symbols} | {orderbook.partitions} | {orderbook.min_date} | {orderbook.max_date} | {orderbook.symbols_ge_100_days} | {orderbook.symbols_le_30_days} | {orderbook.counts_by_first_date} |

## Trade/orderbook pair mismatches

Trades without orderbook: {len(tr_not_ob)}

{_md_table(tr_not_ob.head(20))}

Orderbook without trades: {len(ob_not_tr)}

{_md_table(ob_not_tr.head(20))}

## Live Pacifica symbol comparison

Live `/info` symbols: {live_info_count}  
Live `/info/prices` symbols: {live_prices_count}  
Local raw symbols: {local_symbol_count}

Live symbols not present in local raw data: {len(missing_live)}

{_md_table(missing_live.head(80))}

Local raw symbols not present in current live REST data: {len(stale_local)}

{_md_table(stale_local.head(80))}

## Public websocket stream coverage

{_md_table(stream_rows)}

## Sample local schemas

### Trades

{_md_table(trade_schema)}

### Orderbook

{_md_table(ob_schema)}

## Liquidation cause availability spot-check

{_md_table(cause_counts)}

## Artifacts

- `trades_by_symbol.csv`
- `orderbook_by_symbol.csv`
- `trades_by_date.csv`
- `orderbook_by_date.csv`
- `live_symbol_comparison.csv`
- `april_cause_counts.csv`
- `public_ws_streams.json`

## Recommended collector fixes if the goal is complete Pacifica public market-data capture

1. Add a raw `prices` capture table/partition keyed by timestamp and symbol.
2. Preserve raw websocket fields exactly before transformation:
   - trades: `h`, `s`, `a`, `p`, `d`, `tc`, `t`, `li`, plus local `recv_ms`.
   - book: raw `l` levels including amount `a`, count `n`, price `p`, symbol `s`, timestamp `t`, exchange nonce `li`, aggregation level.
3. Decide whether to store BBO as a separate raw stream. It is partly derivable, but raw `i` and `li` are not.
4. Capture funding history / funding snapshots if any funding, carry, mark/oracle, or liquidation-economics work will continue.
5. Capture mark-price candles or mark-price snapshots if mark/oracle dislocation becomes a hypothesis.
6. Update the derived feature cache/constants separately; raw collection already has a wider universe than `tape.constants.SYMBOLS`.
"""
    report = textwrap.dedent(report)
    path = out_dir / "README.md"
    path.write_text(report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/pacifica-collection-coverage"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    path = write_report(repo, out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
