from pathlib import Path

import pandas as pd
import pytest

from scripts.build_non_hft_regime_state import (
    build_regime_state,
    compute_toxicity_score,
    read_silver_table,
    write_regime_state,
)


def test_build_regime_state_aggregates_slow_one_minute_features(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    silver.mkdir()
    pd.DataFrame(
        [
            {
                "event_ts_ms": 0,
                "symbol": "BTC",
                "mid": 100.0,
                "spread_bps": 10.0,
                "top_bid_notional": 1000.0,
                "top_ask_notional": 900.0,
                "last_order_id": 1,
            },
            {
                "event_ts_ms": 30_000,
                "symbol": "BTC",
                "mid": 101.0,
                "spread_bps": 12.0,
                "top_bid_notional": 800.0,
                "top_ask_notional": 700.0,
                "last_order_id": 4,
            },
            {
                "event_ts_ms": 61_000,
                "symbol": "BTC",
                "mid": 102.0,
                "spread_bps": 20.0,
                "top_bid_notional": 600.0,
                "top_ask_notional": 500.0,
                "last_order_id": 5,
            },
        ]
    ).to_parquet(silver / "bbo.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_ts_ms": 10_000,
                "symbol": "BTC",
                "signed_qty": 1.0,
                "qty": 1.0,
                "notional": 100.0,
                "trade_class": "normal",
            },
            {
                "event_ts_ms": 20_000,
                "symbol": "BTC",
                "signed_qty": -3.0,
                "qty": 3.0,
                "notional": 300.0,
                "trade_class": "liquidation",
            },
            {
                "event_ts_ms": 65_000,
                "symbol": "BTC",
                "signed_qty": 2.0,
                "qty": 2.0,
                "notional": 204.0,
                "trade_class": "normal",
            },
        ]
    ).to_parquet(silver / "trades.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_ts_ms": 0,
                "symbol": "BTC",
                "mark_oracle_basis_bps": 5.0,
                "mid_oracle_basis_bps": 3.0,
                "funding": 0.001,
                "next_funding": 0.002,
                "open_interest": 10.0,
            },
            {
                "event_ts_ms": 61_000,
                "symbol": "BTC",
                "mark_oracle_basis_bps": -5.0,
                "mid_oracle_basis_bps": -3.0,
                "funding": 0.003,
                "next_funding": 0.004,
                "open_interest": 12.0,
            },
        ]
    ).to_parquet(silver / "prices.parquet", index=False)

    state = build_regime_state(silver, bucket="1min")

    assert list(state["bucket_start_ms"]) == [0, 60_000]
    first = state.iloc[0]
    assert first["symbol"] == "BTC"
    assert first["bbo_updates"] == 2
    assert first["avg_spread_bps"] == 11.0
    assert first["trade_count"] == 2
    assert first["trade_notional"] == 400.0
    assert first["signed_trade_qty"] == -2.0
    assert first["liquidation_count"] == 1
    assert first["liquidation_notional"] == 300.0
    assert first["mark_oracle_basis_bps"] == 5.0
    assert first["open_interest"] == 10.0
    assert first["mid_return_bps"] == pytest.approx(100.0)


def test_build_regime_state_counts_pacifica_cause_liquidations(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    silver.mkdir()
    pd.DataFrame(
        [
            {
                "event_ts_ms": 1_000,
                "symbol": "BTC",
                "signed_qty": -1.0,
                "qty": 1.0,
                "notional": 100.0,
                "trade_class": "normal",
                "cause": "market_liquidation",
            },
            {
                "event_ts_ms": 2_000,
                "symbol": "BTC",
                "signed_qty": 2.0,
                "qty": 2.0,
                "notional": 250.0,
                "trade_class": "normal",
                "cause": "backstop_liquidation",
            },
            {
                "event_ts_ms": 3_000,
                "symbol": "BTC",
                "signed_qty": 3.0,
                "qty": 3.0,
                "notional": 300.0,
                "trade_class": "normal",
                "cause": "normal",
            },
        ]
    ).to_parquet(silver / "trades.parquet", index=False)

    state = build_regime_state(silver, bucket="1min")

    assert len(state) == 1
    row = state.iloc[0]
    assert row["liquidation_count"] == 2
    assert row["liquidation_notional"] == 350.0


def test_build_regime_state_counts_pacifica_trade_class_liquidation_variants(
    tmp_path: Path,
) -> None:
    silver = tmp_path / "silver"
    silver.mkdir()
    pd.DataFrame(
        [
            {
                "event_ts_ms": 1_000,
                "symbol": "BTC",
                "signed_qty": -1.0,
                "qty": 1.0,
                "notional": 125.0,
                "trade_class": "insolvency_liquidation",
            },
            {
                "event_ts_ms": 2_000,
                "symbol": "BTC",
                "signed_qty": 1.0,
                "qty": 1.0,
                "notional": 75.0,
                "trade_class": "normal",
            },
        ]
    ).to_parquet(silver / "trades.parquet", index=False)

    state = build_regime_state(silver, bucket="1min")

    row = state.iloc[0]
    assert row["liquidation_count"] == 1
    assert row["liquidation_notional"] == 125.0


def test_read_silver_table_reads_partitioned_channel_layout(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    part = (
        silver
        / "channel=bbo"
        / "symbol=BTC"
        / "date=1970-01-01"
        / "part-000000.parquet"
    )
    part.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_ts_ms": 0,
                "symbol": "BTC",
                "mid": 100.0,
                "spread_bps": 10.0,
                "top_bid_notional": 1.0,
                "top_ask_notional": 1.0,
                "last_order_id": 1,
            }
        ]
    ).to_parquet(part, index=False)

    table = read_silver_table(silver, "bbo")

    assert len(table) == 1
    assert table.loc[0, "symbol"] == "BTC"


def test_compute_toxicity_score_is_slow_regime_ranking_not_trade_trigger() -> None:
    df = pd.DataFrame(
        [
            {
                "avg_spread_bps": 5.0,
                "bbo_churn_per_min": 1.0,
                "abs_trade_imbalance": 0.1,
                "realized_vol_bps": 2.0,
                "liquidation_notional": 0.0,
                "mark_oracle_basis_abs_bps": 1.0,
            },
            {
                "avg_spread_bps": 50.0,
                "bbo_churn_per_min": 10.0,
                "abs_trade_imbalance": 0.9,
                "realized_vol_bps": 20.0,
                "liquidation_notional": 1000.0,
                "mark_oracle_basis_abs_bps": 10.0,
            },
        ]
    )

    scored = compute_toxicity_score(df)

    assert list(scored["toxicity_score"]).__len__() == 2
    assert scored.loc[1, "toxicity_score"] > scored.loc[0, "toxicity_score"]
    assert scored["toxicity_score"].between(0, 1).all()


def test_write_regime_state_creates_parquet_csv_and_report(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    silver.mkdir()
    pd.DataFrame(
        [
            {
                "event_ts_ms": 0,
                "symbol": "ETH",
                "mid": 100.0,
                "spread_bps": 10.0,
                "top_bid_notional": 1.0,
                "top_ask_notional": 1.0,
                "last_order_id": 1,
            }
        ]
    ).to_parquet(silver / "bbo.parquet", index=False)

    out = tmp_path / "out"
    state = write_regime_state(silver, out, bucket="1min")

    assert (out / "regime_state.parquet").exists()
    assert (out / "regime_state_preview.csv").exists()
    assert (out / "README.md").exists()
    assert "non-HFT" in (out / "README.md").read_text()
    assert len(state) == 1
