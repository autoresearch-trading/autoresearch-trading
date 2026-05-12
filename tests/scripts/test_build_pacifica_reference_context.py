from pathlib import Path

import pandas as pd

from scripts.build_pacifica_reference_context import (
    ReferenceContextConfig,
    build_reference_context,
    write_reference_context_report,
)


def _state_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_000,
                "last_mid": 50_000.0,
                "mid_return_bps": 10.0,
                "realized_vol_bps": 12.0,
                "funding": 0.00010,
            },
            {
                "symbol": "ETH",
                "bucket_start_ms": 1_000,
                "last_mid": 4_000.0,
                "mid_return_bps": -5.0,
                "realized_vol_bps": 20.0,
                "funding": -0.00020,
            },
            {
                "symbol": "SOL",
                "bucket_start_ms": 1_000,
                "last_mid": 100.0,
                "mid_return_bps": 30.0,
                "realized_vol_bps": 35.0,
                "funding": 0.00000,
            },
            {
                "symbol": "BTC",
                "bucket_start_ms": 2_000,
                "last_mid": 50_250.0,
                "mid_return_bps": 50.0,
                "realized_vol_bps": 50.0,
                "funding": 0.00015,
            },
        ]
    )


def _reference_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTC",
                "bucket_start_ms": 1_000,
                "reference_mid": 49_950.0,
                "reference_return_bps": 8.0,
                "reference_vol_bps": 15.0,
                "reference_funding": 0.00005,
            },
            {
                "symbol": "ETH",
                "bucket_start_ms": 1_000,
                "reference_mid": 4_020.0,
                "reference_return_bps": -7.0,
                "reference_vol_bps": 22.0,
                "reference_funding": -0.00010,
            },
            {
                "symbol": "SOL",
                "bucket_start_ms": 1_000,
                "reference_mid": 99.0,
                "reference_return_bps": 25.0,
                "reference_vol_bps": 30.0,
                "reference_funding": 0.00005,
            },
            {
                "symbol": "BTC",
                "bucket_start_ms": 2_000,
                "reference_mid": 50_000.0,
                "reference_return_bps": 40.0,
                "reference_vol_bps": 45.0,
                "reference_funding": 0.00005,
            },
        ]
    )


def test_build_reference_context_adds_beta_premium_risk_state_and_funding_divergence() -> (
    None
):
    context = build_reference_context(_state_rows(), _reference_rows())
    by_key = context.set_index(["symbol", "bucket_start_ms"])

    btc = by_key.loc[("BTC", 1_000)]
    assert round(float(btc["btc_eth_beta_proxy_bps"]), 4) == 0.5
    assert round(float(btc["cross_venue_premium_bps"]), 4) == 10.0100
    assert round(float(btc["funding_divergence_bps"]), 4) == 0.5
    assert btc["broad_crypto_risk_state"] == "RISK_NEUTRAL"
    assert (
        btc["reference_context_version"]
        == "pacifica_reference_context_v1_fixed_diagnostic"
    )

    sol = by_key.loc[("SOL", 1_000)]
    assert round(float(sol["btc_eth_beta_proxy_bps"]), 4) == 0.5
    assert sol["broad_crypto_risk_state"] == "RISK_ON"

    btc_later = by_key.loc[("BTC", 2_000)]
    assert btc_later["broad_crypto_risk_state"] == "HIGH_VOL_RISK_ON"


def test_reference_context_fails_closed_on_missing_required_columns() -> None:
    missing_state = _state_rows().drop(columns=["last_mid"])
    try:
        build_reference_context(missing_state, _reference_rows())
    except ValueError as exc:
        assert "state missing required columns" in str(exc)
        assert "last_mid" in str(exc)
    else:
        raise AssertionError("missing state column did not fail closed")

    missing_reference = _reference_rows().drop(columns=["reference_mid"])
    try:
        build_reference_context(_state_rows(), missing_reference)
    except ValueError as exc:
        assert "reference missing required columns" in str(exc)
        assert "reference_mid" in str(exc)
    else:
        raise AssertionError("missing reference column did not fail closed")


def test_reference_context_rejects_dirty_keys_duplicates_and_invalid_prices() -> None:
    dirty_symbol = _state_rows().copy()
    dirty_symbol.loc[0, "symbol"] = " BTC"
    try:
        build_reference_context(dirty_symbol, _reference_rows())
    except ValueError as exc:
        assert "noncanonical symbol" in str(exc)
    else:
        raise AssertionError("dirty symbol did not fail closed")

    duplicated = pd.concat(
        [_reference_rows(), _reference_rows().head(1)], ignore_index=True
    )
    try:
        build_reference_context(_state_rows(), duplicated)
    except ValueError as exc:
        assert "duplicate symbol/bucket_start_ms" in str(exc)
    else:
        raise AssertionError("duplicate reference key did not fail closed")

    bad_price = _reference_rows().copy()
    bad_price.loc[0, "reference_mid"] = 0.0
    try:
        build_reference_context(_state_rows(), bad_price)
    except ValueError as exc:
        assert "positive finite prices" in str(exc)
    else:
        raise AssertionError("invalid reference price did not fail closed")


def test_dirty_bucket_keys_and_negative_volatility_fail_closed() -> None:
    fractional_bucket = _state_rows().head(1).copy()
    fractional_bucket["bucket_start_ms"] = fractional_bucket["bucket_start_ms"].astype(
        object
    )
    fractional_bucket.loc[0, "bucket_start_ms"] = 1_000.5
    try:
        build_reference_context(fractional_bucket, _reference_rows().head(1))
    except ValueError as exc:
        assert "invalid bucket_start_ms" in str(exc)
    else:
        raise AssertionError("fractional bucket key did not fail closed")

    duplicate_after_coercion = pd.concat(
        [_reference_rows().head(1), _reference_rows().head(1)], ignore_index=True
    )
    duplicate_after_coercion["bucket_start_ms"] = duplicate_after_coercion[
        "bucket_start_ms"
    ].astype(object)
    duplicate_after_coercion.loc[1, "bucket_start_ms"] = "1000"
    try:
        build_reference_context(_state_rows().head(1), duplicate_after_coercion)
    except ValueError as exc:
        assert "duplicate symbol/bucket_start_ms" in str(exc)
    else:
        raise AssertionError("canonical duplicate key did not fail closed")

    negative_vol = _reference_rows().head(1).copy()
    negative_vol.loc[0, "reference_vol_bps"] = -1.0
    try:
        build_reference_context(_state_rows().head(1), negative_vol)
    except ValueError as exc:
        assert "nonnegative volatility" in str(exc)
    else:
        raise AssertionError("negative volatility did not fail closed")


def test_missing_reference_rows_are_explicitly_flagged_not_imputed() -> None:
    reference = _reference_rows()
    reference = reference[
        ~((reference["symbol"] == "ETH") & (reference["bucket_start_ms"] == 1_000))
    ]

    context = build_reference_context(_state_rows(), reference)
    eth = context.set_index(["symbol", "bucket_start_ms"]).loc[("ETH", 1_000)]

    assert eth["reference_coverage"] == "MISSING_REFERENCE"
    assert eth["broad_crypto_risk_state"] == "REFERENCE_MISSING"
    assert eth["btc_eth_beta_proxy_coverage"] == "PARTIAL_MAJOR_REFERENCE"
    assert pd.isna(eth["cross_venue_premium_bps"])


def test_missing_major_reference_bucket_is_explicitly_marked() -> None:
    state = pd.DataFrame(
        [
            {
                "symbol": "SOL",
                "bucket_start_ms": 3_000,
                "last_mid": 101.0,
                "mid_return_bps": 12.0,
                "realized_vol_bps": 15.0,
                "funding": 0.00001,
            }
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "symbol": "SOL",
                "bucket_start_ms": 3_000,
                "reference_mid": 100.0,
                "reference_return_bps": 11.0,
                "reference_vol_bps": 14.0,
                "reference_funding": 0.00002,
            }
        ]
    )

    context = build_reference_context(state, reference)
    sol = context.iloc[0]

    assert sol["reference_coverage"] == "REFERENCE_AVAILABLE"
    assert sol["btc_eth_beta_proxy_coverage"] == "MISSING_MAJOR_REFERENCE"
    assert pd.isna(sol["btc_eth_beta_proxy_bps"])


def test_write_reference_context_report_emits_markdown_and_csvs(tmp_path: Path) -> None:
    context = build_reference_context(_state_rows(), _reference_rows())
    report = write_reference_context_report(
        context, tmp_path, config=ReferenceContextConfig()
    )

    assert report["verdict"] == "DIAGNOSTIC_REFERENCE_CONTEXT_BUILT"
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "reference_context.csv").exists()
    assert (tmp_path / "risk_state_summary.csv").exists()
    assert (tmp_path / "symbol_reference_summary.csv").exists()
    assert (tmp_path / "config.csv").exists()
    text = (tmp_path / "README.md").read_text()
    assert "Cross-venue/reference market context" in text
    assert "not a trade signal" in text
