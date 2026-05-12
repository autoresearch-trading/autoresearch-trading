# scripts/build_pacifica_reference_context.py
"""Build diagnostic cross-venue/reference market context for Pacifica rows.

This module distinguishes Pacifica-local states from broad market/reference
states. It intentionally starts from local CSV/parquet inputs so the research
stack is not tied to a paid API or a single vendor schema.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STATE_PATH = Path("docs/experiments/non-hft-regime-state/regime_state.parquet")
DEFAULT_REFERENCE_PATH = Path("data/reference_market_context/reference_context.csv")
DEFAULT_OUT_DIR = Path("docs/experiments/reference-market-context")
REFERENCE_CONTEXT_VERSION = "pacifica_reference_context_v1_fixed_diagnostic"
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
REQUIRED_STATE_COLUMNS = (
    "symbol",
    "bucket_start_ms",
    "last_mid",
    "mid_return_bps",
    "realized_vol_bps",
    "funding",
)
REQUIRED_REFERENCE_COLUMNS = (
    "symbol",
    "bucket_start_ms",
    "reference_mid",
    "reference_return_bps",
    "reference_vol_bps",
    "reference_funding",
)
OUTPUT_COLUMNS_FIRST = [
    "symbol",
    "bucket_start_ms",
    "reference_coverage",
    "broad_crypto_risk_state",
    "reference_context_version",
    "last_mid",
    "reference_mid",
    "cross_venue_premium_bps",
    "funding_divergence_bps",
    "reference_return_bps",
    "reference_vol_bps",
    "btc_reference_return_bps",
    "eth_reference_return_bps",
    "btc_eth_beta_proxy_bps",
    "btc_eth_beta_proxy_coverage",
]


@dataclass(frozen=True)
class ReferenceContextConfig:
    risk_on_return_bps: float = 10.0
    risk_off_return_bps: float = -10.0
    high_vol_bps: float = 40.0
    positive_premium_bps: float = 5.0
    negative_premium_bps: float = -5.0


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown_table(
    df: pd.DataFrame, *, max_rows: int | None = None
) -> str:
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows) if max_rows is not None else df
    headers = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in table.columns) + " |")
    return "\n".join(lines)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def _validate_required(
    frame: pd.DataFrame, required: tuple[str, ...], *, source: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _validate_symbols(frame: pd.DataFrame, *, source: str) -> pd.Series:
    if "symbol" not in frame.columns:
        raise ValueError(f"{source} missing symbol column")
    raw = frame["symbol"]
    if raw.isna().any():
        raise ValueError(f"{source} contains null or blank symbols")
    as_text = raw.astype(str)
    if (as_text != as_text.str.strip()).any():
        raise ValueError(f"{source} contains noncanonical symbol whitespace")
    if (as_text.str.strip() == "").any():
        raise ValueError(f"{source} contains null or blank symbols")
    reserved = as_text.str.lower().isin({"nan", "none", "null"})
    invalid = ~as_text.map(lambda value: bool(SYMBOL_RE.fullmatch(value))) | reserved
    if invalid.any():
        bad = sorted(as_text[invalid].unique())
        raise ValueError(f"{source} contains noncanonical symbol values: {bad}")
    return as_text


def _validate_keys(frame: pd.DataFrame, *, source: str) -> None:
    symbols = _validate_symbols(frame, source=source)
    bucket = pd.to_numeric(frame["bucket_start_ms"], errors="coerce")
    invalid_bucket = (
        bucket.isna()
        | ~bucket.map(lambda value: math.isfinite(float(value)))
        | (bucket < 0)
        | (bucket % 1 != 0)
    )
    if invalid_bucket.any():
        raise ValueError(f"{source} contains invalid bucket_start_ms")
    canonical = pd.DataFrame(
        {"symbol": symbols, "bucket_start_ms": bucket.astype("int64")}
    )
    duplicated = canonical.duplicated(["symbol", "bucket_start_ms"])
    if duplicated.any():
        raise ValueError(f"{source} contains duplicate symbol/bucket_start_ms keys")


def _coerce_numeric(
    frame: pd.DataFrame, columns: tuple[str, ...], *, source: str
) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].isna() | ~out[column].map(
            lambda value: math.isfinite(float(value))
        )
        if invalid.any():
            raise ValueError(f"{source} contains non-finite numeric values in {column}")
    return out


def _validate_positive_prices(
    frame: pd.DataFrame, columns: tuple[str, ...], *, source: str
) -> None:
    for column in columns:
        if (frame[column] <= 0).any():
            raise ValueError(
                f"{source} must contain positive finite prices in {column}"
            )


def _risk_state(row: pd.Series, config: ReferenceContextConfig) -> str:
    if row["reference_coverage"] != "REFERENCE_AVAILABLE":
        return "REFERENCE_MISSING"
    ref_return = float(row["reference_return_bps"])
    ref_vol = float(row["reference_vol_bps"])
    if ref_vol >= config.high_vol_bps:
        if ref_return <= config.risk_off_return_bps:
            return "HIGH_VOL_RISK_OFF"
        if ref_return >= config.risk_on_return_bps:
            return "HIGH_VOL_RISK_ON"
        return "HIGH_VOL_NEUTRAL"
    if ref_return >= config.risk_on_return_bps:
        return "RISK_ON"
    if ref_return <= config.risk_off_return_bps:
        return "RISK_OFF"
    return "RISK_NEUTRAL"


def _market_beta_proxy(reference: pd.DataFrame) -> pd.DataFrame:
    majors = reference[reference["symbol"].isin(["BTC", "ETH"])].copy()
    if majors.empty:
        return pd.DataFrame(
            columns=[
                "bucket_start_ms",
                "btc_reference_return_bps",
                "eth_reference_return_bps",
                "btc_eth_beta_proxy_bps",
                "btc_eth_beta_proxy_coverage",
            ]
        )
    pivot = majors.pivot(
        index="bucket_start_ms", columns="symbol", values="reference_return_bps"
    )
    out = pd.DataFrame(index=pivot.index)
    out["btc_reference_return_bps"] = pivot["BTC"] if "BTC" in pivot.columns else pd.NA
    out["eth_reference_return_bps"] = pivot["ETH"] if "ETH" in pivot.columns else pd.NA
    has_btc = out["btc_reference_return_bps"].notna()
    has_eth = out["eth_reference_return_bps"].notna()
    out["btc_eth_beta_proxy_coverage"] = "MISSING_MAJOR_REFERENCE"
    out.loc[has_btc ^ has_eth, "btc_eth_beta_proxy_coverage"] = (
        "PARTIAL_MAJOR_REFERENCE"
    )
    out.loc[has_btc & has_eth, "btc_eth_beta_proxy_coverage"] = "FULL_MAJOR_REFERENCE"
    out["btc_eth_beta_proxy_bps"] = pd.NA
    out.loc[has_btc & has_eth, "btc_eth_beta_proxy_bps"] = out.loc[
        has_btc & has_eth, ["btc_reference_return_bps", "eth_reference_return_bps"]
    ].mean(axis=1)
    return out.reset_index()


def build_reference_context(
    state: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    config: ReferenceContextConfig = ReferenceContextConfig(),
) -> pd.DataFrame:
    """Join Pacifica rows to reference context without imputing missing rows."""
    _validate_required(state, REQUIRED_STATE_COLUMNS, source="state")
    _validate_required(reference, REQUIRED_REFERENCE_COLUMNS, source="reference")
    _validate_keys(state, source="state")
    _validate_keys(reference, source="reference")

    state_numeric = _coerce_numeric(
        state,
        (
            "bucket_start_ms",
            "last_mid",
            "mid_return_bps",
            "realized_vol_bps",
            "funding",
        ),
        source="state",
    )
    reference_numeric = _coerce_numeric(
        reference,
        (
            "bucket_start_ms",
            "reference_mid",
            "reference_return_bps",
            "reference_vol_bps",
            "reference_funding",
        ),
        source="reference",
    )
    state_numeric["symbol"] = _validate_symbols(state_numeric, source="state")
    reference_numeric["symbol"] = _validate_symbols(
        reference_numeric, source="reference"
    )
    if (state_numeric["realized_vol_bps"] < 0).any():
        raise ValueError(
            "state must contain nonnegative volatility in realized_vol_bps"
        )
    if (reference_numeric["reference_vol_bps"] < 0).any():
        raise ValueError(
            "reference must contain nonnegative volatility in reference_vol_bps"
        )
    _validate_positive_prices(state_numeric, ("last_mid",), source="state")
    _validate_positive_prices(reference_numeric, ("reference_mid",), source="reference")

    merged = state_numeric.merge(
        reference_numeric,
        on=["symbol", "bucket_start_ms"],
        how="left",
        suffixes=("", "_reference_input"),
        indicator=True,
    )
    merged["reference_coverage"] = merged["_merge"].map(
        {
            "both": "REFERENCE_AVAILABLE",
            "left_only": "MISSING_REFERENCE",
            "right_only": "UNUSED_REFERENCE",
        }
    )
    merged = merged.drop(columns=["_merge"])
    available = merged["reference_coverage"] == "REFERENCE_AVAILABLE"
    merged["cross_venue_premium_bps"] = pd.NA
    merged.loc[available, "cross_venue_premium_bps"] = (
        (merged.loc[available, "last_mid"] - merged.loc[available, "reference_mid"])
        / merged.loc[available, "reference_mid"]
        * 10_000.0
    )
    merged["funding_divergence_bps"] = pd.NA
    merged.loc[available, "funding_divergence_bps"] = (
        merged.loc[available, "funding"] - merged.loc[available, "reference_funding"]
    ) * 10_000.0

    beta = _market_beta_proxy(reference_numeric)
    merged = merged.merge(beta, on="bucket_start_ms", how="left")
    if "btc_eth_beta_proxy_coverage" not in merged.columns:
        merged["btc_eth_beta_proxy_coverage"] = "MISSING_MAJOR_REFERENCE"
    else:
        merged["btc_eth_beta_proxy_coverage"] = merged[
            "btc_eth_beta_proxy_coverage"
        ].fillna("MISSING_MAJOR_REFERENCE")
    merged["broad_crypto_risk_state"] = merged.apply(
        _risk_state, axis=1, args=(config,)
    )
    merged["reference_context_version"] = REFERENCE_CONTEXT_VERSION

    first = [column for column in OUTPUT_COLUMNS_FIRST if column in merged.columns]
    rest = [column for column in merged.columns if column not in first]
    return (
        merged[first + rest]
        .sort_values(["symbol", "bucket_start_ms"])
        .reset_index(drop=True)
    )


def summarize_risk_states(context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame(columns=["broad_crypto_risk_state", "rows", "row_share"])
    total = len(context)
    summary = context.groupby("broad_crypto_risk_state", as_index=False).agg(
        rows=("symbol", "size")
    )
    summary["row_share"] = summary["rows"] / total
    return summary.sort_values(
        ["rows", "broad_crypto_risk_state"], ascending=[False, True]
    ).reset_index(drop=True)


def summarize_symbols(context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "rows",
                "reference_available_rows",
                "missing_reference_rows",
                "avg_cross_venue_premium_bps",
                "avg_funding_divergence_bps",
            ]
        )
    summary = (
        context.groupby("symbol", as_index=False)
        .agg(
            rows=("bucket_start_ms", "size"),
            reference_available_rows=(
                "reference_coverage",
                lambda s: int((s == "REFERENCE_AVAILABLE").sum()),
            ),
            missing_reference_rows=(
                "reference_coverage",
                lambda s: int((s == "MISSING_REFERENCE").sum()),
            ),
            avg_cross_venue_premium_bps=("cross_venue_premium_bps", "mean"),
            avg_funding_divergence_bps=("funding_divergence_bps", "mean"),
        )
        .sort_values(
            ["missing_reference_rows", "rows", "symbol"], ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )
    return summary


def _verdict(context: pd.DataFrame) -> str:
    if context.empty:
        return "NO_ROWS_DIAGNOSTIC"
    if (context["reference_coverage"] == "REFERENCE_AVAILABLE").sum() == 0:
        return "NO_REFERENCE_COVERAGE_DIAGNOSTIC"
    return "DIAGNOSTIC_REFERENCE_CONTEXT_BUILT"


def write_reference_context_report(
    context: pd.DataFrame,
    out_dir: Path,
    *,
    config: ReferenceContextConfig = ReferenceContextConfig(),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_summary = summarize_risk_states(context)
    symbol_summary = summarize_symbols(context)
    verdict = _verdict(context)
    config_frame = pd.DataFrame(
        [{**asdict(config), "reference_context_version": REFERENCE_CONTEXT_VERSION}]
    )

    context.to_csv(out_dir / "reference_context.csv", index=False)
    risk_summary.to_csv(out_dir / "risk_state_summary.csv", index=False)
    symbol_summary.to_csv(out_dir / "symbol_reference_summary.csv", index=False)
    config_frame.to_csv(out_dir / "config.csv", index=False)

    lines = [
        "# Cross-venue/reference market context",
        "",
        f"Verdict: `{verdict}`",
        "",
        "This is a diagnostic context layer, not a trade signal and not permission to paper/live trade.",
        "It distinguishes Pacifica-local observations from broad crypto/reference-market states using pluggable local CSV/parquet inputs.",
        "",
        "## Interpretation discipline",
        "",
        "- Missing reference rows are flagged as `MISSING_REFERENCE`; they are not imputed.",
        "- Broad risk states are fixed diagnostic labels, not optimized strategy parameters.",
        "- Cross-venue premium/discount and funding divergence require external source-quality review before use in any strategy.",
        "",
        "## Risk-state summary",
        "",
        dataframe_to_markdown_table(risk_summary),
        "",
        "## Symbol reference summary",
        "",
        dataframe_to_markdown_table(symbol_summary, max_rows=25),
        "",
        "## Config",
        "",
        dataframe_to_markdown_table(config_frame),
        "",
        "## Artifacts",
        "",
        "- `reference_context.csv`",
        "- `risk_state_summary.csv`",
        "- `symbol_reference_summary.csv`",
        "- `config.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    return {
        "verdict": verdict,
        "rows": int(len(context)),
        "reference_available_rows": (
            int((context["reference_coverage"] == "REFERENCE_AVAILABLE").sum())
            if not context.empty
            else 0
        ),
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Pacifica diagnostic reference-market context"
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    state = _read_frame(args.state)
    reference = _read_frame(args.reference)
    context = build_reference_context(state, reference)
    result = write_reference_context_report(context, args.out_dir)
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"reference_available_rows: {result['reference_available_rows']}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
