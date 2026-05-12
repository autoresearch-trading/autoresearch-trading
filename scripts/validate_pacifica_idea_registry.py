# scripts/validate_pacifica_idea_registry.py
"""Validate the Pacifica research idea registry.

The registry is a fail-closed pre-registration layer. It forces each future edge
idea to name a falsifiable hypothesis, mechanical label, cost model, OOS plan,
and kill criteria before implementation or alpha claims.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_REGISTRY_PATH = Path("docs/research/pacifica-idea-registry.md")
DEFAULT_OUT_DIR = Path("docs/research/pacifica-idea-registry-validation")
IDEA_HEADING_RE = re.compile(r"^##\s+(IDEA-\d{3,}):\s+(.+?)\s*$", re.MULTILINE)
FIELD_RE = re.compile(r"^-\s+\*\*(.+?)\:\*\*\s*(.+?)\s*$")
PLACEHOLDERS = {"tbd", "todo", "n/a", "na", "none", "unknown", "?"}
REQUIRED_FIELDS: tuple[str, ...] = (
    "hypothesis",
    "mechanical label",
    "trade/risk action",
    "cost model",
    "validation window",
    "frozen parameters",
    "kill criteria",
    "oos plan",
    "result/verdict",
)
MEASURABLE_OUTCOME_TERMS = (
    "return",
    "drawdown",
    "sortino",
    "pnl",
    "adverse",
    "excursion",
    "deviation",
    "slippage",
    "downside",
    "spread",
    "volatility",
    "premium",
    "discount",
)
COMPARISON_TERMS = (
    "than",
    "versus",
    "vs",
    "control",
    "baseline",
    "ordinary",
    "same-frequency",
    "after costs",
)
HORIZON_TERMS = ("minute", "hour", "bucket", "window", "forward", "oos", "day")
MECHANICAL_LABEL_TERMS = (
    "return",
    "drawdown",
    "sortino",
    "pnl",
    "adverse",
    "excursion",
    "deviation",
    "probability",
    "bps",
    "basis",
    "spread",
    "depth",
    "funding",
    "slippage",
    "bucket",
    "minute",
    "hour",
    "threshold",
    "quantile",
    "percent",
    "ratio",
)
COST_MODEL_TERMS = ("fee", "fees", "bps", "slippage", "adverse", "funding", "cost")
OOS_TERMS = (
    "oos",
    "out-of-sample",
    "walk-forward",
    "walk_forward",
    "chronological",
    "purged",
)
KILL_TERMS = (
    "fail",
    "fails",
    "kill",
    "stop",
    "reject",
    "drawdown",
    "sortino",
    "pnl",
    "gate",
)
VERDICT_TERMS = (
    "diagnostic",
    "insufficient",
    "pending",
    "pass",
    "fail",
    "provisional",
    "validation",
    "killed",
)
NEGATED_CONTROL_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(ignore|exclude|skip)\s+(fees?|costs?|slippage|funding)\b",
        "cost model rejects costs",
    ),
    (r"\b(no|without)\s+(fees?|costs?|slippage|funding)\b", "cost model rejects costs"),
    (
        r"\b(fees?|costs?|slippage|funding)\b.{0,40}\b(not modeled|not included|ignored|excluded)\b",
        "cost model rejects costs",
    ),
    (
        r"\bno\s+(oos|out[- ]of[- ]sample|walk[-_ ]forward)\b",
        "OOS plan rejects out-of-sample validation",
    ),
    (r"\bno\s+(kill|failure gates?|fail gates?)\b", "kill criteria rejects kill gates"),
    (r"\bcontinue\s+retuning\b", "kill criteria rejects fixed failure gates"),
    (
        r"\bin[- ]sample\s+(only|after tuning)\b",
        "validation window is in-sample/post-tuning",
    ),
    (
        r"\b(parameters?|thresholds?)\s+may\s+be\s+changed\s+after\b",
        "frozen parameters allow post-hoc changes",
    ),
    (r"\bnot\s+fixed\s+before\b", "frozen parameters reject fixed pre-registration"),
    (r"\bcan\s+be\s+retuned\b", "frozen parameters allow retuning"),
    (
        r"\b(retune after|tune after|after seeing results)\b",
        "frozen parameters allow retuning",
    ),
    (
        r"\b(edge|alpha)\s+(is\s+)?(claimed|confirmed|proven)\b",
        "result/verdict claims edge before validation",
    ),
)
SUBJECTIVE_PATTERNS = (
    r"\bfeels?\b",
    r"\blooks?\s+good\b",
    r"\bchart\s+(looks|feels|review)\b",
    r"\bvisual\s+chart\s+review\b",
    r"\bbullish[- ]looking\b",
    r"\bbearish[- ]looking\b",
    r"\beyeball\b",
    r"\bdiscretionary\b",
)


@dataclass(frozen=True)
class RegisteredIdea:
    idea_id: str
    title: str
    fields: dict[str, str]


@dataclass(frozen=True)
class RegistryValidationResult:
    verdict: str
    idea_count: int
    error_count: int
    errors: list[str]
    ideas: list[RegisteredIdea]


def _normalize_field_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower().strip(".:-")
    if normalized in PLACEHOLDERS:
        return True
    return "tbd" in normalized or "todo" in normalized


def parse_idea_registry(text: str) -> list[RegisteredIdea]:
    """Parse IDEA headings and bullet fields from registry Markdown."""
    matches = list(IDEA_HEADING_RE.finditer(text))
    ideas: list[RegisteredIdea] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        fields: dict[str, str] = {}
        for raw_line in body.splitlines():
            field_match = FIELD_RE.match(raw_line.strip())
            if not field_match:
                continue
            name = _normalize_field_name(field_match.group(1))
            value = field_match.group(2).strip()
            fields[name] = value
        ideas.append(
            RegisteredIdea(
                idea_id=match.group(1), title=match.group(2).strip(), fields=fields
            )
        )
    return ideas


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in terms)


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _negated_control_errors(idea: RegisteredIdea) -> list[str]:
    errors: list[str] = []
    for field, value in idea.fields.items():
        lowered = value.lower()
        for pattern, message in NEGATED_CONTROL_PATTERNS:
            if re.search(pattern, lowered):
                errors.append(f"{idea.idea_id} field '{field}' {message}")
    return errors


def _validate_idea(idea: RegisteredIdea) -> list[str]:
    errors: list[str] = []
    if not idea.title or _is_placeholder(idea.title):
        errors.append(f"{idea.idea_id} has blank/placeholder title")
    for field in REQUIRED_FIELDS:
        value = idea.fields.get(field, "")
        if not value:
            errors.append(f"{idea.idea_id} missing required field: {field}")
            continue
        if _is_placeholder(value):
            errors.append(f"{idea.idea_id} field '{field}' contains placeholder text")
    hypothesis = idea.fields.get("hypothesis", "")
    if hypothesis:
        if not _contains_any(hypothesis, MEASURABLE_OUTCOME_TERMS) or not _contains_any(
            hypothesis, COMPARISON_TERMS
        ):
            errors.append(
                f"{idea.idea_id} hypothesis must name a measurable outcome and comparison/control"
            )
    mechanical = idea.fields.get("mechanical label", "")
    if mechanical:
        if (
            _matches_any(mechanical, SUBJECTIVE_PATTERNS)
            or not _contains_any(mechanical, MEASURABLE_OUTCOME_TERMS)
            or not _contains_any(mechanical, HORIZON_TERMS)
        ):
            errors.append(
                f"{idea.idea_id} mechanical label must be measurable, not qualitative"
            )
    action = idea.fields.get("trade/risk action", "")
    if action and _matches_any(action, SUBJECTIVE_PATTERNS):
        errors.append(
            f"{idea.idea_id} trade/risk action must be rule-based, not discretionary"
        )
    cost_model = idea.fields.get("cost model", "")
    if cost_model and not _contains_any(cost_model, COST_MODEL_TERMS):
        errors.append(
            f"{idea.idea_id} cost model must name fees/slippage/funding/cost assumptions"
        )
    validation_window = idea.fields.get("validation window", "")
    if validation_window and not _contains_any(
        validation_window, OOS_TERMS + HORIZON_TERMS
    ):
        errors.append(
            f"{idea.idea_id} validation window must name chronological/OOS timing"
        )
    frozen = idea.fields.get("frozen parameters", "")
    if frozen and not _contains_any(
        frozen, ("fixed", "frozen", "pre-registered", "locked", "before")
    ):
        errors.append(
            f"{idea.idea_id} frozen parameters must state parameters are fixed before evaluation"
        )
    oos_plan = idea.fields.get("oos plan", "")
    if oos_plan and not _contains_any(oos_plan, OOS_TERMS):
        errors.append(
            f"{idea.idea_id} OOS plan must name chronological/walk-forward/out-of-sample validation"
        )
    kill = idea.fields.get("kill criteria", "")
    if kill and not _contains_any(kill, KILL_TERMS):
        errors.append(
            f"{idea.idea_id} kill criteria must include explicit failure gates"
        )
    verdict = idea.fields.get("result/verdict", "")
    if verdict and not _contains_any(verdict, VERDICT_TERMS):
        errors.append(
            f"{idea.idea_id} result/verdict must use a clear diagnostic/pass/fail verdict"
        )
    errors.extend(_negated_control_errors(idea))
    return errors


def validate_idea_registry(text: str) -> RegistryValidationResult:
    ideas = parse_idea_registry(text)
    errors: list[str] = []
    if not ideas:
        errors.append("registry contains no IDEA-### entries")
    seen: set[str] = set()
    for idea in ideas:
        if idea.idea_id in seen:
            errors.append(f"duplicate idea id: {idea.idea_id}")
        seen.add(idea.idea_id)
        errors.extend(_validate_idea(idea))
    verdict = "PASS" if not errors else "FAIL"
    return RegistryValidationResult(
        verdict=verdict,
        idea_count=len(ideas),
        error_count=len(errors),
        errors=errors,
        ideas=ideas,
    )


def _fmt_markdown(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(_fmt_markdown(row[column]) for column in df.columns)
            + " |"
        )
    return "\n".join(lines)


def _idea_rows(result: RegistryValidationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idea in result.ideas:
        idea_errors = [error for error in result.errors if idea.idea_id in error]
        row = {
            "idea_id": idea.idea_id,
            "title": idea.title,
            "registration_schema_verdict": "PASS" if not idea_errors else "FAIL",
            "research_result_verdict": idea.fields.get("result/verdict", ""),
            "error_count": len(idea_errors),
            "errors": "; ".join(idea_errors),
        }
        for field in REQUIRED_FIELDS:
            row[field] = idea.fields.get(field, "")
        rows.append(row)
    if not rows:
        rows.append(
            {
                "idea_id": "",
                "title": "",
                "registration_schema_verdict": "FAIL",
                "research_result_verdict": "",
                "error_count": result.error_count,
                "errors": "; ".join(result.errors),
                **{field: "" for field in REQUIRED_FIELDS},
            }
        )
    return rows


def write_registry_validation_report(
    result: RegistryValidationResult, out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _idea_rows(result)
    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_dir / "idea_registry_validation.csv", index=False)
    summary = {
        "registry_validation_verdict": result.verdict,
        "idea_count": result.idea_count,
        "error_count": result.error_count,
        "errors": result.errors,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = [
        "# Research Idea Registry Validation",
        "",
        f"Registry schema verdict: `{result.verdict}`",
        "",
        "Research/edge verdicts remain separate from the registry schema verdict. A schema PASS only means each idea is falsifiable enough to test; it is not evidence of alpha and does not authorize paper/live trading.",
        "",
        "## Summary",
        "",
        f"- Ideas: {result.idea_count}",
        f"- Errors: {result.error_count}",
        "",
        "## Errors",
        "",
    ]
    if result.errors:
        readme.extend(f"- {error}" for error in result.errors)
    else:
        readme.append("- None")
    readme.extend(
        [
            "",
            "## Ideas",
            "",
            dataframe_to_markdown_table(
                report_df[
                    [
                        "idea_id",
                        "title",
                        "registration_schema_verdict",
                        "research_result_verdict",
                        "error_count",
                    ]
                ]
            ),
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme))
    return summary


def validate_file(
    registry_path: Path = DEFAULT_REGISTRY_PATH, out_dir: Path = DEFAULT_OUT_DIR
) -> RegistryValidationResult:
    text = registry_path.read_text()
    result = validate_idea_registry(text)
    write_registry_validation_report(result, out_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    result = validate_file(args.registry, args.out_dir)
    print(f"verdict: {result.verdict}")
    print(f"ideas: {result.idea_count}")
    print(f"errors: {result.error_count}")
    print(f"wrote report: {args.out_dir / 'README.md'}")
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
