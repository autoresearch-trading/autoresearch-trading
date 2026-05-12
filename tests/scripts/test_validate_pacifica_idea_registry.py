from pathlib import Path

from scripts.validate_pacifica_idea_registry import (
    REQUIRED_FIELDS,
    parse_idea_registry,
    validate_idea_registry,
    write_registry_validation_report,
)

VALID_REGISTRY = """
# Pacifica Research Idea Registry

## IDEA-001: Toxic regime no-trade overlay

- **Hypothesis:** High toxicity minutes have worse forward downside than ordinary minutes after costs.
- **Mechanical label:** Forward 60 minute adverse excursion and downside deviation from bucket close.
- **Trade/risk action:** Skip or reduce size during the fixed top toxicity buckets; no directional entry signal.
- **Cost model:** Pacifica locked execution economics v1: 4 bps taker per side, slippage/adverse-selection bps, and funding debits included.
- **Validation window:** Purged chronological OOS windows after at least 30 distinct archive days; 60+ preferred.
- **Frozen parameters:** Top 10/20/30 percent toxicity cuts, fixed before additional maturity reruns.
- **Kill criteria:** Fails if post-cost Sortino/drawdown does not improve versus same-frequency controls or if retention/sample/concentration gates fail.
- **OOS plan:** Use run_pacifica_walk_forward_validation with random same-frequency controls and no threshold retuning.
- **Result/verdict:** INSUFFICIENT_SAMPLE_DIAGNOSTIC; current archive is too young for an edge claim.
""".strip()


def test_parse_idea_registry_extracts_required_fields() -> None:
    ideas = parse_idea_registry(VALID_REGISTRY)

    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.idea_id == "IDEA-001"
    assert idea.title == "Toxic regime no-trade overlay"
    assert set(REQUIRED_FIELDS).issubset(idea.fields)
    assert "High toxicity" in idea.fields["hypothesis"]


def test_validate_idea_registry_accepts_complete_falsifiable_idea() -> None:
    result = validate_idea_registry(VALID_REGISTRY)

    assert result.verdict == "PASS"
    assert result.idea_count == 1
    assert result.error_count == 0
    assert result.errors == []


def test_validate_idea_registry_fails_closed_on_missing_required_field() -> None:
    text = VALID_REGISTRY.replace("- **Kill criteria:**", "- **Kill note:**")

    result = validate_idea_registry(text)

    assert result.verdict == "FAIL"
    assert any(
        "IDEA-001 missing required field: kill criteria" in error
        for error in result.errors
    )


def test_validate_idea_registry_fails_closed_on_duplicate_ids_and_placeholders() -> (
    None
):
    duplicate = (
        VALID_REGISTRY
        + "\n\n"
        + VALID_REGISTRY.replace(
            "High toxicity minutes have worse forward downside than ordinary minutes after costs.",
            "TBD",
        )
    )

    result = validate_idea_registry(duplicate)

    assert result.verdict == "FAIL"
    assert any("duplicate idea id: IDEA-001" in error for error in result.errors)
    assert any("placeholder" in error for error in result.errors)


def test_validate_idea_registry_rejects_non_mechanical_language() -> None:
    vague = VALID_REGISTRY.replace(
        "Forward 60 minute adverse excursion and downside deviation from bucket close.",
        "Look at whether the chart feels bullish.",
    )

    result = validate_idea_registry(vague)

    assert result.verdict == "FAIL"
    assert any(
        "mechanical label" in error and "measurable" in error for error in result.errors
    )


def test_validate_idea_registry_rejects_adversarial_negated_controls() -> None:
    bad = VALID_REGISTRY
    replacements = {
        "High toxicity minutes have worse forward downside than ordinary minutes after costs.": "We will find alpha.",
        "Forward 60 minute adverse excursion and downside deviation from bucket close.": "The chart feels bullish in minute buckets.",
        "Skip or reduce size during the fixed top toxicity buckets; no directional entry signal.": "Buy when it looks good.",
        "Pacifica locked execution economics v1: 4 bps taker per side, slippage/adverse-selection bps, and funding debits included.": "Ignore fees and slippage because edge is huge.",
        "Purged chronological OOS windows after at least 30 distinct archive days; 60+ preferred.": "In-sample after tuning.",
        "Top 10/20/30 percent toxicity cuts, fixed before additional maturity reruns.": "Parameters may be changed after seeing results.",
        "Fails if post-cost Sortino/drawdown does not improve versus same-frequency controls or if retention/sample/concentration gates fail.": "No kill criteria; keep trying until it works.",
        "Use run_pacifica_walk_forward_validation with random same-frequency controls and no threshold retuning.": "No OOS planned.",
        "INSUFFICIENT_SAMPLE_DIAGNOSTIC; current archive is too young for an edge claim.": "PASS edge claimed.",
    }
    for old, new in replacements.items():
        bad = bad.replace(old, new)

    result = validate_idea_registry(bad)

    assert result.verdict == "FAIL"
    joined = "\n".join(result.errors)
    assert "hypothesis" in joined
    assert "mechanical label" in joined
    assert "trade/risk action" in joined
    assert "cost model" in joined
    assert "validation window" in joined
    assert "frozen parameters" in joined
    assert "kill criteria" in joined
    assert "OOS plan" in joined
    assert "result/verdict" in joined


def test_validate_idea_registry_rejects_specific_negated_semantics() -> None:
    bad_cases = [
        (
            "cost model",
            "No fees, slippage, funding, or costs are modeled because the edge is large.",
        ),
        ("cost model", "Evaluate without fees, slippage, funding, or other costs."),
        (
            "kill criteria",
            "No failure gates; continue retuning until Sortino improves.",
        ),
        (
            "frozen parameters",
            "Parameters are not fixed before evaluation and can be retuned.",
        ),
        ("result/verdict", "PASS; edge is proven."),
        (
            "mechanical label",
            "Bullish-looking forward 60 minute return bucket from visual chart review.",
        ),
    ]
    for field, replacement in bad_cases:
        original_line = next(
            line
            for line in VALID_REGISTRY.splitlines()
            if line.lower().startswith(f"- **{field}:**")
        )
        bad = VALID_REGISTRY.replace(
            original_line, f"- **{field.title()}:** {replacement}"
        )
        result = validate_idea_registry(bad)
        assert result.verdict == "FAIL", field
        assert any(field in error for error in result.errors), result.errors


def test_write_registry_validation_report_outputs_markdown_csv_and_json(
    tmp_path: Path,
) -> None:
    result = validate_idea_registry(VALID_REGISTRY)
    written = write_registry_validation_report(result, tmp_path)

    assert written["registry_validation_verdict"] == "PASS"
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "idea_registry_validation.csv").exists()
    assert (tmp_path / "summary.json").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "Research Idea Registry Validation" in readme
    assert "Registry schema verdict: `PASS`" in readme
    assert "Research/edge verdicts remain separate" in readme
    csv_text = (tmp_path / "idea_registry_validation.csv").read_text()
    assert "registration_schema_verdict" in csv_text
    assert "research_result_verdict" in csv_text
    assert "IDEA-001" in csv_text
