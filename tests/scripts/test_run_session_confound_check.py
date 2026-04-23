import json
from pathlib import Path

from scripts.run_session_confound_check import run_check


def test_run_check_writes_json_and_md(tmp_path):
    # Use a tiny synthetic cache for smoke-only — real run is in CLI
    out = tmp_path / "session-check"
    res = run_check(
        cache_dir=Path("data/cache"),
        symbols=["BTC", "ETH", "SOL"],
        out_path=out,
    )
    # Must always emit two artifacts even on small data
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".md").exists()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert "per_symbol" in payload
    assert "decision" in payload
    assert payload["decision"] in {"prune_last_features", "no_action"}
