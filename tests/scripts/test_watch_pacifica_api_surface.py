import json
from pathlib import Path

from scripts.watch_pacifica_api_surface import (
    ApiSurface,
    compare_surfaces,
    extract_surface_from_text,
    load_baseline,
    write_report,
)


def test_extract_surface_from_docs_text_finds_public_rest_ws_sources_and_intervals():
    text = """
    Subscribe with {"method":"subscribe","params":{"source":"trades","symbol":"BTC"}}
    Other sources: source: "liquidations", source: 'funding_rate'.
    GET https://api.pacifica.fi/api/v1/info
    GET /api/v1/info/prices
    GET /api/v1/funding/history
    intervals supported: 1m, 5m, 1h, 1d
    Private endpoint POST /api/v1/orders is out of scope.
    """

    surface = extract_surface_from_text(text, url="https://docs.pacifica.fi/api")

    assert surface.rest_paths == {"/info", "/info/prices", "/funding/history"}
    assert surface.ws_sources == {"trades", "liquidations", "funding_rate"}
    assert surface.intervals == {"1m", "5m", "1h", "1d"}
    assert surface.source_urls == {"https://docs.pacifica.fi/api"}


def test_compare_surfaces_reports_added_and_removed_items():
    baseline = ApiSurface(
        rest_paths={"/info", "/info/prices"},
        ws_sources={"prices", "trades"},
        intervals={"1m", "5m"},
        source_urls={"baseline"},
    )
    current = ApiSurface(
        rest_paths={"/info", "/funding/history"},
        ws_sources={"prices", "trades", "liquidations"},
        intervals={"1m", "15m"},
        source_urls={"current"},
    )

    diff = compare_surfaces(baseline, current)

    assert diff["rest_paths"]["added"] == ["/funding/history"]
    assert diff["rest_paths"]["removed"] == ["/info/prices"]
    assert diff["ws_sources"]["added"] == ["liquidations"]
    assert diff["ws_sources"]["removed"] == []
    assert diff["intervals"]["added"] == ["15m"]
    assert diff["intervals"]["removed"] == ["5m"]
    assert diff["changed"] is True


def test_load_baseline_round_trips_json(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "rest_paths": ["/info"],
                "ws_sources": ["trades"],
                "intervals": ["1m"],
                "source_urls": ["manual"],
            }
        )
    )

    assert load_baseline(path) == ApiSurface(
        rest_paths={"/info"},
        ws_sources={"trades"},
        intervals={"1m"},
        source_urls={"manual"},
    )


def test_write_report_creates_machine_and_human_readable_outputs(tmp_path):
    baseline = ApiSurface(rest_paths={"/info"}, ws_sources={"trades"}, intervals={"1m"})
    current = ApiSurface(
        rest_paths={"/info", "/funding/history"},
        ws_sources={"trades"},
        intervals={"1m"},
        source_urls={"https://docs.pacifica.fi/api"},
    )
    diff = compare_surfaces(baseline, current)

    readme = write_report(tmp_path, baseline=baseline, current=current, diff=diff)

    assert readme == tmp_path / "README.md"
    assert "Pacifica API Surface Watch" in readme.read_text()
    assert "CHANGED" in readme.read_text()
    assert "/funding/history" in readme.read_text()
    assert (
        json.loads((tmp_path / "api_surface_diff.json").read_text())["changed"] is True
    )
    assert json.loads((tmp_path / "current_surface.json").read_text())[
        "rest_paths"
    ] == [
        "/funding/history",
        "/info",
    ]
