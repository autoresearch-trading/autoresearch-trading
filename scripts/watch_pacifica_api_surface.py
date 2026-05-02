#!/usr/bin/env python3
"""Watch Pacifica's public API/docs surface for new collectable market data.

This is a read-only docs/API-surface watcher. It compares currently discoverable
public REST paths, websocket subscription sources, and candle intervals against a
reviewed baseline. It does not mutate the collector or delete data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_BASELINE = Path("docs/ops/pacifica-api-surface-baseline.json")
DEFAULT_OUT_DIR = Path("docs/ops/pacifica-api-surface-watch")
DEFAULT_URLS = (
    "https://docs.pacifica.fi/api-documentation/api",
    "https://docs.pacifica.fi/api-documentation/changelog",
)
PUBLIC_REST_PREFIX = "/api/v1"
MARKET_DATA_REST_PREFIXES = (
    "/info",
    "/funding",
    "/kline",
)
PRIVATE_REST_PREFIXES = (
    "/orders",
    "/order",
    "/positions",
    "/account",
    "/user",
    "/wallet",
    "/auth",
    "/login",
    "/withdraw",
    "/deposit",
)
PRIVATE_WS_SOURCES = {
    "account",
    "orders",
    "positions",
    "fills",
    "user",
    "wallet",
    "private",
}
KNOWN_PUBLIC_WS_SOURCES = {
    "prices",
    "trades",
    "book",
    "bbo",
    "candle",
    "mark_price_candle",
}
KNOWN_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"}


@dataclass(frozen=True)
class ApiSurface:
    rest_paths: set[str] = field(default_factory=set)
    ws_sources: set[str] = field(default_factory=set)
    intervals: set[str] = field(default_factory=set)
    source_urls: set[str] = field(default_factory=set)
    fetch_errors: dict[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rest_paths": sorted(self.rest_paths),
            "ws_sources": sorted(self.ws_sources),
            "intervals": sorted(self.intervals),
            "source_urls": sorted(self.source_urls),
        }
        if self.fetch_errors:
            out["fetch_errors"] = dict(sorted(self.fetch_errors.items()))
        return out

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ApiSurface":
        return cls(
            rest_paths=set(map(str, data.get("rest_paths", []))),
            ws_sources=set(map(str, data.get("ws_sources", []))),
            intervals=set(map(str, data.get("intervals", []))),
            source_urls=set(map(str, data.get("source_urls", []))),
            fetch_errors={
                str(k): str(v) for k, v in data.get("fetch_errors", {}).items()
            },
        )

    def merged(self, other: "ApiSurface") -> "ApiSurface":
        errors = dict(self.fetch_errors)
        errors.update(other.fetch_errors)
        return ApiSurface(
            rest_paths=self.rest_paths | other.rest_paths,
            ws_sources=self.ws_sources | other.ws_sources,
            intervals=self.intervals | other.intervals,
            source_urls=self.source_urls | other.source_urls,
            fetch_errors=errors,
        )


def _strip_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text)


def _normalize_rest_path(raw_path: str) -> str | None:
    path = raw_path.strip().rstrip(".,;:'\")]")
    if not path.startswith("/"):
        return None
    if path.startswith(PUBLIC_REST_PREFIX):
        path = path[len(PUBLIC_REST_PREFIX) :] or "/"
    elif path.startswith("/api/"):
        return None
    if any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in PRIVATE_REST_PREFIXES
    ):
        return None
    if path in {"/", "/ws"}:
        return None
    if not any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in MARKET_DATA_REST_PREFIXES
    ):
        return None
    return path


def _normalize_ws_source(source: str) -> str | None:
    source = source.strip().strip("`'\"").lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", source):
        return None
    if source in PRIVATE_WS_SOURCES:
        return None
    return source


def extract_surface_from_text(text: str, *, url: str) -> ApiSurface:
    """Extract collectable public market-data surface hints from docs text/HTML."""

    clean = _strip_html(text)
    rest_paths: set[str] = set()
    ws_sources: set[str] = set()
    intervals: set[str] = set()

    rest_patterns = [
        r"https?://api\.pacifica\.fi/api/v1(?P<path>/[A-Za-z0-9_./{}:-]+)",
        r"^\s+(?P<path>/api/v1/[A-Za-z0-9_./{}:-]+):\s*$",
        r"\b(?:GET|POST|PUT|DELETE)\s+(?P<path>/api/v1/[A-Za-z0-9_./{}:-]+)",
        r"\b(?:GET|POST|PUT|DELETE)\s+(?P<path>/[A-Za-z0-9_./{}:-]+)",
    ]
    for pattern in rest_patterns:
        for match in re.finditer(pattern, clean, flags=re.I | re.M):
            path = _normalize_rest_path(match.group("path"))
            if path:
                rest_paths.add(path)

    source_patterns = [
        r"[\"']source[\"']\s*:\s*[\"'](?P<source>[A-Za-z][A-Za-z0-9_]{1,40})[\"']",
        r"\bsource\s*[:=]\s*[\"'](?P<source>[A-Za-z][A-Za-z0-9_]{1,40})[\"']",
        r"\bsource\s*[:=]\s*(?P<source>[A-Za-z][A-Za-z0-9_]{1,40})\b",
    ]
    for pattern in source_patterns:
        for match in re.finditer(pattern, clean, flags=re.I):
            source = _normalize_ws_source(match.group("source"))
            if source:
                ws_sources.add(source)

    intervals.update(
        re.findall(r"\b(?:1|3|5|15|30)m\b|\b(?:1|2|4|8|12)h\b|\b1d\b", clean)
    )
    intervals = {interval for interval in intervals if interval in KNOWN_INTERVALS}

    return ApiSurface(
        rest_paths=rest_paths,
        ws_sources=ws_sources,
        intervals=intervals,
        source_urls={url},
    )


def fetch_text(url: str, *, timeout_s: float = 30.0) -> str:
    req = Request(
        url, headers={"User-Agent": "autoresearch-trading-api-surface-watch/1.0"}
    )
    with urlopen(
        req, timeout=timeout_s
    ) as response:  # noqa: S310 - public docs watcher
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def linked_openapi_urls(text: str) -> list[str]:
    urls: set[str] = set()
    for match in re.findall(r"https?://[^\"\s<>]+openapi\.ya?ml[^\"\s<>]*", text):
        url = unescape(match).replace("\\u0026", "&").rstrip("\\")
        urls.add(url)
    return sorted(urls)


def discover_surface(urls: list[str], *, timeout_s: float = 30.0) -> ApiSurface:
    surface = ApiSurface()
    seen_urls: set[str] = set()
    pending = list(urls)
    while pending:
        url = pending.pop(0)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            text = fetch_text(url, timeout_s=timeout_s)
        except Exception as exc:  # pragma: no cover - network dependent
            surface = surface.merged(ApiSurface(fetch_errors={url: str(exc)}))
            continue
        surface = surface.merged(extract_surface_from_text(text, url=url))
        for linked in linked_openapi_urls(text):
            if linked not in seen_urls:
                pending.append(linked)
    return surface


def load_baseline(path: Path) -> ApiSurface:
    return ApiSurface.from_json_dict(json.loads(path.read_text()))


def write_baseline(path: Path, surface: ApiSurface) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = surface.to_json_dict()
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["note"] = (
        "Reviewed baseline for public Pacifica market-data surface. Update only "
        "after confirming collector coverage implications."
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def compare_surfaces(baseline: ApiSurface, current: ApiSurface) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    changed = False
    for name in ("rest_paths", "ws_sources", "intervals"):
        old = getattr(baseline, name)
        new = getattr(current, name)
        added = sorted(new - old)
        removed = sorted(old - new) if new else []
        diff[name] = {"added": added, "removed": removed}
        changed = changed or bool(added)
    diff["changed"] = changed
    diff["baseline_source_urls"] = sorted(baseline.source_urls)
    diff["current_source_urls"] = sorted(current.source_urls)
    if current.fetch_errors:
        diff["fetch_errors"] = dict(sorted(current.fetch_errors.items()))
    return diff


def _bullet_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- `{value}`" for value in values)


def write_report(
    out_dir: Path, *, baseline: ApiSurface, current: ApiSurface, diff: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_surface.json").write_text(
        json.dumps(baseline.to_json_dict(), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "current_surface.json").write_text(
        json.dumps(current.to_json_dict(), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "api_surface_diff.json").write_text(
        json.dumps(diff, indent=2, sort_keys=True) + "\n"
    )

    verdict = "CHANGED" if diff.get("changed") else "UNCHANGED"
    lines = [
        "# Pacifica API Surface Watch",
        "",
        f"Checked at: {datetime.now(timezone.utc).isoformat()}",
        f"Verdict: {verdict}",
        "",
        "Purpose: alert when Pacifica's public docs/API surface appears to expose new collectable market-data endpoints, websocket sources, or intervals. This is read-only and does not update the collector automatically.",
        "",
        "## REST paths",
        "",
        "Added:",
        _bullet_list(diff["rest_paths"]["added"]),
        "",
        "Removed:",
        _bullet_list(diff["rest_paths"]["removed"]),
        "",
        "## WebSocket sources",
        "",
        "Added:",
        _bullet_list(diff["ws_sources"]["added"]),
        "",
        "Removed:",
        _bullet_list(diff["ws_sources"]["removed"]),
        "",
        "## Intervals",
        "",
        "Added:",
        _bullet_list(diff["intervals"]["added"]),
        "",
        "Removed:",
        _bullet_list(diff["intervals"]["removed"]),
        "",
        "## Current discovered surface",
        "",
        "REST paths:",
        _bullet_list(sorted(current.rest_paths)),
        "",
        "WebSocket sources:",
        _bullet_list(sorted(current.ws_sources)),
        "",
        "Intervals:",
        _bullet_list(sorted(current.intervals)),
        "",
        "Source URLs:",
        _bullet_list(sorted(current.source_urls)),
    ]
    if current.fetch_errors:
        lines.extend(["", "## Fetch errors", ""])
        lines.extend(
            f"- `{url}`: {error}" for url, error in sorted(current.fetch_errors.items())
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- baseline_surface.json",
            "- current_surface.json",
            "- api_surface_diff.json",
            "",
            "If verdict is CHANGED, manually inspect the docs/API, decide whether the new surface is public market data, add tests, then update the collector/silver layers and baseline in a separate reviewed change.",
            "",
        ]
    )
    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines))
    return readme


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="Docs/API URL to inspect. Repeatable.",
    )
    parser.add_argument(
        "--from-file",
        action="append",
        dest="files",
        type=Path,
        help="Local docs/text fixture to inspect. Repeatable.",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current discovered surface as baseline, then exit.",
    )
    parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Exit 2 if current surface differs from baseline.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    urls = args.urls or list(DEFAULT_URLS)

    current = discover_surface(urls, timeout_s=args.timeout_s)
    for file_path in args.files or []:
        current = current.merged(
            extract_surface_from_text(file_path.read_text(), url=str(file_path))
        )

    if args.write_baseline:
        write_baseline(args.baseline, current)
        print(f"wrote baseline {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(
            f"baseline not found: {args.baseline}; run with --write-baseline after review",
            file=sys.stderr,
        )
        return 2

    baseline = load_baseline(args.baseline)
    diff = compare_surfaces(baseline, current)
    readme = write_report(args.out_dir, baseline=baseline, current=current, diff=diff)
    print(f"wrote {readme}; changed={diff['changed']}")
    if args.fail_on_change and diff["changed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
