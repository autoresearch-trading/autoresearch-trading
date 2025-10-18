from __future__ import annotations

import math
import re
import signal
import time
from contextlib import contextmanager
from typing import Callable, Dict, Generator, Iterable, Optional

_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)(ms|s|m)?\s*$")


def parse_duration(raw: str, *, default_unit: str = "s") -> float:
    """Parse a duration string such as '250ms', '1.5s', or '2m' and return seconds."""
    match = _DURATION_PATTERN.match(raw)
    if not match:
        raise ValueError(
            f"Invalid duration '{raw}'. Expected formats like '250ms', '1s', or '2m'."
        )
    value = float(match.group(1))
    unit = (match.group(2) or default_unit).lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "s":
        return value
    if unit == "m":
        return value * 60.0
    raise ValueError(f"Unsupported duration unit '{unit}'.")


def now_ms() -> int:
    return int(time.time() * 1000)


@contextmanager
def graceful_shutdown(callback: Callable[[], None]) -> Generator[None, None, None]:
    """Invoke callback when SIGINT/SIGTERM occurs; useful for async loops."""
    signals = (signal.SIGINT, signal.SIGTERM)
    previous: Dict[int, Callable] = {}

    def handler(signum, _frame):
        callback()

    try:
        for sig in signals:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        yield
    finally:
        for sig, old_handler in previous.items():
            signal.signal(sig, old_handler)
