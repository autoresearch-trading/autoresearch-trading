from __future__ import annotations

from typing import Iterable, List

import numpy as np
from signals.base import Signal, SignalDirection, SignalType


class SignalAggregator:
    """Combine multiple indicator outputs into discrete trade decisions."""

    @staticmethod
    def should_enter_long(
        signals: List[Signal],
        *,
        min_confidence: float,
        min_signals: int,
        require_cvd: bool,
        require_tfi: bool,
        require_ofi: bool,
    ) -> tuple[bool, float]:
        return _evaluate_signals(
            signals,
            direction=SignalDirection.BULLISH,
            min_confidence=min_confidence,
            min_signals=min_signals,
            require_cvd=require_cvd,
            require_tfi=require_tfi,
            require_ofi=require_ofi,
        )

    @staticmethod
    def should_enter_short(
        signals: List[Signal],
        *,
        min_confidence: float,
        min_signals: int,
        require_cvd: bool,
        require_tfi: bool,
        require_ofi: bool,
    ) -> tuple[bool, float]:
        return _evaluate_signals(
            signals,
            direction=SignalDirection.BEARISH,
            min_confidence=min_confidence,
            min_signals=min_signals,
            require_cvd=require_cvd,
            require_tfi=require_tfi,
            require_ofi=require_ofi,
        )

    @staticmethod
    def latest_signal_value(
        signals: Iterable[Signal], signal_type: SignalType
    ) -> float | None:
        """Get the most recent value for a given signal type."""
        filtered = [s for s in signals if s.signal_type == signal_type]
        if not filtered:
            return None
        # Signals are expected to be chronological; fallback to last element.
        return filtered[-1].value


def _evaluate_signals(
    signals: List[Signal],
    *,
    direction: SignalDirection,
    min_confidence: float,
    min_signals: int,
    require_cvd: bool,
    require_tfi: bool,
    require_ofi: bool,
) -> tuple[bool, float]:
    directional = [s for s in signals if s.direction == direction]
    if len(directional) < min_signals:
        return False, 0.0

    if not _satisfies_requirements(
        directional,
        direction=direction,
        require_cvd=require_cvd,
        require_tfi=require_tfi,
        require_ofi=require_ofi,
    ):
        return False, 0.0

    high_conf = [s for s in directional if s.confidence >= min_confidence]
    if len(high_conf) < min_signals:
        return False, 0.0

    confidences = [s.confidence for s in high_conf]
    combined = np.prod(confidences) ** (1.0 / len(confidences))
    return True, float(combined)


def _satisfies_requirements(
    signals: Iterable[Signal],
    *,
    direction: SignalDirection,
    require_cvd: bool,
    require_tfi: bool,
    require_ofi: bool,
) -> bool:
    if require_cvd and not _has_signal(signals, SignalType.CVD, direction):
        return False
    if require_tfi and not _has_signal(signals, SignalType.TFI, direction):
        return False
    if require_ofi and not _has_signal(signals, SignalType.OFI, direction):
        return False
    return True


def _has_signal(
    signals: Iterable[Signal], signal_type: SignalType, direction: SignalDirection
) -> bool:
    return any(
        s.signal_type == signal_type and s.direction == direction for s in signals
    )
