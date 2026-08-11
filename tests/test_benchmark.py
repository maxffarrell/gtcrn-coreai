from __future__ import annotations

import pytest

from gtcrn_coreai.benchmark import summarize_latencies


def test_summarize_latencies() -> None:
    result = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["max_ms"] == pytest.approx(4.0)
    assert result["realtime_ratio_mean"] == pytest.approx(2.5 / 16.0)
