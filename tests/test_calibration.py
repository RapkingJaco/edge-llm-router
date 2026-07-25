"""校準引擎測試（用合成資料，不需 Ollama）。"""

import pytest

from edge_llm_router.calibration import (
    Measurement,
    calibrate,
    fit_edge_params,
    gap_wasserstein,
    predict_ttft,
)
from edge_llm_router.config import load_config


def _synthetic(rtt: float, slope: float) -> list[Measurement]:
    # 真實 = 一條 TTFT = rtt + slope × input 的線。
    return [Measurement(it, rtt + slope * it) for it in (16, 64, 256, 512)]


def test_fit_recovers_line() -> None:
    ms = _synthetic(rtt=20.0, slope=0.5)
    p = fit_edge_params(ms)
    assert p["rtt_ms"] == pytest.approx(20.0, abs=1e-3)
    assert p["prefill_ms_per_token"] == pytest.approx(0.5, abs=1e-4)


def test_gap_zero_when_params_match() -> None:
    ms = _synthetic(rtt=20.0, slope=0.5)
    gap = gap_wasserstein(ms, {"rtt_ms": 20.0, "prefill_ms_per_token": 0.5})
    assert gap == pytest.approx(0.0, abs=1e-6)


def test_predict_ttft() -> None:
    assert predict_ttft({"rtt_ms": 10.0, "prefill_ms_per_token": 2.0}, 100) == 210.0


def test_calibrate_shrinks_gap() -> None:
    # config 的 placeholder 參數與真實不同 → 校準後 gap 應變小。
    ms = _synthetic(rtt=200.0, slope=3.0)  # 與 config edge 起始值差很多
    result = calibrate(load_config(), ms)
    assert result["gap_after"] <= result["gap_before"]
