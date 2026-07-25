"""Sim-to-Real 校準：量真實 Ollama TTFT → 擬合 sim 參數 → 算 gap（模擬 vs 真實差距）。

單請求（無排隊）的 TTFT ≈ rtt + prefill_ms_per_token × input_tokens，是一條線；量幾組
不同輸入長度做線性回歸，就能把 sim 的 rtt / prefill 斜率校準到真實硬體。

gap 用 Wasserstein 距離衡量「模擬預測的 TTFT 分布」與「實測分布」有多遠；校準後應大幅縮小
——這就是「模擬越來越像真實」的鐵證。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backends.base import Request
from .backends.ollama_backend import OllamaBackend


@dataclass(slots=True)
class Measurement:
    input_tokens: int
    ttft_ms: float


def measure_ollama(
    input_grid: tuple[int, ...] = (16, 64, 256, 512),
    repeats: int = 3,
    backend: OllamaBackend | None = None,
) -> list[Measurement]:
    """對不同輸入長度各量 repeats 次 TTFT（先跑一次暖機載入模型並丟棄）。"""
    backend = backend or OllamaBackend()
    backend.infer(Request(input_tokens=8, output_tokens=8))  # warm-up
    out: list[Measurement] = []
    for it in input_grid:
        for _ in range(repeats):
            res = backend.infer(Request(input_tokens=it, output_tokens=8))
            if not res.dropped:
                out.append(Measurement(input_tokens=it, ttft_ms=res.ttft_ms))
    return out


def fit_edge_params(ms: list[Measurement]) -> dict[str, float]:
    """線性回歸 TTFT ~ input_tokens → 擬合 rtt(截距) 與 prefill_ms_per_token(斜率)。"""
    xs = np.array([m.input_tokens for m in ms], dtype=float)
    ys = np.array([m.ttft_ms for m in ms], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "prefill_ms_per_token": float(max(0.0, slope)),
        "rtt_ms": float(max(0.0, intercept)),
    }


def predict_ttft(params: dict[str, float], input_tokens: int) -> float:
    """sim 對單請求 TTFT 的預測（無排隊）。"""
    return params["rtt_ms"] + params["prefill_ms_per_token"] * input_tokens


def gap_wasserstein(ms: list[Measurement], params: dict[str, float]) -> float:
    """實測 TTFT 分布 vs sim 預測分布 的 Wasserstein 距離（越小越像）。"""
    from scipy.stats import wasserstein_distance

    measured = [m.ttft_ms for m in ms]
    predicted = [predict_ttft(params, m.input_tokens) for m in ms]
    return float(wasserstein_distance(measured, predicted))


def calibrate(
    config: dict[str, Any], ms: list[Measurement]
) -> dict[str, Any]:
    """回傳校準前後的參數與 gap（前=config 現值，後=擬合值）。"""
    edge = config["nodes"]["edge"]
    before = {"rtt_ms": edge["rtt_ms"], "prefill_ms_per_token": edge["prefill_ms_per_token"]}
    after = fit_edge_params(ms)
    return {
        "before": before,
        "after": after,
        "gap_before": gap_wasserstein(ms, before),
        "gap_after": gap_wasserstein(ms, after),
        "n_measurements": len(ms),
    }
