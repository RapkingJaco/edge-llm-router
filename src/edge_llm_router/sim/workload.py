"""工作負載產生器：請求何時到達、每個請求多大。

- **到達流程**：非齊次卜瓦松（Poisson）過程——到達率隨時間變動，中段有一個高斯突波
  製造「尖峰」，逼 agent 處理壅塞。
- **token 分布**：短/中/長三段混合（短請求多、長請求少）。
- 一整個 episode 的工作負載在 ``reset()`` 時一次產生（給定 seed 即可重現）。

時間單位：毫秒（ms）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Arrival:
    """一個請求的到達：時間 + 真實 token 數。"""

    t_ms: float
    input_tokens: int
    output_tokens: int


def _rate_per_s(t_ms: float, duration_ms: float, base: float, peak_amp: float) -> float:
    """時間 t 的瞬時到達率（每秒）：base × (1 + 中段高斯突波)。"""
    center = duration_ms / 2.0
    width = duration_ms / 8.0
    bump = peak_amp * math.exp(-(((t_ms - center) / width) ** 2))
    return base * (1.0 + bump)


def _sample_tokens(rng: np.random.Generator, tok: dict[str, Any]) -> tuple[int, int]:
    """從短/中/長三段混合抽 (input, output) token 數。"""
    lo_i, hi_i = tok["input"]["min"], tok["input"]["max"]
    lo_o, hi_o = tok["output"]["min"], tok["output"]["max"]
    # 短請求最多、長請求最少。
    cat = int(rng.choice(3, p=[0.4, 0.4, 0.2]))
    band = [(0.0, 0.25), (0.25, 0.6), (0.6, 1.0)][cat]

    def draw(lo: int, hi: int) -> int:
        a = lo + (hi - lo) * band[0]
        b = lo + (hi - lo) * band[1]
        return int(rng.integers(int(a), int(b) + 1))

    return draw(lo_i, hi_i), draw(lo_o, hi_o)


def generate_episode(rng: np.random.Generator, config: dict[str, Any]) -> list[Arrival]:
    """產生一個 episode 的到達序列（依時間排序）。"""
    wl = config["workload"]
    base = float(wl["arrival_rate_base"])
    peak_amp = float(wl.get("peak_amplitude", 3.0))
    duration_ms = float(wl["episode_seconds"]) * 1000.0
    tok = wl["tokens"]

    # 卜瓦松細化法（thinning）：以峰值率提議、再依實際率接受。
    max_rate = base * (1.0 + peak_amp)
    arrivals: list[Arrival] = []
    t = 0.0
    while t < duration_ms:
        t += float(rng.exponential(1000.0 / max_rate))
        if t >= duration_ms:
            break
        if rng.random() <= _rate_per_s(t, duration_ms, base, peak_amp) / max_rate:
            it, ot = _sample_tokens(rng, tok)
            arrivals.append(Arrival(t_ms=t, input_tokens=it, output_tokens=ot))

    # 極端情況（例如超短 episode）保底至少一個請求。
    if not arrivals:
        it, ot = _sample_tokens(rng, tok)
        arrivals.append(Arrival(t_ms=0.0, input_tokens=it, output_tokens=ot))
    return arrivals
