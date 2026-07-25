"""指標計算：TTFT / 成本的正規化、reward 的標量化、AI vs 基準線領先%。

集中放這裡，讓 sim / agent / eval / server 共用同一套定義，避免各處重寫。
"""

from __future__ import annotations


def normalize(value: float, baseline: float) -> float:
    """以固定基準正規化並夾到 [0, 1]。

    用固定基準（而非動態統計）避免訓練時分布漂移、也讓數字可解釋。
    """
    if baseline <= 0:
        raise ValueError("baseline 必須為正數")
    return max(0.0, min(1.0, value / baseline))


def scalarize_reward(
    ttft_ms: float,
    cost: float,
    w_lat: float,
    w_cost: float,
    ttft_baseline_ms: float,
    cost_baseline: float,
    penalty: float = 0.0,
) -> float:
    """把（延遲, 成本）依權重 w 合成單一 reward（越大越好）。

    r = -(w_lat * normTTFT + w_cost * normCost) - penalty
    """
    norm_ttft = normalize(ttft_ms, ttft_baseline_ms)
    norm_cost = normalize(cost, cost_baseline)
    return -(w_lat * norm_ttft + w_cost * norm_cost) - penalty


def lead_pct(ai_value: float, baseline_value: float) -> float:
    """AI 相對基準線的領先百分比（值越小越好的指標，如 TTFT/成本）。

    正數代表 AI 較低（較好）。baseline 為 0 時回 0 避免除零。
    """
    if baseline_value == 0:
        return 0.0
    return (baseline_value - ai_value) / baseline_value * 100.0
