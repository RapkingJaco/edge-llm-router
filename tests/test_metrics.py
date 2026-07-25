"""指標函式測試。"""

import pytest

from edge_llm_router.metrics import lead_pct, normalize, scalarize_reward


def test_normalize_clips_to_unit_range() -> None:
    assert normalize(1000, 2000) == 0.5
    assert normalize(5000, 2000) == 1.0  # 超過基準夾到 1
    assert normalize(0, 2000) == 0.0


def test_normalize_rejects_bad_baseline() -> None:
    with pytest.raises(ValueError):
        normalize(1.0, 0.0)


def test_scalarize_reward_is_negative_cost() -> None:
    # 只在乎延遲：成本權重 0，reward 只反映正規化 TTFT。
    r = scalarize_reward(
        ttft_ms=1000, cost=999, w_lat=1.0, w_cost=0.0,
        ttft_baseline_ms=2000, cost_baseline=0.01,
    )
    assert r == pytest.approx(-0.5)


def test_lead_pct_positive_when_ai_lower() -> None:
    assert lead_pct(ai_value=80, baseline_value=100) == pytest.approx(20.0)
    assert lead_pct(ai_value=100, baseline_value=0) == 0.0
