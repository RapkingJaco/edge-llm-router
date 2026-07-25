"""控制層（規則版）測試。"""

import pytest

from edge_llm_router.control.base import intent_to_weights, normalize_weights
from edge_llm_router.control.rule_based import RuleBasedControl

CTRL = RuleBasedControl()
BALANCED = (0.5, 0.5)


def test_cheaper_shifts_toward_cost() -> None:
    r = CTRL.parse("成本太高，多用本機", BALANCED)
    assert r.intent == "cheaper"
    assert r.w[1] > r.w[0]  # 成本權重較大


def test_faster_shifts_toward_latency() -> None:
    r = CTRL.parse("我要最快", BALANCED)
    assert r.intent == "faster"
    assert r.w[0] > r.w[1]  # 延遲權重較大


def test_balanced() -> None:
    r = CTRL.parse("平衡就好", BALANCED)
    assert r.intent == "balanced"
    assert r.w == pytest.approx((0.5, 0.5))


def test_gibberish_keeps_current_w() -> None:
    current = (0.3, 0.7)
    r = CTRL.parse("asdf 隨便亂打", current)
    assert r.intent == "unknown"
    assert r.w == current  # 安全網：維持原方針


def test_empty_keeps_current_w() -> None:
    current = (0.2, 0.8)
    assert CTRL.parse("   ", current).w == current


def test_weights_always_valid() -> None:
    for text in ["省錢", "要快", "成本太高多用本機", "平衡"]:
        w = CTRL.parse(text, BALANCED).w
        assert w[0] + w[1] == pytest.approx(1.0)
        assert 0.0 <= w[0] <= 1.0 and 0.0 <= w[1] <= 1.0


def test_strong_words_push_harder() -> None:
    mild = CTRL.parse("想省一點成本", BALANCED)
    strong = CTRL.parse("成本太貴了拜託省", BALANCED)
    assert strong.w[1] >= mild.w[1]  # 強語氣 → 成本權重更高


def test_helpers() -> None:
    assert intent_to_weights("faster", 1.0) == (1.0, 0.0)
    assert intent_to_weights("cheaper", 1.0) == (0.0, 1.0)
    assert normalize_weights((2.0, 2.0)) == (0.5, 0.5)
    assert normalize_weights((0.0, 0.0)) == (0.5, 0.5)
