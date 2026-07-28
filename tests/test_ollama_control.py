"""OllamaControl 測試。真呼叫在 Ollama 未啟動時自動略過；fallback 路徑不需 Ollama。"""

import pytest

from edge_llm_router.backends.ollama_backend import ollama_available
from edge_llm_router.config import load_config
from edge_llm_router.control.base import ControlLLM
from edge_llm_router.control.ollama_control import OllamaControl, build_control
from edge_llm_router.control.rule_based import RuleBasedControl


def test_is_control_llm() -> None:
    assert isinstance(OllamaControl(), ControlLLM)


def test_dead_url_falls_back_to_rule() -> None:
    # 指向死 port → parse 退回規則版（仍給合理 w），不崩。
    c = OllamaControl(base_url="http://localhost:1", timeout=1.0)
    r = c.parse("成本太高，多用本機", (0.5, 0.5))
    assert r.w[1] > r.w[0]  # 規則版判成偏成本
    assert r.w[0] + r.w[1] == pytest.approx(1.0)


def test_dead_url_gibberish_keeps_current() -> None:
    c = OllamaControl(base_url="http://localhost:1", timeout=1.0)
    assert c.parse("asdf 隨便", (0.3, 0.7)).w == (0.3, 0.7)


def test_build_control_rule_provider() -> None:
    cfg = load_config()
    cfg["control"] = {"provider": "rule"}
    assert isinstance(build_control(cfg), RuleBasedControl)


def test_build_control_returns_control_llm() -> None:
    cfg = load_config()
    cfg["control"] = {"provider": "ollama"}
    assert isinstance(build_control(cfg), ControlLLM)  # 在線→Ollama、否則規則版


@pytest.mark.skipif(not ollama_available(), reason="Ollama 未啟動")
def test_real_ollama_parse_is_valid() -> None:
    r = OllamaControl().parse("成本太高，多用本機", (0.5, 0.5))
    assert r.intent in ("cheaper", "faster", "balanced")
    assert r.w[0] + r.w[1] == pytest.approx(1.0)
