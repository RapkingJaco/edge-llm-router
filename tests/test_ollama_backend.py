"""OllamaBackend 測試。真打測試在 Ollama 未啟動時自動略過（CI 友善）。"""

import pytest

from edge_llm_router.backends.base import NodeBackend, Request
from edge_llm_router.backends.ollama_backend import (
    OllamaBackend,
    build_edge_backend,
    ollama_available,
)
from edge_llm_router.backends.simulated import SimulatedBackend
from edge_llm_router.config import load_config


def test_fallback_to_simulated_when_unavailable() -> None:
    # 指向死 port → 一定 fallback 回模擬。
    node = build_edge_backend(load_config(), base_url="http://localhost:1")
    assert isinstance(node, SimulatedBackend)
    assert node.name == "edge"


def test_ollama_backend_is_a_node_backend() -> None:
    assert isinstance(OllamaBackend(), NodeBackend)


@pytest.mark.skipif(not ollama_available(), reason="Ollama 未啟動")
def test_real_ollama_measures_ttft() -> None:
    backend = OllamaBackend(name="edge", cost_per_request=0.001)
    res = backend.infer(Request(input_tokens=32, output_tokens=16))
    assert res.is_measured
    assert not res.dropped
    assert res.ttft_ms > 0.0
    assert res.cost == 0.001


@pytest.mark.skipif(not ollama_available(), reason="Ollama 未啟動")
def test_build_edge_uses_ollama_when_available() -> None:
    node = build_edge_backend(load_config())
    assert isinstance(node, OllamaBackend)
