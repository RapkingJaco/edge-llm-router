"""GeminiBackend 測試。沒 key/SDK 時走 fallback（本專案預設狀態）。"""

import pytest

from edge_llm_router.backends.base import NodeBackend
from edge_llm_router.backends.gemini_backend import (
    GeminiBackend,
    build_cloud_backend,
    gemini_available,
)
from edge_llm_router.backends.simulated import SimulatedBackend
from edge_llm_router.config import load_config


def test_gemini_backend_is_a_node_backend() -> None:
    assert isinstance(GeminiBackend(), NodeBackend)


def test_fallback_to_simulated_without_key_or_sdk() -> None:
    node = build_cloud_backend(load_config())
    if gemini_available():
        pytest.skip("此環境有 Gemini key+SDK，跳過 fallback 測試")
    assert isinstance(node, SimulatedBackend)
    assert node.name == "cloud"
