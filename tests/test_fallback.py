"""FallbackBackend 測試（用 stub 後端，不碰網路）。"""

import pytest

from edge_llm_router.backends.base import InferResult, NodeBackend, NodeState, Request
from edge_llm_router.backends.fallback import FallbackBackend

REQ = Request(input_tokens=16, output_tokens=8)


class _Drop(NodeBackend):
    name = "drop"

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        return InferResult(node=self.name, ttft_ms=1.0, cost=0.0, dropped=True)

    def state(self, now: float = 0.0) -> NodeState:
        return NodeState(0.0, 0, 0.0)

    def reset(self) -> None:
        self.was_reset = True


class _Ok(NodeBackend):
    name = "ok"

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        return InferResult(node=self.name, ttft_ms=5.0, cost=0.5, is_measured=True)

    def state(self, now: float = 0.0) -> NodeState:
        return NodeState(0.2, 1, 3.0)

    def reset(self) -> None:
        self.was_reset = True


def test_uses_first_success() -> None:
    fb = FallbackBackend([_Drop(), _Ok()], name="cloud")
    res = fb.infer(REQ)
    assert not res.dropped
    assert res.ttft_ms == 5.0 and res.cost == 0.5
    assert res.node == "cloud"  # 名字統一


def test_all_drop_returns_dropped() -> None:
    res = FallbackBackend([_Drop(), _Drop()]).infer(REQ)
    assert res.dropped


def test_reset_propagates() -> None:
    a, b = _Drop(), _Ok()
    FallbackBackend([a, b]).reset()
    assert a.was_reset and b.was_reset


def test_state_uses_last_backend() -> None:
    fb = FallbackBackend([_Ok(), _Drop()])
    assert fb.state().utilization == 0.0  # 最後一個 (_Drop) 的 state


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        FallbackBackend([])
