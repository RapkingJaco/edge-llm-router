"""FallbackBackend：依序試多個後端，回第一個成功（未 dropped）的結果。

用途：雲端節點 Gemini 額度沒了/失敗時，自動退到本機 Ollama，再退到模擬——抽驗/探測
永遠有結果、不會卡在 Google 額度。所有結果的 ``node`` 統一成 wrapper 的名字。
"""

from __future__ import annotations

from .base import InferResult, NodeBackend, NodeState, Request


class FallbackBackend(NodeBackend):
    """後端鏈：依序 infer，用第一個沒被 dropped 的結果。"""

    def __init__(self, backends: list[NodeBackend], name: str | None = None) -> None:
        if not backends:
            raise ValueError("FallbackBackend 需要至少一個後端")
        self.backends = backends
        self.name = name or backends[0].name

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        last: InferResult | None = None
        for backend in self.backends:
            last = backend.infer(req, now=now)
            if not last.dropped:
                return InferResult(
                    node=self.name,
                    ttft_ms=last.ttft_ms,
                    cost=last.cost,
                    dropped=False,
                    is_measured=last.is_measured,
                    full_ms=last.full_ms,
                )
        return InferResult(
            node=self.name,
            ttft_ms=last.ttft_ms if last else 0.0,
            cost=0.0,
            dropped=True,
        )

    def state(self, now: float = 0.0) -> NodeState:
        # 用最終（模擬）後端的狀態，最穩定。
        return self.backends[-1].state(now)

    def reset(self) -> None:
        for backend in self.backends:
            backend.reset()
