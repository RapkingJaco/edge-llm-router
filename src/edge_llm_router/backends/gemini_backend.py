"""GeminiBackend：真打 Gemini 免費層（雲端節點），量真實延遲。

可插拔、防禦性：沒有 `GEMINI_API_KEY` 或沒裝 SDK → `build_cloud_backend` 自動 fallback 回
`SimulatedBackend`（和 Ollama 同套路）。成本一律「自己標價」（config 的 cloud 成本），不是
Gemini 真花費（免費層 0 元）。

要啟用真 Gemini：
1. `uv add google-genai`
2. 在 `.env` 放 `GEMINI_API_KEY=...`
本檔用 lazy import，SDK 缺席時完全不影響其他功能。
"""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

from .base import InferResult, NodeBackend, NodeState, Request
from .simulated import SimulatedBackend

_DEFAULT_MODEL = "gemini-flash-latest"


def gemini_available() -> bool:
    """有 key 且裝了 `google-genai` SDK 才算可用。"""
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    return importlib.util.find_spec("google.genai") is not None


class GeminiBackend(NodeBackend):
    """真打 Gemini，量首字延遲（含網路 RTT）。"""

    def __init__(
        self,
        name: str = "cloud",
        model: str = _DEFAULT_MODEL,
        cost_per_request: float = 0.01,
    ) -> None:
        self.name = name
        self.model = model
        self.cost_per_request = cost_per_request

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        prompt = req.prompt_text or "請用一句話回答：什麼是邊緣運算？"
        try:
            from google import genai  # lazy：SDK 缺席也不影響其他功能

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            t0 = time.perf_counter()
            stream = client.models.generate_content_stream(model=self.model, contents=prompt)
            for _chunk in stream:  # 第一個 chunk = 首字
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                return InferResult(
                    node=self.name, ttft_ms=ttft_ms, cost=self.cost_per_request, is_measured=True
                )
        except Exception:  # noqa: BLE001 — 真打失敗當丟棄，呼叫端可 fallback
            return InferResult(node=self.name, ttft_ms=0.0, cost=0.0, dropped=True)
        return InferResult(node=self.name, ttft_ms=0.0, cost=0.0, dropped=True)

    def state(self, now: float = 0.0) -> NodeState:
        return NodeState(utilization=0.0, queue_len=0, est_wait_ms=0.0)

    def reset(self) -> None:
        pass


def _simulated_cloud(config: dict[str, Any]) -> SimulatedBackend:
    cloud = config["nodes"]["cloud"]
    return SimulatedBackend(
        name="cloud",
        rtt_ms=cloud["rtt_ms"],
        prefill_ms_per_token=cloud["prefill_ms_per_token"],
        decode_ms_per_token=cloud["decode_ms_per_token"],
        capacity=cloud["capacity"],
        cost_per_request=cloud["cost_per_request"],
        drop_wait_ms=config.get("sim", {}).get("drop_wait_ms", 4000.0),
    )


def build_cloud_backend(config: dict[str, Any]) -> NodeBackend:
    """雲端節點降級鏈：真 Gemini →（額度沒了）本機 Ollama →（再不行）模擬。

    有 key 才試 Gemini、Ollama 在線才加進鏈；只有模擬時直接回 SimulatedBackend。
    """
    cost = config["nodes"]["cloud"]["cost_per_request"]
    chain: list[NodeBackend] = []
    if gemini_available():
        chain.append(GeminiBackend(name="cloud", cost_per_request=cost))

    from .ollama_backend import OllamaBackend, ollama_available

    if ollama_available():
        chain.append(OllamaBackend(name="cloud", cost_per_request=cost))
    chain.append(_simulated_cloud(config))

    if len(chain) == 1:
        return chain[0]
    from .fallback import FallbackBackend

    return FallbackBackend(chain, name="cloud")
