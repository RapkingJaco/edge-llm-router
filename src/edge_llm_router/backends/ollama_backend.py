"""OllamaBackend：真打本機 Ollama（4070 邊緣節點），量測真實 TTFT。

用途是**校準與抽驗**，不是主迴圈——真生成一次要幾百 ms 到數秒，訓練/即時 demo 仍走
模擬。介面與 `SimulatedBackend` 一致，所以 `sim` 不知道後端真假。

- `is_measured=True` 標記這是實測值。
- 成本一律「自己標價」（用 config 的 edge 成本），不是 Ollama 真花費（免費）。
- 偵測不到 Ollama → 用 `build_edge_backend` 自動 fallback 回模擬。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import InferResult, NodeBackend, NodeState, Request
from .simulated import SimulatedBackend

_DEFAULT_BASE_URL = "http://localhost:11434"


def ollama_available(base_url: str = _DEFAULT_BASE_URL, timeout: float = 2.0) -> bool:
    """Ollama 服務是否在線。"""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — 探測失敗即視為不可用
        return False


def _filler_prompt(input_tokens: int) -> str:
    """用重複詞湊出大約 input_tokens 長的 prompt（校準用，夠真實即可）。"""
    n = max(1, min(input_tokens, 2000))
    return "請簡短回答。 " + "資料 " * n


class OllamaBackend(NodeBackend):
    """真打本機 Ollama，量首字延遲（streaming 的第一個 chunk）。"""

    def __init__(
        self,
        name: str = "edge",
        model: str = "llama3.2",
        cost_per_request: float = 0.001,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.model = model
        self.cost_per_request = cost_per_request
        self.base_url = base_url
        self.timeout = timeout

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        prompt = req.prompt_text or _filler_prompt(req.input_tokens)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max(1, min(req.output_tokens, 64))},
        }
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client, client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:  # 第一個 streamed chunk = 首字
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                        return InferResult(
                            node=self.name,
                            ttft_ms=ttft_ms,
                            cost=self.cost_per_request,
                            is_measured=True,
                        )
        except Exception:  # noqa: BLE001 — 真打失敗當作丟棄，呼叫端可 fallback
            return InferResult(node=self.name, ttft_ms=0.0, cost=0.0, dropped=True)
        return InferResult(node=self.name, ttft_ms=0.0, cost=0.0, dropped=True)

    def state(self, now: float = 0.0) -> NodeState:
        # 真後端不在 RL obs 路徑上；回中性狀態即可。
        return NodeState(utilization=0.0, queue_len=0, est_wait_ms=0.0)

    def reset(self) -> None:
        pass


def build_edge_backend(config: dict[str, Any], base_url: str = _DEFAULT_BASE_URL) -> NodeBackend:
    """邊緣節點：Ollama 在線就真跑，否則自動 fallback 回 SimulatedBackend。"""
    edge_cfg = config["nodes"]["edge"]
    if ollama_available(base_url):
        return OllamaBackend(
            name="edge", cost_per_request=edge_cfg["cost_per_request"], base_url=base_url
        )
    drop_wait = config.get("sim", {}).get("drop_wait_ms", 4000.0)
    return SimulatedBackend(
        name="edge",
        rtt_ms=edge_cfg["rtt_ms"],
        prefill_ms_per_token=edge_cfg["prefill_ms_per_token"],
        decode_ms_per_token=edge_cfg["decode_ms_per_token"],
        capacity=edge_cfg["capacity"],
        cost_per_request=edge_cfg["cost_per_request"],
        drop_wait_ms=drop_wait,
    )
