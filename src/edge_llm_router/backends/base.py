"""節點後端介面：把一個推論請求變成 TTFT / 成本結果。

同一介面三種實作：
- ``SimulatedBackend``：排隊模型模擬（本 Phase）。
- ``OllamaBackend`` / ``GeminiBackend``：真打（Phase 6）。

時間單位一律**毫秒（ms）**。``infer`` / ``state`` 的 ``now`` 是模擬時鐘（ms）；
真實後端會忽略它、直接量測 wall-clock。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class Request:
    """一個推論請求。

    ``input_tokens`` 已知（有 prompt 就能 tokenize）；``output_tokens`` 是這次「真正會
    生成」的長度，供模擬器算服務時間。**agent 觀測不到 output_tokens 真值**，只拿得到
    帶雜訊的估計（見 ``sim/env.py``）——不得洩上帝視角。
    """

    input_tokens: int
    output_tokens: int
    prompt_text: str | None = None


@dataclass(slots=True)
class InferResult:
    """一次推論的結果。"""

    node: str
    ttft_ms: float
    cost: float
    dropped: bool = False
    is_measured: bool = False  # True = 真打後端實測；False = 模擬
    full_ms: float = 0.0  # 整段生成佔用節點的時間（prefill + decode），供結算完成時點


@dataclass(slots=True)
class NodeState:
    """節點當前負載快照（給 observation 用）。"""

    utilization: float  # busy_slots / capacity，範圍 [0, 1]
    queue_len: int
    est_wait_ms: float


class NodeBackend(ABC):
    """節點後端抽象。實作者需保證 ``infer`` / ``state`` / ``reset`` 三個方法。

    ``sim`` 只依賴這個介面，不知道後端是真是假——這讓模擬/真打可無縫替換。
    """

    name: str

    @abstractmethod
    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        """處理請求，回 TTFT / 成本。``now`` 為模擬時鐘（ms）。"""

    @abstractmethod
    def state(self, now: float = 0.0) -> NodeState:
        """回目前負載快照。"""

    @abstractmethod
    def reset(self) -> None:
        """清空佇列狀態（每個 episode 開始時呼叫）。"""
