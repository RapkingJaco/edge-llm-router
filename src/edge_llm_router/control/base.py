"""控制層介面：把「中文方針」翻成 reward 權重 w。

**只吐意圖，不讓 LLM 直接吐數字**：LLM/規則判斷 `intent`（cheaper/faster/balanced）與
`magnitude`（強度），再由確定性程式 `intent_to_weights` 換算成 w。這層是「絕不盲信 LLM」
的安全網——LLM 只做它擅長的語意判斷，數字由我們控制、可驗證、可夾範圍。

可插拔：`RuleBasedControl`（離線、免 key）/ `GeminiControl`（真 LLM，之後接）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

WeightPair = tuple[float, float]  # (w_lat, w_cost)，和為 1


@dataclass(slots=True)
class ControlResult:
    """一次方針解讀的結果。"""

    w: WeightPair
    note: str  # 給使用者看的一句話解讀
    intent: str  # cheaper | faster | balanced | unknown
    magnitude: float  # 0~1


def intent_to_weights(intent: str, magnitude: float) -> WeightPair:
    """意圖 + 強度 → (w_lat, w_cost)，和為 1、皆夾在 [0,1]。"""
    m = max(0.0, min(1.0, magnitude))
    if intent == "faster":
        w_lat = 0.5 + 0.5 * m
    elif intent == "cheaper":
        w_lat = 0.5 - 0.5 * m
    else:  # balanced / unknown
        w_lat = 0.5
    w_lat = max(0.0, min(1.0, w_lat))
    return (round(w_lat, 3), round(1.0 - w_lat, 3))


def normalize_weights(w: WeightPair) -> WeightPair:
    """夾非負並正規化到和為 1；全零退回平衡。"""
    a, b = max(0.0, w[0]), max(0.0, w[1])
    total = a + b
    if total <= 0:
        return (0.5, 0.5)
    return (round(a / total, 3), round(b / total, 3))


class ControlLLM(ABC):
    """方針→權重的抽象。實作者保證 `parse` 一定回合法（已夾範圍）的 w。"""

    name: str = "control"

    @abstractmethod
    def parse(self, text: str, current_w: WeightPair) -> ControlResult:
        """把中文方針 `text` 轉成新的 w；看不懂就維持 `current_w`（安全網）。"""
