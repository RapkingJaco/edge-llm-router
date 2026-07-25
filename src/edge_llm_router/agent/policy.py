"""策略介面：吃 observation、吐一個動作（0=本機 / 1=邊緣 / 2=雲）。

PPO 與所有基準線都實作這個介面，才能用同一套 runner / eval 公平比較。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """分流策略抽象。"""

    name: str = "policy"

    @abstractmethod
    def predict(self, obs: np.ndarray) -> int:
        """依 observation 選一個節點動作。"""

    def reset(self) -> None:  # noqa: B027 — 刻意的可選鉤子，預設 no-op（無狀態策略免實作）
        """episode 開始時呼叫（有內部狀態的策略覆寫，如 round-robin 計數器）。"""
