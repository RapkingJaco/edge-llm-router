"""基準線策略：PPO 要打敗的四條傳統做法。

- Greedy：挑預估最快的節點（最小 est_wait，平手偏好本身較快的 cloud>edge>local）。
- RoundRobin：輪流本機→邊緣→雲。
- AllCloud / AllLocal：永遠丟同一個節點。

都只讀 observation（和 PPO 同樣的輸入），比較才公平。
"""

from __future__ import annotations

import numpy as np

from ..sim.env import NODE_ORDER, node_obs_index
from .policy import Policy


class GreedyPolicy(Policy):
    """最小預估等待；平手時偏好本身較快的節點（cloud > edge > local）。"""

    name = "greedy"
    # 平手偏好：加一個微小 bias，讓較快的節點勝出（local 最大、cloud 為 0）。
    _tie_bias = (0.002, 0.001, 0.0)

    def predict(self, obs: np.ndarray) -> int:
        scored = [
            float(obs[node_obs_index(k)[2]]) + self._tie_bias[k]
            for k in range(len(NODE_ORDER))
        ]
        return int(np.argmin(scored))


class RoundRobinPolicy(Policy):
    """輪流：本機 → 邊緣 → 雲 → …"""

    name = "round_robin"

    def __init__(self) -> None:
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def predict(self, obs: np.ndarray) -> int:
        action = self._i % len(NODE_ORDER)
        self._i += 1
        return action


class AllCloudPolicy(Policy):
    """永遠丟雲：穩但貴。"""

    name = "all_cloud"

    def predict(self, obs: np.ndarray) -> int:
        return NODE_ORDER.index("cloud")


class AllLocalPolicy(Policy):
    """永遠丟本機：便宜但一擠就爆。"""

    name = "all_local"

    def predict(self, obs: np.ndarray) -> int:
        return NODE_ORDER.index("local")


def all_baselines() -> list[Policy]:
    """回傳全部四條基準線（各自全新實例）。"""
    return [GreedyPolicy(), RoundRobinPolicy(), AllCloudPolicy(), AllLocalPolicy()]
