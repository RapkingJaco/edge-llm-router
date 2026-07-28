"""RouterEnv：逐請求分流的 Gymnasium 環境。

每一步（step）= 一個到達的請求，agent 選 本機/邊緣/雲（``Discrete(3)``）其中一個節點；
環境呼叫該節點後端算 TTFT/成本，依當前偏好權重 ``w`` 結算 reward。

**權重條件化**：每個 episode 開始隨機抽一組 ``w=(w_lat, w_cost)`` 並放進 observation，
讓同一個 policy 學會覆蓋整條「延遲 vs 成本」偏好光譜（見 docs/concepts.md）。

**不洩上帝視角**：observation 只放真實環境也拿得到的量（已知的輸入 token、帶雜訊的
輸出長度估計、觀測到的節點負載、當前 w）；不給模擬器內部真值/未來。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..backends.base import Request
from ..backends.simulated import build_nodes, build_nodes_randomized
from ..config import load_config
from ..metrics import scalarize_reward
from .workload import Arrival, generate_episode, generate_n

NODE_ORDER = ("local", "edge", "cloud")
OBS_DIM = 15
_TOKEN_NORM = 512.0  # token 正規化基準（與 config 的 token max 對齊）

# obs 版面（讀 obs 的策略共用，避免與 _obs() 順序脫鉤）：
#   [0,1] 請求 input/output(估計)；[2..10] 三節點各 (util, queue, est_wait)；
#   [11] 累積成本；[12] 全域負載；[13,14] w_lat, w_cost
W_LAT_INDEX = OBS_DIM - 2
W_COST_INDEX = OBS_DIM - 1


def node_obs_index(k: int) -> tuple[int, int, int]:
    """第 k 個節點（依 NODE_ORDER）在 obs 的 (使用率, 佇列, 預估等待) 索引。"""
    base = 2 + 3 * k
    return base, base + 1, base + 2


class RouterEnv(gym.Env):
    """LLM 推論分流環境。observation 15 維（皆正規化到 [0,1]），動作 Discrete(3)。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        fixed_w: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else load_config()
        self.nodes = build_nodes(self.config)
        self._dr_enabled = bool(self.config.get("domain_randomization", {}).get("enabled", False))
        self._fixed_w = fixed_w
        self._n_requests: int | None = None  # None=時長制 episode；設了=剛好跑這麼多筆

        reward_cfg = self.config["reward"]
        self.ttft_baseline = float(reward_cfg["ttft_baseline_ms"])
        self.cost_baseline = float(reward_cfg["cost_baseline"])
        self.timeout_penalty = float(reward_cfg["timeout_penalty"])
        self.token_est_noise = float(self.config["workload"].get("token_est_noise", 0.2))
        self._drop_wait_ms = float(self.config.get("sim", {}).get("drop_wait_ms", 4000.0))
        self._cloud_cost = float(self.config["nodes"]["cloud"]["cost_per_request"])
        self._total_capacity = sum(n.capacity for n in self.nodes.values())

        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # episode 狀態（reset 時初始化）
        self._arrivals: list[Arrival] = []
        self._i = 0
        self._w = (0.5, 0.5)
        self._cum_cost = 0.0
        self._cost_budget = 1.0
        self._stats: dict[str, Any] = {}
        self._last_now = 0.0

    # ── Gym API ─────────────────────────────────────────────────────
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._dr_enabled:
            # DR：每局重建節點、抽一組隨機參數；容量會變，重算 obs 正規化用的總容量。
            self.nodes = build_nodes_randomized(self.config, self.np_random)
            self._total_capacity = sum(n.capacity for n in self.nodes.values())
        else:
            for node in self.nodes.values():
                node.reset()
        self._w = self._sample_w()
        if self._n_requests is not None:
            self._arrivals = generate_n(self.np_random, self.config, self._n_requests)
        else:
            self._arrivals = generate_episode(self.np_random, self.config)
        self._i = 0
        self._last_now = 0.0
        self._cum_cost = 0.0
        self._cost_budget = max(self._cloud_cost * len(self._arrivals), 1e-9)
        self._stats = {
            "n": 0,
            "n_dropped": 0,
            "sum_ttft_served": 0.0,
            "sum_cost": 0.0,
            "sum_reward": 0.0,
            "by_node": {name: 0 for name in NODE_ORDER},
        }
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        arr = self._arrivals[self._i]
        self._last_now = arr.t_ms
        node_name = NODE_ORDER[int(action)]
        node = self.nodes[node_name]
        req = Request(input_tokens=arr.input_tokens, output_tokens=arr.output_tokens)
        res = node.infer(req, now=arr.t_ms)

        penalty = self.timeout_penalty if res.dropped else 0.0
        reward = scalarize_reward(
            ttft_ms=res.ttft_ms,
            cost=res.cost,
            w_lat=self._w[0],
            w_cost=self._w[1],
            ttft_baseline_ms=self.ttft_baseline,
            cost_baseline=self.cost_baseline,
            penalty=penalty,
        )

        self._cum_cost += res.cost
        self._accumulate(node_name, res, reward)

        self._i += 1
        truncated = self._i >= len(self._arrivals)
        terminated = False
        obs = self._obs() if not truncated else np.zeros(OBS_DIM, dtype=np.float32)

        info: dict[str, Any] = {
            "node": node_name,
            "ttft_ms": res.ttft_ms,
            "cost": res.cost,
            "dropped": res.dropped,
        }
        if truncated:
            info["episode"] = self._episode_summary()
        return obs, float(reward), terminated, truncated, info

    # ── 內部 ────────────────────────────────────────────────────────
    def _sample_w(self) -> tuple[float, float]:
        if self._fixed_w is not None:
            return self._fixed_w
        w_lat = float(self.np_random.uniform(0.0, 1.0))
        return (w_lat, 1.0 - w_lat)

    def _obs(self) -> np.ndarray:
        arr = self._arrivals[self._i]
        now = arr.t_ms

        # 這個請求：input 已知；output 只給帶雜訊的估計。
        noise = 1.0 + float(self.np_random.normal(0.0, self.token_est_noise))
        out_est = max(0.0, arr.output_tokens * noise)
        vec: list[float] = [
            arr.input_tokens / _TOKEN_NORM,
            out_est / _TOKEN_NORM,
        ]

        # 每個節點：使用率、佇列長度、預估等待。
        busy_total = 0
        for name in NODE_ORDER:
            node = self.nodes[name]
            st = node.state(now)
            busy_total += st.queue_len
            vec.append(st.utilization)
            vec.append(st.queue_len / node.capacity)
            vec.append(st.est_wait_ms / self._drop_wait_ms)

        # 全域：累積成本（占最壞情況比例）、當前負載程度。
        vec.append(self._cum_cost / self._cost_budget)
        vec.append(busy_total / self._total_capacity)

        # 方針權重。
        vec.append(self._w[0])
        vec.append(self._w[1])

        return np.clip(np.asarray(vec, dtype=np.float32), 0.0, 1.0)

    def set_w(self, w_lat: float, w_cost: float) -> None:
        """即時更新偏好權重（含當前 episode）。obs 下一步就反映新 w → PPO 零重訓改行為。"""
        self._fixed_w = (w_lat, w_cost)
        self._w = (w_lat, w_cost)

    def set_n_requests(self, n: int | None) -> None:
        """設定「跑固定筆數」；None 恢復時長制 episode。下次 reset 生效。"""
        self._n_requests = n

    def peek_utilizations(self) -> dict[str, float]:
        """三節點在最近處理時點的使用率（給即時儀表板用；不影響訓練成本）。"""
        return {
            name: self.nodes[name].state(self._last_now).utilization for name in NODE_ORDER
        }

    def _accumulate(self, node_name: str, res: Any, reward: float) -> None:
        s = self._stats
        s["n"] += 1
        s["sum_cost"] += res.cost
        s["sum_reward"] += reward
        s["by_node"][node_name] += 1
        if res.dropped:
            s["n_dropped"] += 1
        else:
            s["sum_ttft_served"] += res.ttft_ms

    def _episode_summary(self) -> dict[str, Any]:
        s = self._stats
        served = s["n"] - s["n_dropped"]
        return {
            "n_requests": s["n"],
            "n_dropped": s["n_dropped"],
            "drop_rate": s["n_dropped"] / s["n"] if s["n"] else 0.0,
            "avg_ttft_ms_served": s["sum_ttft_served"] / served if served else 0.0,
            "total_cost": s["sum_cost"],
            "total_reward": s["sum_reward"],
            "by_node": dict(s["by_node"]),
            "w": self._w,
        }
