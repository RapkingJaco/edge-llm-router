"""即時模擬：兩條環境並行跑同一請求流，一條用 PPO（AI）、一條用基準線（對照）。

逐 tick 推進一個請求，累積雙方指標，產生給前端渲染的快照。同 seed → 兩條看到完全相同
的到達序列，比較才公平。episode 跑完自動換下一局（連續 demo）。
"""

from __future__ import annotations

import copy
from typing import Any

from ..agent.baselines import GreedyPolicy
from ..agent.policy import Policy
from ..agent.ppo import PPOPolicy, load_agent
from ..backends.base import NodeBackend, Request
from ..backends.gemini_backend import build_cloud_backend
from ..backends.ollama_backend import build_edge_backend
from ..config import load_config
from ..control.base import ControlLLM
from ..control.rule_based import RuleBasedControl
from ..sim.env import RouterEnv


def load_ai_policy() -> tuple[Policy, bool]:
    """載入訓好的權重條件化 PPO；載不到就退回 greedy（回傳是否成功載入）。"""
    try:
        return PPOPolicy(load_agent("ppo_wc.pt")), True
    except Exception:  # noqa: BLE001 — 載不到 checkpoint 就優雅退回基準線
        return GreedyPolicy(), False


def _new_lane() -> dict[str, float]:
    return {"cum_reward": 0.0, "cost": 0.0, "served": 0.0, "drops": 0.0, "sum_ttft": 0.0}


class LiveSimulation:
    """驅動 AI vs 基準線的即時對照。"""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        w: tuple[float, float] = (0.5, 0.5),
        base_policy: Policy | None = None,
        control: ControlLLM | None = None,
        seed: int = 0,
        peak_factor: float = 2.5,
    ) -> None:
        self._base_config = config if config is not None else load_config()
        self.w = w
        self.ai_policy, self.ai_loaded = load_ai_policy()
        self.base_policy = base_policy if base_policy is not None else GreedyPolicy()
        self.control = control if control is not None else RuleBasedControl()
        self._note = ""
        # 抽驗：真實後端（lazy build，第一次抽驗才探測）+ 最近一次實測結果
        self._real: dict[str, NodeBackend] = {}
        self._last_sample: dict[str, Any] | None = None
        self._peak_factor = peak_factor
        self._peak = False
        self._seed = seed

        self.ai_env = RouterEnv(config=self._effective_config(), fixed_w=w)
        self.base_env = RouterEnv(config=self._effective_config(), fixed_w=w)
        self._new_episode(seed)

    # ── 對外 ────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> None:
        self._new_episode(seed if seed is not None else self._seed + 1)

    def set_peak(self, on: bool) -> None:
        """尖峰模式：拉高到達率製造壅塞，立即換一局生效。"""
        self._peak = on
        self._new_episode(self._seed)

    def set_policy(self, text: str) -> str:
        """中文方針 → 權重，即時套用（不重訓、不換局）。回傳解讀字串。"""
        result = self.control.parse(text, self.w)
        self.w = result.w
        self.ai_env.set_w(*result.w)
        self.base_env.set_w(*result.w)  # 基準線無視 w，套了也不影響
        self._note = result.note
        return result.note

    def _real_backend(self, node_name: str) -> NodeBackend:
        """lazy 建真實後端：edge→Ollama、cloud→Gemini(降級鏈)；建一次快取。"""
        if node_name not in self._real:
            builder = build_edge_backend if node_name == "edge" else build_cloud_backend
            self._real[node_name] = builder(self._base_config)
        return self._real[node_name]

    def real_sample(self, node_name: str = "edge") -> dict[str, Any]:
        """對真實後端抽打一次、量真實 TTFT（會阻塞數秒，應在背景執行緒跑）。"""
        backend = self._real_backend(node_name)
        res = backend.infer(Request(input_tokens=64, output_tokens=16))
        self._last_sample = {
            "node": node_name,
            "ttft_ms": round(res.ttft_ms, 0),
            "is_measured": res.is_measured,  # True = 真打；False = 該節點退回模擬
            "dropped": res.dropped,
            "backend": type(backend).__name__,
            "t": self._t,
        }
        return self._last_sample

    def tick(self) -> dict[str, Any]:
        if self._done:
            self._new_episode(self._seed + 1)

        a_action = self.ai_policy.predict(self._ai_obs)
        b_action = self.base_policy.predict(self._base_obs)
        self._ai_obs, a_r, _, a_trunc, a_info = self.ai_env.step(a_action)
        self._base_obs, b_r, _, b_trunc, b_info = self.base_env.step(b_action)
        self._t += 1
        self._accumulate(self._ai, a_r, a_info)
        self._accumulate(self._base, b_r, b_info)
        self._done = a_trunc or b_trunc
        return self._snapshot(a_info, b_info)

    # ── 內部 ────────────────────────────────────────────────────────
    def _effective_config(self) -> dict[str, Any]:
        cfg = copy.deepcopy(self._base_config)
        if self._peak:
            cfg["workload"]["arrival_rate_base"] *= self._peak_factor
        return cfg

    def _new_episode(self, seed: int) -> None:
        self._seed = seed
        eff = self._effective_config()
        self.ai_env.config = eff
        self.base_env.config = eff
        self._ai_obs, _ = self.ai_env.reset(seed=seed)
        self._base_obs, _ = self.base_env.reset(seed=seed)
        self.ai_policy.reset()
        self.base_policy.reset()
        self._ai = _new_lane()
        self._base = _new_lane()
        self._t = 0
        self._done = False

    @staticmethod
    def _accumulate(lane: dict[str, float], reward: float, info: dict[str, Any]) -> None:
        lane["cum_reward"] += reward
        lane["cost"] += info["cost"]
        if info["dropped"]:
            lane["drops"] += 1.0
        else:
            lane["served"] += 1.0
            lane["sum_ttft"] += info["ttft_ms"]

    def _lane_view(self, lane: dict[str, float], name: str, info: dict[str, Any]) -> dict[str, Any]:
        served = lane["served"]
        return {
            "name": name,
            "node": info["node"],
            "dropped": info["dropped"],
            "cum_reward": round(lane["cum_reward"], 1),
            "cost": round(lane["cost"], 3),
            "served": int(served),
            "drops": int(lane["drops"]),
            "avg_ttft_ms": round(lane["sum_ttft"] / served, 0) if served else 0.0,
        }

    def _snapshot(self, a_info: dict[str, Any], b_info: dict[str, Any]) -> dict[str, Any]:
        ai_r, base_r = self._ai["cum_reward"], self._base["cum_reward"]
        lead = (ai_r - base_r) / abs(base_r) * 100.0 if abs(base_r) > 1e-6 else 0.0
        return {
            "t": self._t,
            "w": [round(self.w[0], 2), round(self.w[1], 2)],
            "peak": self._peak,
            "note": self._note,
            "measured": self._last_sample,
            "episode_over": self._done,
            "ai_loaded": self.ai_loaded,
            "lead_pct": round(lead, 1),
            "ai_utils": {k: round(v, 2) for k, v in self.ai_env.peek_utilizations().items()},
            "base_utils": {k: round(v, 2) for k, v in self.base_env.peek_utilizations().items()},
            "ai": self._lane_view(self._ai, "PPO", a_info),
            "base": self._lane_view(self._base, self.base_policy.name, b_info),
        }
