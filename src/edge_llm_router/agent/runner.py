"""跑一個 policy 走完一個 episode，回傳環境的 episode 摘要。

給 baselines / PPO / eval / 測試共用。同一個 ``seed`` → 同一份工作負載與同一組 w，
比較才公平。
"""

from __future__ import annotations

from typing import Any

from ..sim.env import RouterEnv
from .policy import Policy


def run_episode(env: RouterEnv, policy: Policy, *, seed: int | None = None) -> dict[str, Any]:
    """用 ``policy`` 在 ``env`` 跑一整局，回傳 ``info["episode"]`` 摘要 dict。"""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        obs, _reward, terminated, truncated, info = env.step(policy.predict(obs))
    return info["episode"]
