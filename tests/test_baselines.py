"""基準線策略測試。"""

import numpy as np

from edge_llm_router.agent.baselines import (
    AllCloudPolicy,
    AllLocalPolicy,
    GreedyPolicy,
    RoundRobinPolicy,
    all_baselines,
)
from edge_llm_router.agent.runner import run_episode
from edge_llm_router.sim.env import OBS_DIM, RouterEnv, node_obs_index


def _zeros() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)


def test_all_actions_valid() -> None:
    obs = _zeros()
    for p in all_baselines():
        p.reset()
        assert p.predict(obs) in (0, 1, 2)


def test_round_robin_cycles() -> None:
    p = RoundRobinPolicy()
    p.reset()
    seq = [p.predict(_zeros()) for _ in range(7)]
    assert seq == [0, 1, 2, 0, 1, 2, 0]


def test_constant_policies() -> None:
    assert AllCloudPolicy().predict(_zeros()) == 2
    assert AllLocalPolicy().predict(_zeros()) == 0


def test_greedy_prefers_idle_fast_node() -> None:
    # 全空（est_wait 全 0）→ 平手偏好 cloud（最快）。
    assert GreedyPolicy().predict(_zeros()) == 2


def test_greedy_avoids_busy_node() -> None:
    obs = _zeros()
    obs[node_obs_index(2)[2]] = 0.9  # cloud 預估等待很高
    assert GreedyPolicy().predict(obs) != 2


def test_all_cloud_pricier_all_local_droppier() -> None:
    env = RouterEnv()
    ac = run_episode(env, AllCloudPolicy(), seed=42)
    al = run_episode(env, AllLocalPolicy(), seed=42)
    assert ac["total_cost"] > al["total_cost"]  # 全雲貴
    assert al["drop_rate"] > ac["drop_rate"]  # 全本機狂丟


def test_greedy_drops_less_than_all_local() -> None:
    env = RouterEnv()
    g = run_episode(env, GreedyPolicy(), seed=7)
    al = run_episode(env, AllLocalPolicy(), seed=7)
    assert g["drop_rate"] < al["drop_rate"]
