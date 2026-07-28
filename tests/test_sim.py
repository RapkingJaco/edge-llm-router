"""RouterEnv Gymnasium 環境測試。"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from edge_llm_router.config import load_config
from edge_llm_router.sim.env import OBS_DIM, RouterEnv
from edge_llm_router.sim.workload import generate_episode


def _dr_config() -> dict:
    cfg = load_config()
    cfg["domain_randomization"] = {
        "enabled": True, "prefill_jitter": 0.3, "rtt_jitter_ms": 20, "capacity_jitter": 1,
    }
    return cfg


def test_dr_off_is_stable() -> None:
    env = RouterEnv()  # DR 預設關
    env.reset(seed=1)
    p1, c1 = env.nodes["edge"].prefill_ms_per_token, env.nodes["edge"].capacity
    env.reset(seed=2)
    assert env.nodes["edge"].prefill_ms_per_token == p1
    assert env.nodes["edge"].capacity == c1


def test_dr_on_varies_params() -> None:
    env = RouterEnv(config=_dr_config())
    env.reset(seed=1)
    p1 = env.nodes["edge"].prefill_ms_per_token
    env.reset(seed=2)
    assert env.nodes["edge"].prefill_ms_per_token != p1  # 每局隨機化


def test_dr_env_still_passes_check() -> None:
    check_env(RouterEnv(config=_dr_config()), skip_render_check=True)


def test_passes_gym_env_checker() -> None:
    # Gymnasium 官方檢查：reset/step 簽名、空間、dtype 等。
    check_env(RouterEnv(), skip_render_check=True)


def test_reset_returns_obs_in_space() -> None:
    env = RouterEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(info, dict)


def test_full_episode_truncates() -> None:
    env = RouterEnv()
    env.reset(seed=1)
    truncated = False
    steps = 0
    while not truncated:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert not terminated  # 本環境只用 truncation 結束
        steps += 1
    assert steps > 0
    assert "episode" in info
    assert info["episode"]["n_requests"] == steps


def test_obs_always_in_unit_range() -> None:
    env = RouterEnv()
    env.reset(seed=2)
    truncated = False
    while not truncated:
        obs, _, _, truncated, _ = env.step(env.action_space.sample())
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_weight_conditioning_in_obs() -> None:
    # fixed_w 應原封不動出現在 obs 末兩維，且和為 1。
    env = RouterEnv(fixed_w=(0.8, 0.2))
    obs, _ = env.reset(seed=3)
    assert obs[-2] == np.float32(0.8)
    assert obs[-1] == np.float32(0.2)


def test_all_cloud_costs_more_than_all_local() -> None:
    # 同一份工作負載，全丟雲 vs 全丟本機：雲端總成本應高很多。
    env = RouterEnv()
    env.reset(seed=4)
    arrivals = list(env._arrivals)

    def run(action: int) -> float:
        env.reset(seed=4)
        env._arrivals = arrivals  # 固定同一份負載
        total = 0.0
        truncated = False
        while not truncated:
            _, _, _, truncated, info = env.step(action)
            total += info["cost"]
        return total

    cost_local = run(0)
    cost_cloud = run(2)
    assert cost_cloud > cost_local


def test_workload_is_reproducible() -> None:
    env = RouterEnv()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a1 = generate_episode(rng1, env.config)
    a2 = generate_episode(rng2, env.config)
    assert len(a1) == len(a2)
    assert a1[0].t_ms == a2[0].t_ms
