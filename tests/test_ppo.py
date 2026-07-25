"""PPO 訓練迴圈煙霧測試（極短訓練，只確認不崩、能預測）。"""

import numpy as np

from edge_llm_router.agent.ppo import PPOAgent, PPOPolicy, train
from edge_llm_router.agent.runner import run_episode
from edge_llm_router.sim.env import OBS_DIM, RouterEnv


def test_train_briefly_returns_agent() -> None:
    agent = train(
        total_timesteps=1024,
        num_envs=2,
        num_steps=256,
        num_minibatches=4,
        use_mlflow=False,
        seed=0,
    )
    assert isinstance(agent, PPOAgent)


def test_train_weight_conditioned_runs() -> None:
    # fixed_w=None → 每局隨機 w（權重條件化）路徑不崩。
    agent = train(
        total_timesteps=1024, fixed_w=None, num_envs=2, num_steps=256, use_mlflow=False, seed=0
    )
    assert isinstance(agent, PPOAgent)


def test_ppo_policy_predicts_and_runs_episode() -> None:
    agent = train(
        total_timesteps=1024, num_envs=2, num_steps=256, use_mlflow=False, seed=0
    )
    policy = PPOPolicy(agent)
    assert policy.predict(np.zeros(OBS_DIM, dtype=np.float32)) in (0, 1, 2)
    ep = run_episode(RouterEnv(fixed_w=(0.5, 0.5)), policy, seed=0)
    assert 0.0 <= ep["drop_rate"] <= 1.0
