"""PPO（改寫自 CleanRL 單檔版）訓練分流 policy。

單檔、無層層抽象，方便面試時直接指程式講 advantage / GAE / clip。權重 w 已在 observation
裡（見 ``sim/env.py``），所以本體幾乎是標準 PPO：
- 固定 w 訓練（P3-3）＝ 一般 PPO。
- 每局隨機 w 訓練（P3-4）＝ 權重條件化——程式相同，只差 env 的 ``fixed_w=None``。

自行管理數個環境 + 立即重置（不依賴 gym.vector 的 autoreset），GAE 的 done 記帳可控、
跨 gymnasium 版本穩定。訓練走 CPU（小 MLP 足夠）。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ..config import load_config
from ..sim.env import OBS_DIM, RouterEnv
from .policy import Policy

_CKPT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """Actor-critic MLP（兩個獨立小網路）。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPOPolicy(Policy):
    """把訓好的 agent 包成 Policy（預設取 argmax 動作）。"""

    name = "ppo"

    def __init__(self, agent: PPOAgent, deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = deterministic

    def predict(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.agent.actor(x)
            if self.deterministic:
                return int(logits.argmax(dim=-1).item())
            return int(Categorical(logits=logits).sample().item())


def save_agent(agent: PPOAgent, name: str = "ppo.pt") -> Path:
    _CKPT_DIR.mkdir(exist_ok=True, parents=True)
    path = _CKPT_DIR / name
    torch.save(agent.state_dict(), path)
    return path


def load_agent(name: str = "ppo.pt", obs_dim: int = OBS_DIM, n_actions: int = 3) -> PPOAgent:
    agent = PPOAgent(obs_dim, n_actions)
    agent.load_state_dict(torch.load(_CKPT_DIR / name))
    agent.eval()
    return agent


def train(
    *,
    config: dict[str, Any] | None = None,
    total_timesteps: int | None = None,
    fixed_w: tuple[float, float] | None = (0.5, 0.5),
    seed: int = 1,
    num_envs: int = 4,
    num_steps: int = 256,
    update_epochs: int = 4,
    num_minibatches: int = 4,
    use_mlflow: bool = True,
    run_name: str = "ppo-fixed-w",
) -> PPOAgent:
    """訓練並回傳 PPOAgent。``fixed_w=None`` → 每局隨機 w（權重條件化）。"""
    config = config if config is not None else load_config()
    ppo_cfg = config.get("ppo", {})
    total_timesteps = total_timesteps or int(ppo_cfg.get("total_timesteps", 500000))
    lr = float(ppo_cfg.get("learning_rate", 3e-4))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_coef = float(ppo_cfg.get("clip_coef", 0.2))
    ent_coef, vf_coef, max_grad_norm = 0.01, 0.5, 0.5

    torch.manual_seed(seed)
    device = torch.device("cpu")

    envs = [RouterEnv(config=config, fixed_w=fixed_w) for _ in range(num_envs)]
    agent = PPOAgent(OBS_DIM, 3).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = max(1, total_timesteps // batch_size)
    # MLflow 檔案後端每次 log 都 fsync（Windows 很慢）→ 粗粒度記錄約 40 個點即可畫曲線。
    log_every = max(1, num_updates // 40)

    obs = torch.zeros((num_steps, num_envs, OBS_DIM))
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
    logprobs = torch.zeros((num_steps, num_envs))
    rewards = torch.zeros((num_steps, num_envs))
    dones = torch.zeros((num_steps, num_envs))
    values = torch.zeros((num_steps, num_envs))

    next_obs = torch.tensor(
        np.array([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]), dtype=torch.float32
    )
    next_done = torch.zeros(num_envs)
    running_ret = np.zeros(num_envs)
    ep_returns: list[float] = []

    ctx = _mlflow_run(use_mlflow, run_name, locals())
    with ctx:
        for update in range(1, num_updates + 1):
            for step in range(num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                step_obs, step_rew, step_done = [], [], []
                for i, e in enumerate(envs):
                    o, r, term, trunc, _ = e.step(int(action[i].item()))
                    running_ret[i] += r
                    done = term or trunc
                    if done:
                        ep_returns.append(running_ret[i])
                        running_ret[i] = 0.0
                        o, _ = e.reset()
                    step_obs.append(o)
                    step_rew.append(r)
                    step_done.append(float(done))
                rewards[step] = torch.tensor(step_rew, dtype=torch.float32)
                next_obs = torch.tensor(np.array(step_obs), dtype=torch.float32)
                next_done = torch.tensor(step_done, dtype=torch.float32)

            # GAE
            with torch.no_grad():
                next_value = agent.get_value(next_obs).flatten()
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0.0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape((-1, OBS_DIM))
            b_actions = actions.reshape(-1)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            idx = np.arange(batch_size)
            last_stats: dict[str, float] = {}
            for _ in range(update_epochs):
                np.random.shuffle(idx)
                for start in range(0, batch_size, minibatch_size):
                    mb = idx[start : start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb], b_actions[mb]
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()

                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.flatten()
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()
                    last_stats = {
                        "loss/policy": pg_loss.item(),
                        "loss/value": v_loss.item(),
                        "loss/entropy": entropy_loss.item(),
                    }

            if use_mlflow and (update % log_every == 0 or update == num_updates):
                import mlflow

                gstep = update * batch_size
                metrics = dict(last_stats)
                if ep_returns:
                    metrics["charts/episodic_return"] = float(np.mean(ep_returns[-50:]))
                mlflow.log_metrics(metrics, step=gstep)
    return agent


def _mlflow_run(use_mlflow: bool, run_name: str, params: dict[str, Any]):
    if not use_mlflow:
        return contextlib.nullcontext()
    import mlflow

    mlflow.set_experiment("edge-llm-router-ppo")
    ctx = mlflow.start_run(run_name=run_name)
    keys = (
        "total_timesteps",
        "lr",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "num_envs",
        "num_steps",
        "fixed_w",
        "seed",
    )
    mlflow.log_params({k: params[k] for k in keys if k in params})
    return ctx
