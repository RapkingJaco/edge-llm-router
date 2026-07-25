"""P3-4 CLI：權重條件化訓練（每局隨機 w），跨 w 光譜比 PPO vs 基準線，並看 w 響應。

用法：
    uv run python scripts/train_wc.py [total_timesteps]
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.agent.baselines import all_baselines  # noqa: E402
from edge_llm_router.agent.ppo import PPOPolicy, save_agent, train  # noqa: E402
from edge_llm_router.agent.runner import run_episode  # noqa: E402
from edge_llm_router.evaluation import DEFAULT_W_COSTS, evaluate, save, to_console  # noqa: E402
from edge_llm_router.sim.env import RouterEnv  # noqa: E402


def _routing_distribution(policy: PPOPolicy, w_cost: float, seed: int = 0) -> dict[str, str]:
    env = RouterEnv(fixed_w=(round(1 - w_cost, 3), w_cost))
    ep = run_episode(env, policy, seed=seed)
    total = sum(ep["by_node"].values()) or 1
    return {name: f"{cnt / total:.0%}" for name, cnt in ep["by_node"].items()}


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    print(f"權重條件化訓練 PPO（每局隨機 w，total_timesteps={steps}）…")
    agent = train(total_timesteps=steps, fixed_w=None, run_name="ppo-weight-conditioned")
    path = save_agent(agent, "ppo_wc.pt")
    print(f"已存 checkpoint：{path}\n")

    ppo = PPOPolicy(agent)
    results = evaluate([ppo, *all_baselines()], w_costs=DEFAULT_W_COSTS)
    save(results, DEFAULT_W_COSTS, (0, 1, 2, 3, 4), stem="eval_ppo_vs_baselines")
    print(to_console(results, DEFAULT_W_COSTS))

    print("\n=== PPO 的 w 響應（單局分流分布）===")
    for wc in (0.1, 0.9):
        dist = _routing_distribution(ppo, wc)
        tag = "要快" if wc < 0.5 else "要省"
        print(f"w_cost={wc}（{tag}）→ {dist}")


if __name__ == "__main__":
    main()
