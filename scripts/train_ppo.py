"""P3-3 CLI：固定 w=0.5/0.5 訓練 PPO，存 checkpoint，再和基準線比一次。

用法：
    uv run python scripts/train_ppo.py [total_timesteps]
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
from edge_llm_router.evaluation import evaluate, to_console  # noqa: E402


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    print(f"訓練 PPO（固定 w=0.5/0.5，total_timesteps={steps}）…")
    agent = train(total_timesteps=steps, fixed_w=(0.5, 0.5), run_name="ppo-fixed-w")
    path = save_agent(agent, "ppo_fixed_w.pt")
    print(f"已存 checkpoint：{path}")

    ppo = PPOPolicy(agent)
    results = evaluate([ppo, *all_baselines()], w_costs=(0.5,))
    print(to_console(results, (0.5,)))


if __name__ == "__main__":
    main()
