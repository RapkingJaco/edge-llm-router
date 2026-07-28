"""訓練「開 Domain Randomization」的權重條件化 PPO → checkpoints/ppo_wc_dr.pt。

每局在真值±範圍隨機化節點參數，讓 policy 對「真實與模擬有落差」更穩。
之後用 eval/robustness.py 跟沒開 DR 的 ppo_wc.pt 比穩健度。

用法：
    uv run python scripts/train_dr.py [total_timesteps]
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.agent.ppo import save_agent, train  # noqa: E402
from edge_llm_router.config import load_config  # noqa: E402


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    cfg = load_config()
    cfg["domain_randomization"]["enabled"] = True
    print(f"訓練 PPO（開 DR，每局隨機化節點參數，total_timesteps={steps}）…")
    agent = train(config=cfg, total_timesteps=steps, fixed_w=None, run_name="ppo-dr")
    path = save_agent(agent, "ppo_wc_dr.pt")
    print(f"已存 checkpoint：{path}")


if __name__ == "__main__":
    main()
