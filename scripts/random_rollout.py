"""Phase 2 demo：用隨機策略跑一整個 episode，印出 TTFT / 成本 / 丟棄摘要。

用法：
    uv run python scripts/random_rollout.py [seed]

隨機亂選節點是最弱的基準——之後 PPO 要明顯贏過它。
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.sim.env import RouterEnv  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = RouterEnv()
    obs, _ = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "初始 obs 不在觀測空間內！"

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    ep = info["episode"]
    w_lat, w_cost = ep["w"]
    print(f"seed={seed}  本局權重 w=(延遲 {w_lat:.2f}, 成本 {w_cost:.2f})")
    print(f"步數（請求數）：{steps}")
    print(f"分流分布：{ep['by_node']}")
    print(f"丟棄：{ep['n_dropped']} / {ep['n_requests']}  (drop rate {ep['drop_rate']:.1%})")
    print(f"平均 TTFT（未丟棄）：{ep['avg_ttft_ms_served']:.1f} ms")
    print(f"總成本：{ep['total_cost']:.4f}")
    print(f"總 reward：{ep['total_reward']:.2f}（越接近 0 越好）")


if __name__ == "__main__":
    main()
