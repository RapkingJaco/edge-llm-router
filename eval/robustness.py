"""Sim-to-Real 穩健度：DR vs 非 DR 策略，丟到「參數被改過的世界」看誰比較穩。

模擬「真實與訓練模擬有落差」：把節點參數整體偏移（prefill ×1.4、rtt +40ms、容量 −1），
比較兩個策略從 base → shifted 的 reward 掉多少。DR 策略應掉得比較少。

用法（需先有 ppo_wc.pt 與 ppo_wc_dr.pt）：
    uv run python eval/robustness.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.agent.ppo import PPOPolicy, load_agent  # noqa: E402
from edge_llm_router.agent.runner import run_episode  # noqa: E402
from edge_llm_router.config import load_config  # noqa: E402
from edge_llm_router.sim.env import RouterEnv  # noqa: E402

_OUT = Path(__file__).resolve().parents[1] / "outputs"
_SEEDS = (0, 1, 2, 3, 4)

# 落差強度：(prefill 倍率, rtt 加多少 ms, 容量減多少)
_SHIFTS = {
    "mild": (1.2, 20.0, 0),
    "moderate": (1.5, 50.0, 1),
    "severe": (2.0, 100.0, 1),
}


def _shifted(base: dict, pf_mul: float, rtt_add: float, cap_sub: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["domain_randomization"]["enabled"] = False  # 用固定的偏移參數評估
    for nc in cfg["nodes"].values():
        nc["prefill_ms_per_token"] *= pf_mul
        nc["rtt_ms"] += rtt_add
        nc["capacity"] = max(1, nc["capacity"] - cap_sub)
    return cfg


def _mean_reward(policy: PPOPolicy, cfg: dict) -> float:
    rewards = [
        run_episode(RouterEnv(config=cfg, fixed_w=(0.5, 0.5)), policy, seed=s)["total_reward"]
        for s in _SEEDS
    ]
    return sum(rewards) / len(rewards)


def main() -> None:
    base = load_config()
    base["domain_randomization"]["enabled"] = False
    nodr = PPOPolicy(load_agent("ppo_wc.pt"))
    dr = PPOPolicy(load_agent("ppo_wc_dr.pt"))

    base_nodr = _mean_reward(nodr, base)
    base_dr = _mean_reward(dr, base)
    print(f"base reward：no-DR={base_nodr:.1f}, with-DR={base_dr:.1f}（越接近 0 越好）\n")
    print(f"{'落差強度':<12}{'no-DR 掉幅':>12}{'with-DR 掉幅':>14}{'DR 少掉':>10}")
    print("-" * 50)

    result: dict[str, dict[str, float]] = {}
    for level, (pf, rtt, cap) in _SHIFTS.items():
        shifted = _shifted(base, pf, rtt, cap)
        drop_nodr = base_nodr - _mean_reward(nodr, shifted)
        drop_dr = base_dr - _mean_reward(dr, shifted)
        result[level] = {
            "drop_nodr": drop_nodr, "drop_dr": drop_dr, "dr_saves": drop_nodr - drop_dr,
        }
        print(f"{level:<12}{drop_nodr:>12.1f}{drop_dr:>14.1f}{drop_nodr - drop_dr:>10.1f}")

    _OUT.mkdir(exist_ok=True, parents=True)
    (_OUT / "robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n掉幅越小 = 對『真實落差』越穩；『DR 少掉』>0 表示 DR 更穩。落差越大 DR 優勢應越明顯。")
    print(f"已寫出 {_OUT / 'robustness.json'}")


if __name__ == "__main__":
    main()
