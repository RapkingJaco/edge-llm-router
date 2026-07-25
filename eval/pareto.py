"""P7-1 CLI：掃一整排 w，畫權重條件化 PPO 的 Pareto 前緣（成本 vs TTFT）。

一個 policy 沿著 w 從「重延遲」掃到「重成本」，描出整條「成本 vs TTFT」取捨曲線；
傳統策略各自只是一個固定點。用法：

    uv run python eval/pareto.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from edge_llm_router.agent.baselines import all_baselines  # noqa: E402
from edge_llm_router.agent.ppo import PPOPolicy, load_agent  # noqa: E402
from edge_llm_router.evaluation import evaluate, pareto_points  # noqa: E402

_OUT = Path(__file__).resolve().parents[1] / "outputs"
_W_GRID = tuple(round(i / 10, 1) for i in range(11))  # 0.0 … 1.0
_SEEDS = (0, 1, 2)


def _plot(ppo_pts, baseline_single, path: Path) -> None:
    plt.figure(figsize=(7, 4.6))
    xs = [p["cost"] for p in ppo_pts]
    ys = [p["ttft"] for p in ppo_pts]
    plt.plot(xs, ys, "-o", color="#3987e5", label="PPO (one policy, swept w)", zorder=3)
    colors = {"greedy": "#eb6834", "round_robin": "#e0a53a", "all_cloud": "#8a8f9c",
              "all_local": "#63c98c"}
    for name, (c, t) in baseline_single.items():
        plt.scatter([c], [t], color=colors.get(name, "#888"), marker="s", s=60,
                    label=name, zorder=4)
    plt.xlabel("total cost per episode  (← cheaper)")
    plt.ylabel("avg TTFT (ms)  (↓ faster)")
    plt.title("Pareto frontier — PPO sweeps the cost/latency trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main() -> None:
    ppo = PPOPolicy(load_agent("ppo_wc.pt"))
    results = evaluate([ppo, *all_baselines()], w_costs=_W_GRID, seeds=_SEEDS)

    ppo_pts = pareto_points(results, "ppo")
    # 基準線不看 w → 取 w=0.5 當代表點。
    baseline_single = {
        name: (results[name][0.5]["total_cost"], results[name][0.5]["avg_ttft_ms_served"])
        for name in ("greedy", "round_robin", "all_cloud", "all_local")
    }

    _OUT.mkdir(exist_ok=True, parents=True)
    (_OUT / "pareto.json").write_text(
        json.dumps({"ppo": ppo_pts, "baselines": baseline_single}, indent=2),
        encoding="utf-8",
    )
    _plot(ppo_pts, baseline_single, _OUT / "pareto.png")

    print("PPO Pareto 前緣（w_cost 由小到大）：")
    print(f"{'w_cost':>7}{'成本':>9}{'TTFT(ms)':>11}{'丟棄率':>9}")
    for p in sorted(ppo_pts, key=lambda x: x["w_cost"]):
        print(f"{p['w_cost']:>7}{p['cost']:>9.3f}{p['ttft']:>10.0f}{p['drop_rate']:>9.1%}")
    print(f"\n圖：{_OUT / 'pareto.png'}")


if __name__ == "__main__":
    main()
