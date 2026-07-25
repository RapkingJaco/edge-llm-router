"""評估引擎：同一請求流、同一組 w，跨多個 w 與多個 seed 比較各策略。

邏輯放在套件裡（可 import、可測）；``eval/compare.py`` 只是薄 CLI 入口。
之後 PPO 訓好，把它的 policy 加進 ``policies`` 一起比即可（P3-4）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent.policy import Policy
from .agent.runner import run_episode
from .sim.env import RouterEnv

_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
_METRICS = ("drop_rate", "avg_ttft_ms_served", "total_cost", "total_reward")

DEFAULT_W_COSTS = (0.1, 0.5, 0.9)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _mean_metrics(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {m: sum(r[m] for r in runs) / len(runs) for m in _METRICS}


def evaluate(
    policies: list[Policy],
    *,
    w_costs: tuple[float, ...] = DEFAULT_W_COSTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[float, dict[str, float]]]:
    """回傳 ``results[policy_name][w_cost] = {metric: 平均值}``。"""
    results: dict[str, dict[float, dict[str, float]]] = {}
    for wc in w_costs:
        env = RouterEnv(config=config, fixed_w=(round(1.0 - wc, 3), wc))
        for policy in policies:
            runs = [run_episode(env, policy, seed=s) for s in seeds]
            results.setdefault(policy.name, {})[wc] = _mean_metrics(runs)
    return results


def pareto_points(results: dict, name: str) -> list[dict[str, float]]:
    """某策略跨 w 的 (w_cost, cost, ttft, drop_rate) 點，依成本排序，供畫 Pareto 前緣。"""
    pts = [
        {
            "w_cost": wc,
            "cost": m["total_cost"],
            "ttft": m["avg_ttft_ms_served"],
            "drop_rate": m["drop_rate"],
        }
        for wc, m in results[name].items()
    ]
    return sorted(pts, key=lambda p: p["cost"])


def best_reward_policy(results: dict, wc: float) -> str:
    """該 w 下 total_reward 最高（最接近 0）的策略名。"""
    return max(results, key=lambda name: results[name][wc]["total_reward"])


def to_console(results: dict, w_costs: tuple[float, ...]) -> str:
    lines: list[str] = []
    for wc in w_costs:
        best = best_reward_policy(results, wc)
        lines.append(f"\n== w_cost={wc}（w_lat={round(1 - wc, 2)}）==")
        lines.append(f"{'策略':<13}{'丟棄率':>8}{'平均TTFT':>11}{'總成本':>9}{'總reward':>11}")
        lines.append("-" * 52)
        for name, per_w in results.items():
            m = per_w[wc]
            star = " *最佳" if name == best else ""
            lines.append(
                f"{name:<13}{m['drop_rate']:>7.1%}{m['avg_ttft_ms_served']:>9.0f}ms"
                f"{m['total_cost']:>9.3f}{m['total_reward']:>11.1f}{star}"
            )
    return "\n".join(lines)


def render_html(results: dict, w_costs: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    sections: list[str] = []
    for wc in w_costs:
        best = best_reward_policy(results, wc)
        rows = []
        for name, per_w in results.items():
            m = per_w[wc]
            cls = ' class="best"' if name == best else ""
            rows.append(
                f"<tr{cls}><td>{name}</td><td>{m['drop_rate']:.1%}</td>"
                f"<td>{m['avg_ttft_ms_served']:.0f} ms</td>"
                f"<td>{m['total_cost']:.3f}</td><td>{m['total_reward']:.1f}</td></tr>"
            )
        sections.append(
            f"<h2>w_cost = {wc} &nbsp;(w_lat = {round(1 - wc, 2)})</h2>"
            "<table><thead><tr><th>策略</th><th>丟棄率</th><th>平均 TTFT</th>"
            "<th>總成本</th><th>總 reward</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    style = (
        "body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.6}"
        "table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "tr.best{background:#e6f4ea;font-weight:600}"
        "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}"
        "th,td{border-color:#444}tr.best{background:#14351f}}"
    )
    return (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>edge-llm-router 基準線對照</title>"
        f"<style>{style}</style></head><body>"
        "<h1>基準線對照（control group）</h1>"
        f"<p>每格為 {len(seeds)} 個 seed 的平均；同一請求流、同一組 w。"
        "高亮列 = 該 w 下 reward 最佳。reward 越接近 0 越好。</p>"
        + "".join(sections)
        + "</body></html>"
    )


def save(
    results: dict,
    w_costs: tuple[float, ...],
    seeds: tuple[int, ...],
    outputs: Path | None = None,
    stem: str = "eval_baselines",
) -> tuple[Path, Path]:
    """寫出 JSON + HTML，回傳兩個路徑。"""
    out = outputs if outputs is not None else _OUTPUTS
    out.mkdir(exist_ok=True, parents=True)
    payload = {
        "w_costs": list(w_costs),
        "seeds": list(seeds),
        "results": {
            name: {str(k): v for k, v in per_w.items()} for name, per_w in results.items()
        },
    }
    json_path = out / f"{stem}.json"
    html_path = out / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(results, w_costs, seeds), encoding="utf-8")
    return json_path, html_path
