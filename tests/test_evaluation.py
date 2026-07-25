"""評估引擎測試（用小 sweep 保持快速）。"""

from edge_llm_router.agent.baselines import all_baselines
from edge_llm_router.evaluation import (
    best_reward_policy,
    evaluate,
    pareto_points,
    render_html,
    save,
)

_W = (0.5,)
_SEEDS = (0, 1)


def _run() -> dict:
    return evaluate(all_baselines(), w_costs=_W, seeds=_SEEDS)


def test_results_have_all_policies_and_metrics() -> None:
    results = _run()
    assert set(results) == {"greedy", "round_robin", "all_cloud", "all_local"}
    for per_w in results.values():
        m = per_w[0.5]
        for key in ("drop_rate", "avg_ttft_ms_served", "total_cost", "total_reward"):
            assert isinstance(m[key], float)


def test_all_local_is_not_best_at_balanced_w() -> None:
    # 全本機在平衡 w 下狂丟棄，不該是 reward 最佳。
    results = _run()
    assert best_reward_policy(results, 0.5) != "all_local"


def test_save_writes_json_and_html(tmp_path) -> None:
    results = _run()
    json_path, html_path = save(results, _W, _SEEDS, outputs=tmp_path)
    assert json_path.exists() and html_path.exists()
    assert "基準線" in html_path.read_text(encoding="utf-8")


def test_pareto_points_sorted_by_cost() -> None:
    results = evaluate(all_baselines(), w_costs=(0.2, 0.8), seeds=(0,))
    pts = pareto_points(results, "greedy")
    assert [p["cost"] for p in pts] == sorted(p["cost"] for p in pts)
    assert set(pts[0]) == {"w_cost", "cost", "ttft", "drop_rate"}


def test_render_html_marks_a_best_row() -> None:
    results = _run()
    html = render_html(results, _W, _SEEDS)
    assert 'class="best"' in html
