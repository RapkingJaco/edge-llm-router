"""設定載入測試。"""

from edge_llm_router.config import load_config


def test_default_config_has_three_nodes() -> None:
    cfg = load_config()
    assert set(cfg["nodes"]) == {"local", "edge", "cloud"}


def test_cloud_is_pricier_than_local() -> None:
    cfg = load_config()
    local = cfg["nodes"]["local"]["cost_per_request"]
    cloud = cfg["nodes"]["cloud"]["cost_per_request"]
    # 雲端刻意標得比本機貴很多，逼 RL 別全丟雲。
    assert cloud >= local * 50


def test_reward_baselines_present() -> None:
    cfg = load_config()
    assert cfg["reward"]["ttft_baseline_ms"] > 0
    assert cfg["reward"]["cost_baseline"] > 0
