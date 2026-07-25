"""SimulatedBackend 排隊模型測試。"""

from edge_llm_router.backends.base import NodeBackend, Request
from edge_llm_router.backends.simulated import SimulatedBackend, build_nodes
from edge_llm_router.config import load_config


def _node(**overrides) -> SimulatedBackend:
    params = dict(
        name="t",
        rtt_ms=0.0,
        prefill_ms_per_token=1.0,
        decode_ms_per_token=1.0,
        capacity=1,
        cost_per_request=1.0,
        drop_wait_ms=1e9,
    )
    params.update(overrides)
    return SimulatedBackend(**params)


def test_is_a_node_backend() -> None:
    assert isinstance(_node(), NodeBackend)


def test_busier_means_higher_ttft() -> None:
    # 同一瞬間連發兩個請求，單槽節點：第二個得等第一個做完 → TTFT 較高。
    n = _node(capacity=1)
    req = Request(input_tokens=10, output_tokens=100)
    first = n.infer(req, now=0.0)
    second = n.infer(req, now=0.0)
    assert second.ttft_ms > first.ttft_ms
    assert not first.dropped and not second.dropped


def test_overload_gets_dropped() -> None:
    # 低容量 + 長生成 + 低丟棄門檻：連續灌爆終會被丟。
    n = _node(capacity=1, drop_wait_ms=50.0)
    req = Request(input_tokens=10, output_tokens=1000)
    assert any(n.infer(req, now=0.0).dropped for _ in range(10))


def test_dropped_request_costs_nothing() -> None:
    n = _node(capacity=1, drop_wait_ms=50.0, cost_per_request=1.0)
    req = Request(input_tokens=10, output_tokens=1000)
    results = [n.infer(req, now=0.0) for _ in range(10)]
    for r in results:
        if r.dropped:
            assert r.cost == 0.0


def test_reset_clears_queue() -> None:
    n = _node(capacity=1)
    n.infer(Request(input_tokens=10, output_tokens=100), now=0.0)
    n.reset()
    assert n.state(now=0.0).utilization == 0.0


def test_cloud_is_pricier_than_local_from_config() -> None:
    nodes = build_nodes(load_config())
    req = Request(input_tokens=10, output_tokens=10)
    local = nodes["local"].infer(req, now=0.0).cost
    cloud = nodes["cloud"].infer(req, now=0.0).cost
    assert cloud >= local * 50  # 雲端刻意標貴，逼 RL 別全丟雲


def test_state_utilisation_in_unit_range() -> None:
    n = _node(capacity=2)
    st = n.state(now=0.0)
    assert 0.0 <= st.utilization <= 1.0
    assert st.est_wait_ms == 0.0  # 全空時無需等待
