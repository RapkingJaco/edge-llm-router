"""server：LiveSimulation 邏輯 + WebSocket 煙霧測試。"""

from fastapi.testclient import TestClient

from edge_llm_router.control.rule_based import RuleBasedControl
from edge_llm_router.server.app import app
from edge_llm_router.server.simulation import LiveSimulation


def test_tick_returns_both_lanes() -> None:
    sim = LiveSimulation()
    snap = sim.tick()
    for key in ("ai", "base", "lead_pct", "ai_utils", "base_utils", "t"):
        assert key in snap
    assert snap["ai"]["node"] in ("local", "edge", "cloud")
    assert set(snap["ai_utils"]) == {"local", "edge", "cloud"}


def test_same_stream_is_fair() -> None:
    # 兩條 env 同 seed → 同一份工作負載（step 數相同）。
    sim = LiveSimulation()
    assert len(sim.ai_env._arrivals) == len(sim.base_env._arrivals)


def test_cumulative_reward_grows_negative() -> None:
    sim = LiveSimulation()
    for _ in range(20):
        snap = sim.tick()
    assert snap["t"] == 20
    assert snap["ai"]["cum_reward"] <= 0.0


def test_peak_increases_arrivals() -> None:
    sim = LiveSimulation()
    normal = len(sim.ai_env._arrivals)
    sim.set_peak(True)
    assert len(sim.ai_env._arrivals) > normal


def test_websocket_streams_snapshots() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert "ai" in first and "base" in first
        ws.send_json({"cmd": "reset"})
        after = ws.receive_json()
        assert "ai" in after


def test_set_policy_changes_w_live() -> None:
    # 用規則版控制層保持測試確定性（不依賴 llama3.2）。
    sim = LiveSimulation(control=RuleBasedControl())
    sim.set_policy("成本太高，多用本機")
    assert sim.w[1] > sim.w[0]  # 偏成本
    snap = sim.tick()
    assert snap["w"][1] > snap["w"][0]
    assert snap["note"]  # 有解讀字串


def test_real_sample_populates_measured() -> None:
    # 抽驗：edge 有 Ollama 就真打(is_measured=True)，沒有就退模擬——都要能填進快照。
    sim = LiveSimulation()
    result = sim.real_sample("edge")
    assert result["node"] == "edge"
    assert {"ttft_ms", "is_measured", "backend"} <= set(result)
    assert sim.tick()["measured"] is not None


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
