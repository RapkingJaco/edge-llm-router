"""server：LiveSimulation 可控狀態機 + WebSocket 煙霧測試。"""

from fastapi.testclient import TestClient

from edge_llm_router.control.rule_based import RuleBasedControl
from edge_llm_router.server.app import app
from edge_llm_router.server.simulation import LiveSimulation


def test_idle_does_not_step() -> None:
    sim = LiveSimulation()  # 一開始靜止
    snap = sim.tick()
    assert snap["mode"] == "idle"
    assert snap["progress"] == 0
    assert snap["ai"]["node"] is None


def test_run_steps_and_reports_lanes() -> None:
    sim = LiveSimulation()
    sim.start_run(40)
    snap = sim.tick()
    for key in ("ai", "base", "lead_pct", "ai_utils", "base_utils", "mode", "n_total", "progress"):
        assert key in snap
    assert snap["mode"] in ("running", "finished")
    assert snap["ai"]["node"] in ("local", "edge", "cloud")
    assert set(snap["ai_utils"]) == {"local", "edge", "cloud"}


def test_run_finishes_and_stops() -> None:
    sim = LiveSimulation()
    sim.start_run(30)
    snap = sim.tick()
    for _ in range(500):
        if snap["mode"] == "finished":
            break
        snap = sim.tick()
    assert snap["mode"] == "finished"
    assert snap["progress"] == snap["n_total"]
    # 停了：再 tick 不再前進
    assert sim.tick()["progress"] == snap["progress"]


def test_same_stream_is_fair() -> None:
    sim = LiveSimulation()
    sim.start_run(100)
    assert len(sim.ai_env._arrivals) == len(sim.base_env._arrivals) == sim._n_total


def test_reset_back_to_idle() -> None:
    sim = LiveSimulation()
    sim.start_run(20)
    sim.tick()
    sim.reset()
    assert sim.tick()["mode"] == "idle"


def test_set_policy_changes_w_live() -> None:
    # 用規則版控制層保持測試確定性（不依賴 llama3.2）。
    sim = LiveSimulation(control=RuleBasedControl())
    sim.set_policy("成本太高，多用本機")
    assert sim.w[1] > sim.w[0]  # 偏成本
    snap = sim.tick()
    assert snap["w"][1] > snap["w"][0]
    assert snap["note"]


def test_real_sample_populates_measured() -> None:
    sim = LiveSimulation()
    result = sim.real_sample("edge")
    assert result["node"] == "edge"
    assert {"ttft_ms", "is_measured", "backend"} <= set(result)
    assert sim.tick()["measured"] is not None


def test_websocket_run_reaches_finished() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["mode"] == "idle"
        ws.send_json({"cmd": "run", "n": 25})
        mode = "idle"
        for _ in range(300):
            mode = ws.receive_json()["mode"]
            if mode == "finished":
                break
        assert mode == "finished"


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
