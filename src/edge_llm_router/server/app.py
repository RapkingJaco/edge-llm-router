"""FastAPI + WebSocket：即時廣播 AI vs 基準線快照，接前端指令。

啟動：
    uv run uvicorn edge_llm_router.server.app:app --reload

WebSocket ``/ws``：每 tick 送一個快照；可送指令 ``{"cmd":"reset"}`` /
``{"cmd":"peak","on":true}``。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .simulation import LiveSimulation

load_dotenv()  # 讀 .env（若有 GEMINI_API_KEY 供之後真雲端/抽驗用）

TICK_INTERVAL = 0.06  # 秒/步（≈16 步/秒，看得清又不太慢）
SAMPLE_EVERY = 160  # 每 ~10 秒背景抽驗一次真實後端（不阻塞模擬迴圈）
_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

app = FastAPI(title="edge-llm-router")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    _, ai_loaded = _probe_ai()
    return {"status": "ok", "ai_loaded": ai_loaded}


def _probe_ai() -> tuple[Any, bool]:
    from .simulation import load_ai_policy

    return load_ai_policy()


def _handle_command(sim: LiveSimulation, msg: dict[str, Any]) -> None:
    cmd = msg.get("cmd")
    if cmd == "run":
        sim.start_run(int(msg.get("n", 200)), bool(msg.get("peak", False)))
    elif cmd == "reset":
        sim.reset()
    elif cmd == "classify":
        sim.classify_request(str(msg.get("text", "")))
    elif cmd == "policy":
        sim.set_policy(str(msg.get("text", "")))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    sim = LiveSimulation()
    tick = 0
    sample_task: asyncio.Task | None = None
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=TICK_INTERVAL)
                if msg.get("cmd") == "policy":
                    # 控制層可能是真 LLM（llama3.2，數秒）→ 丟背景執行緒，別卡住迴圈
                    await asyncio.to_thread(sim.set_policy, str(msg.get("text", "")))
                else:
                    _handle_command(sim, msg)
            except TimeoutError:
                pass

            tick += 1
            # 跑的時候每 SAMPLE_EVERY 步在背景真打一次後端（edge/cloud 輪流），不阻塞迴圈
            if (
                sim.mode == "running"
                and tick % SAMPLE_EVERY == 0
                and (sample_task is None or sample_task.done())
            ):
                node = "edge" if (tick // SAMPLE_EVERY) % 2 == 0 else "cloud"
                sample_task = asyncio.create_task(asyncio.to_thread(sim.real_sample, node))

            await websocket.send_json(sim.tick())
    except WebSocketDisconnect:
        return


# 有 build 產物就由 FastAPI 服務前端（單 port demo / 部署）；放最後避免蓋掉 /ws、/health。
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
