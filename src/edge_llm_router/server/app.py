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
    if cmd == "reset":
        sim.reset()
    elif cmd == "peak":
        sim.set_peak(bool(msg.get("on", True)))
    elif cmd == "policy":
        sim.set_policy(str(msg.get("text", "")))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    sim = LiveSimulation()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=TICK_INTERVAL)
                _handle_command(sim, msg)
            except TimeoutError:
                pass
            await websocket.send_json(sim.tick())
    except WebSocketDisconnect:
        return


# 有 build 產物就由 FastAPI 服務前端（單 port demo / 部署）；放最後避免蓋掉 /ws、/health。
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
