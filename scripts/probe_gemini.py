"""驗證真 Gemini 雲端後端：讀 .env 的 key，真打幾次量首字延遲（TTFT）。

用法：
    uv run python scripts/probe_gemini.py

需要 D:\\RLrepo\\.env 內有 `GEMINI_API_KEY=...`（免費層即可）。沒 key 會清楚說明缺什麼。
"""

from __future__ import annotations

import importlib.util
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from edge_llm_router.backends.base import Request  # noqa: E402
from edge_llm_router.backends.gemini_backend import (  # noqa: E402
    build_cloud_backend,
    gemini_available,
)
from edge_llm_router.config import load_config  # noqa: E402


def main() -> None:
    if not gemini_available():
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        has_sdk = importlib.util.find_spec("google.genai") is not None
        print(f"Gemini 尚未就緒 → key={'有' if has_key else '無'}, SDK={'有' if has_sdk else '無'}")
        if not has_key:
            print("請在 D:\\RLrepo\\.env 放一行：GEMINI_API_KEY=你的key")
            print("（免費 key：https://aistudio.google.com/apikey）")
        raise SystemExit(1)

    backend = build_cloud_backend(load_config())
    print(f"雲端後端類型：{type(backend).__name__}（Gemini→Ollama→模擬 降級鏈）")
    print("真打雲端 3 次量 TTFT（Gemini 額度沒了會自動退本機 Ollama）：")
    for i in range(3):
        res = backend.infer(Request(input_tokens=32, output_tokens=16))
        if res.dropped:
            print(f"  第{i + 1}次：失敗/丟棄（檢查 key 是否有效、是否超額）")
        else:
            print(f"  第{i + 1}次：TTFT={res.ttft_ms:.0f}ms（實測），成本標價={res.cost}")


if __name__ == "__main__":
    main()
