"""Phase 1 demo：直觀看 SimulatedBackend 的排隊行為（忙 → 變慢 → 塞爆丟棄）。

用法：
    uv run python scripts/demo_backends.py

同一瞬間（now=0）連發多個請求，對比「便宜但一擠就爆」的 local 與「穩但貴」的 cloud。
"""

from __future__ import annotations

import sys

# 讓中文在任何 console codepage 都正確輸出（Windows 預設非 UTF-8）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.backends.base import Request  # noqa: E402
from edge_llm_router.backends.simulated import build_nodes  # noqa: E402
from edge_llm_router.config import load_config  # noqa: E402


def _hammer(node, req: Request, n: int) -> None:
    for i in range(n):
        r = node.infer(req, now=0.0)
        st = node.state(now=0.0)
        tag = "  <== 丟棄!" if r.dropped else ""
        print(
            f"  #{i + 1}: TTFT={r.ttft_ms:8.1f}ms  util={st.utilization:.2f}  "
            f"cost={r.cost:.4f}{tag}"
        )


def main() -> None:
    nodes = build_nodes(load_config())
    req = Request(input_tokens=128, output_tokens=256)
    n = 8

    for name in ("local", "cloud"):
        node = nodes[name]
        node.reset()
        print(
            f"\n[{name}] capacity={node.capacity}, drop_wait={node.drop_wait_ms:.0f}ms"
            f" — 同一瞬間連發 {n} 個請求："
        )
        _hammer(node, req, n)

    print(
        "\n重點：local 一擠就爆（排隊超過門檻被丟棄），cloud 穩但每個都貴 100×。"
        "\n這就是 RL agent 之後要學會權衡的張力。"
    )


if __name__ == "__main__":
    main()
