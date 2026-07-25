"""CLI：跑基準線對照、存 JSON + HTML、印摘要表。

用法：
    uv run python eval/compare.py
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from edge_llm_router.agent.baselines import all_baselines  # noqa: E402
from edge_llm_router.evaluation import (  # noqa: E402
    DEFAULT_SEEDS,
    DEFAULT_W_COSTS,
    evaluate,
    save,
    to_console,
)


def main() -> None:
    results = evaluate(all_baselines())
    json_path, html_path = save(results, DEFAULT_W_COSTS, DEFAULT_SEEDS)
    print(to_console(results, DEFAULT_W_COSTS))
    print(f"\n已寫出 {json_path}")
    print(f"已寫出 {html_path}")


if __name__ == "__main__":
    main()
