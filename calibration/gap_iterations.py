"""P7-2 CLI：gap 隨校準迭代下降。

每一輪多收一批真實 Ollama 量測、重新擬合 sim 參數，在**獨立 held-out 真實量測**上算 gap。
隨著校準資料變多，模擬對真實的 gap 下降並收斂——「模擬越調越像真實」的鐵證。

用法：
    uv run python calibration/gap_iterations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from edge_llm_router.backends.base import Request  # noqa: E402
from edge_llm_router.backends.ollama_backend import OllamaBackend, ollama_available  # noqa: E402
from edge_llm_router.calibration import (  # noqa: E402
    Measurement,
    fit_edge_params,
    gap_wasserstein,
)
from edge_llm_router.config import load_config  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"
_INPUT_GRID = (16, 64, 256, 512)
_ROUNDS = 6


def _measure_round(backend: OllamaBackend) -> list[Measurement]:
    out: list[Measurement] = []
    for it in _INPUT_GRID:
        res = backend.infer(Request(input_tokens=it, output_tokens=8))
        if not res.dropped:
            out.append(Measurement(input_tokens=it, ttft_ms=res.ttft_ms))
    return out


def main() -> None:
    if not ollama_available():
        print("Ollama 未啟動，無法量測。")
        raise SystemExit(1)

    backend = OllamaBackend()
    backend.infer(Request(input_tokens=8, output_tokens=8))  # warm-up
    print(f"量測 {_ROUNDS} 輪 train + 1 輪 held-out（每輪 {len(_INPUT_GRID)} 點）…")
    train_rounds = [_measure_round(backend) for _ in range(_ROUNDS)]
    holdout = _measure_round(backend)

    cfg = load_config()["nodes"]["edge"]
    placeholder = {"rtt_ms": cfg["rtt_ms"], "prefill_ms_per_token": cfg["prefill_ms_per_token"]}

    gaps = [gap_wasserstein(holdout, placeholder)]  # iteration 0：未校準
    cumulative: list[Measurement] = []
    for r in train_rounds:
        cumulative += r
        gaps.append(gap_wasserstein(holdout, fit_edge_params(cumulative)))

    _OUT.mkdir(exist_ok=True, parents=True)
    (_OUT / "gap_iterations.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4.2))
    plt.plot(range(len(gaps)), gaps, "-o", color="#3987e5")
    plt.xlabel("calibration iteration (0 = uncalibrated placeholder)")
    plt.ylabel("gap to real Ollama — Wasserstein (ms)")
    plt.title("Sim-to-Real gap shrinks as calibration data grows")
    plt.tight_layout()
    plt.savefig(_OUT / "gap_iterations.png", dpi=110)
    plt.close()

    print("\niteration :  gap(ms)")
    for i, g in enumerate(gaps):
        tag = "  (未校準)" if i == 0 else ""
        print(f"    {i:>2}    : {g:>8.1f}{tag}")
    print(f"\n圖：{_OUT / 'gap_iterations.png'}")


if __name__ == "__main__":
    main()
