"""P6-2 CLI：量真實 Ollama TTFT → 擬合 sim 參數 → 算校準前後 gap，出圖。

用法：
    uv run python calibration/run_calibration.py

需要 Ollama 在線。量測存 calibration/measurements/、gap 圖存 outputs/。
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

from edge_llm_router.backends.ollama_backend import ollama_available  # noqa: E402
from edge_llm_router.calibration import (  # noqa: E402
    calibrate,
    measure_ollama,
    predict_ttft,
)
from edge_llm_router.config import load_config  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_MEAS = _ROOT / "calibration" / "measurements"
_OUT = _ROOT / "outputs"


def _plot(ms, before, after, path: Path) -> None:
    xs = [m.input_tokens for m in ms]
    ys = [m.ttft_ms for m in ms]
    grid = sorted(set(xs))
    plt.figure(figsize=(7, 4.2))
    plt.scatter(xs, ys, color="#3987e5", label="measured (Ollama)", zorder=3)
    plt.plot(grid, [predict_ttft(before, g) for g in grid], "--", color="#8a8f9c",
             label="sim before")
    plt.plot(grid, [predict_ttft(after, g) for g in grid], "-", color="#63c98c",
             label="sim after (calibrated)")
    plt.xlabel("input tokens")
    plt.ylabel("TTFT (ms)")
    plt.title("Sim-to-Real calibration (edge / Ollama)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main() -> None:
    if not ollama_available():
        print("Ollama 未啟動（http://localhost:11434），無法校準。請先 `ollama serve`。")
        raise SystemExit(1)

    config = load_config()
    print("量測 Ollama TTFT（暖機 + 各輸入長度 3 次）…")
    ms = measure_ollama()
    result = calibrate(config, ms)

    _MEAS.mkdir(parents=True, exist_ok=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_MEAS / "edge_ttft.json").write_text(
        json.dumps([{"input_tokens": m.input_tokens, "ttft_ms": m.ttft_ms} for m in ms],
                   indent=2),
        encoding="utf-8",
    )
    _plot(ms, result["before"], result["after"], _OUT / "calibration_gap.png")

    b, a = result["before"], result["after"]
    gap_b = result["gap_before"]
    reduction = (1 - result["gap_after"] / gap_b) * 100 if gap_b else 0.0
    print(f"\n量測數：{result['n_measurements']}")
    print(f"校準前 params：rtt={b['rtt_ms']:.0f}ms, prefill={b['prefill_ms_per_token']:.3f} ms/tok")
    print(f"校準後 params：rtt={a['rtt_ms']:.0f}ms, prefill={a['prefill_ms_per_token']:.3f} ms/tok")
    print(
        f"gap（Wasserstein）：{result['gap_before']:.1f} → {result['gap_after']:.1f} ms"
        f"（縮小 {reduction:.0f}%）"
    )
    print(f"圖：{_OUT / 'calibration_gap.png'}")


if __name__ == "__main__":
    main()
