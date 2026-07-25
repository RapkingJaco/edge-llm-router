"""環境檢查：印出這台機器對 edge-llm-router 的就緒狀態。

用法：
    uv run python scripts/check_env.py

核心項目（Python 版本 + 核心套件）缺失會以非 0 結束碼退出，方便 CI 把關；
GPU / Ollama / Gemini key 屬選配（Phase 6 / Phase 5 才需要），缺了只警告不失敗。
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

# 讓中文在任何 console codepage 都正確輸出（專案已知雷：Windows 預設非 UTF-8）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

OK = "[OK]"
WARN = "[--]"
FAIL = "[XX]"


def check_python() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 12)
    mark = OK if ok else FAIL
    print(f"{mark} Python {v.major}.{v.minor}.{v.micro}（需要 3.12）")
    return ok


def check_import(module: str, *, core: bool) -> bool:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        print(f"{OK} import {module} ({ver})")
        return True
    except Exception as exc:  # noqa: BLE001 — 環境檢查刻意寬鬆
        mark = FAIL if core else WARN
        note = "核心" if core else "選配"
        print(f"{mark} import {module} 失敗（{note}）：{exc}")
        return not core


def check_torch_cpu() -> bool:
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"{OK} torch {torch.__version__}（CUDA available={cuda}；訓練用 CPU 即可）")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} torch 匯入失敗（核心）：{exc}")
        return False


def check_cli(name: str, args: list[str]) -> None:
    path = shutil.which(name)
    if not path:
        print(f"{WARN} 找不到 {name}（選配）")
        return
    try:
        out = subprocess.run(  # noqa: S603
            [name, *args], capture_output=True, text=True, timeout=15
        )
        first = (out.stdout or out.stderr).strip().splitlines()
        print(f"{OK} {name}: {first[0] if first else path}")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} 執行 {name} 失敗（選配）：{exc}")


def check_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        print(f"{WARN} 無 nvidia-smi（GPU 選配；只有 Phase 6 Ollama 需要）")
        return
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"{OK} GPU: {out.stdout.strip() or '(查無)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} nvidia-smi 失敗（選配）：{exc}")


def check_gemini_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        print(f"{OK} GEMINI_API_KEY 已設定")
    else:
        print(f"{WARN} GEMINI_API_KEY 未設定（Phase 5/6 才需要）")


def main() -> int:
    print("=== edge-llm-router 環境檢查 ===")
    core_ok = True
    core_ok &= check_python()
    core_ok &= check_torch_cpu()
    for mod in ("numpy", "gymnasium", "yaml", "fastapi", "pydantic"):
        core_ok &= check_import(mod, core=True)
    for mod in ("mlflow", "matplotlib", "httpx", "ollama"):
        check_import(mod, core=False)
    print("--- 選配（Phase 5/6）---")
    check_gpu()
    check_cli("ollama", ["--version"])
    check_gemini_key()
    print("=" * 33)
    if core_ok:
        print(f"{OK} 核心環境就緒。")
        return 0
    print(f"{FAIL} 核心環境未就緒，請看上面 [XX]。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
