"""設定載入。

所有可調參數（節點延遲/成本、正規化基準、工作負載、DR 範圍、訓練超參）集中在
``configs/default.yaml``；校準時只改那一個檔，程式不動。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# config.py 在 src/edge_llm_router/ 底下；往上三層是 repo 根目錄。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """讀取 YAML 設定檔並回傳巢狀 dict。

    Args:
        path: 設定檔路徑；``None`` 時用預設的 ``configs/default.yaml``。
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
