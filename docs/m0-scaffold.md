# m0 — 專案骨架 + 環境（Phase 0）

> 里程碑筆記：做了什麼、為什麼這樣設計、踩到什麼坑。

## 做了什麼

- `uv` 專案：`pyproject.toml`、`.python-version`、`uv.lock`；Python 3.12。
- 套件：gymnasium / numpy / torch(cpu) / httpx / ollama / fastapi / uvicorn / websockets /
  pydantic / pyyaml / mlflow / matplotlib；開發用 pytest / pytest-asyncio / ruff。
- 套件骨架 `src/edge_llm_router/`：`config.py`、`metrics.py` + 五個子模組
  （backends / sim / agent / control / server）空殼。
- 設定：`configs/default.yaml`（節點參數、reward 基準、工作負載、DR、PPO 超參）
  + `config.load_config()`。
- 工具：`scripts/check_env.py`（環境就緒檢查）、`.github/workflows/ci.yml`
  （ruff + pytest + check_env）、`CLAUDE.md`、`.env.example`。
- 測試：`tests/`（smoke / config / metrics），共 8 個。

## 為什麼這樣設計

- **`src/` 佈局**：避免測試時 import 到當前資料夾而非安裝的套件；逼自己當真套件寫。
- **設定集中在 YAML**：節點延遲/成本等數字之後要被真實量測校準，集中一檔就只改一處。
- **`metrics.py` 獨立**：TTFT/成本正規化、reward 標量化、領先% 會被 sim/agent/eval/server
  共用，集中定義避免各處重寫、定義漂移。
- **torch 綁 CPU index**：見 [environment.md](environment.md)。

## 踩到的坑

- **uv 裝 Python 3.11 失敗**：uv 0.11.27 在此機解壓 3.11 後，建 minor-version link 步驟
  報「Missing expected target directory」，重裝/清快取都無效。3.12 已就緒且無此問題，
  故全專案改用 3.12（套件全相容）。
- **`.env.*` 誤傷 `.env.example`**：`.gitignore` 的 `.env.*` 會一併忽略範例檔，
  已加 `!.env.example` 例外。
- **中文 console 亂碼**：`check_env.py` 開頭 reconfigure stdout 為 UTF-8。

## 驗收

- `uv run python -c "import edge_llm_router"` ✅
- `uv run ruff check` ✅ All checks passed
- `uv run pytest -q` ✅ 8 passed
- `uv run python scripts/check_env.py` ✅ 核心就緒（GPU/Ollama 也偵測到）
- CI 綠燈：待首次 push 驗證。

## 下一步

Phase 1：`backends/` 介面 + `SimulatedBackend`（會因忙碌變慢的排隊模型）。
