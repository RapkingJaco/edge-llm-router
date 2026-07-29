# CLAUDE.md — edge-llm-router

給在此 repo 工作的 Claude 的專案指示。

## 這是什麼
AI Agent 智慧 LLM 推論分流器的**互動展示平台**（求職作品集，不是聊天機器人）。PPO agent
逐請求決定在 本機/邊緣/雲 哪裡跑，優化 TTFT + 成本；使用者用中文下方針，LLM 轉成
reward 權重，行為零重訓即時改變。

- 設計規格（先讀）：`docs/superpowers/specs/2026-07-24-edge-llm-router-design.md`
- 實作計畫（照它做）：`docs/superpowers/plans/2026-07-24-edge-llm-router-implementation-plan.md`

## 環境 / 指令（用 uv，勿用 pip/conda）
- Python 3.12，全走 `uv`（原計畫寫 3.11，但 uv 0.11.27 在此機安裝 3.11 卡在建 link 步驟，
  改用已裝好的 3.12；套件全相容，對專案無影響）。
- 安裝：`uv sync --dev`
- 測試：`uv run pytest -q`
- Lint：`uv run ruff check`（`--fix` 自動修）
- 環境檢查：`uv run python scripts/check_env.py`
- torch 綁 PyTorch CPU index（見 pyproject `[tool.uv.sources]`）：GPU 只給 Ollama，
  PPO 小 MLP 用 CPU 即可。

## 鐵則
- **成本節制**：預設優先用免費資源（本機 Ollama）；雲端 Gemini 走**小額預付**額度，
  僅用於真打驗證 / 線上抽驗。避免無謂大額花費；實際加值一律由作者本人執行。
- **設定集中**：可調參數放 `configs/default.yaml`，程式從 `config.load_config()` 讀，
  不要把數字散寫在程式裡。
- **隱私**：`.env`（API key）不進版控。`_交接_繼續開發.md` 若轉 public 前需處理。
- **不用上帝視角**：agent 的 observation 只能有真實環境也拿得到的量。
- **文件庫**：每做完一個 Phase，補一份 `docs/mN-*.md`（做了什麼、為什麼、踩什麼坑）。

## 教學風格（作者 Jacob）
RL 有底子在鞏固、LLM 新學。要白話 + 比喻 + 記憶鉤 + 邊做邊教；術語配真解釋、不硬翻
怪詞；先結論再解釋。

## 建置順序
backends → sim → agent（先拿到「AI 贏基準線」）→ server + web → control → 校準/部署。
