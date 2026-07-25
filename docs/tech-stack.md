# 技術棧

| 層 | 技術 | 備註 |
|---|---|---|
| 語言 / 環境 | Python 3.12 + uv | lockfile 可重現 |
| RL / 模擬 | Gymnasium、NumPy、PyTorch(CPU)、CleanRL(PPO) | 訓練走 CPU |
| 真實後端 | Ollama(llama3.2, 邊緣)、Gemini 免費層(雲) | 成本自訂標價 |
| 控制層 | Gemini 結構化輸出 + 驗證安全網 | 中文方針 → 權重 |
| 服務 | FastAPI + WebSocket + asyncio | 即時推數據 |
| 前端 | Vite + React + TypeScript | 三面板儀表板 |
| 追蹤 / 出圖 | MLflow、matplotlib | 訓練回溯、Pareto/gap 圖 |
| 部署 | Docker、GitHub Actions、GCP | 雲端節點線上真跑 |

**0 元鐵則**：全開源；不接付費 API；LLM 走 Ollama / Gemini 免費層。

*(建置中)*
