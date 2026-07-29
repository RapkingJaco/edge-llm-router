# 技術棧

| 層 | 技術 | 備註 |
|---|---|---|
| 語言 / 環境 | Python 3.12 + uv | lockfile 可重現 |
| RL / 模擬 | Gymnasium、NumPy、PyTorch(CPU)、CleanRL(PPO) | 訓練走 CPU |
| 真實後端 | Ollama(llama3.2, 邊緣)、Gemini(雲, 小額預付) | RL 用自訂標價；真打計真實小額費用 |
| 控制層 | Ollama(llama3.2) 結構化輸出 + 規則版 fallback | 中文方針 → 權重 |
| 服務 | FastAPI + WebSocket + asyncio | 即時推數據 |
| 前端 | Vite + React + TypeScript | 三面板儀表板 |
| 追蹤 / 出圖 | MLflow、matplotlib | 訓練回溯、Pareto/gap 圖 |
| 部署 | Docker、GitHub Actions、GCP | 雲端節點線上真跑 |

**成本節制**：預設全開源、免費（本機 Ollama）；雲端 Gemini 走**小額預付**，僅用於真打驗證。

*(建置中)*
