# edge-llm-router

**RL 智慧 LLM 推論分流器** — 一個互動展示平台。

很多使用者同時發 LLM 對話請求，一個用**強化學習（PPO）**訓練的 agent 即時決定每個請求在 **本機 / 邊緣 / 雲** 哪裡處理，最佳化**首字延遲（TTFT）與成本**；使用者可用**自然語言**即時調整優化目標，一個真的 LLM 把白話翻成 RL 的獎勵權重，排程行為當場改變。

> LLM serving / inference routing 方向的作品集專案。詳細設計見 [`docs/superpowers/specs/2026-07-24-edge-llm-router-design.md`](docs/superpowers/specs/2026-07-24-edge-llm-router-design.md)。

## 亮點
- **位置/層級路由**（本機/邊緣/雲）：computation offloading，RL 動態分流。
- **目標條件化 RL**：一句自然語言即時改優化目標、零重訓切換行為；可掃出 Pareto 前緣。
- **Sim-to-Real 校準**：邊緣節點真跑 Ollama、雲端真接 Gemini，模擬器被真實硬體量測校準。

## 技術棧
Python · PyTorch · CleanRL(PPO) · Gymnasium · Ollama · Gemini · FastAPI + WebSocket · Docker · GitHub Actions

*(建置中)*
