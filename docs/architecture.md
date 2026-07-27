# 系統架構

`edge-llm-router` 分成四層、六個模組。設計核心是**乾淨的邊界**：每個模組只做一件事、透過明確介面溝通，可以各自獨立理解與測試。

![系統架構圖](img/architecture.png)

---

## 資料流：一個請求的一生（即時 demo）

1. **`sim`** 產生請求流（多使用者、隨機到達、每個請求帶估計 token 數），組成 observation——**含當前的偏好權重 `w`**。
2. **`server`** 呼叫 **`agent`** 的 `policy.predict(obs)`，PPO 策略**逐請求**選一個節點（本機/邊緣/雲）。
3. **`sim`** 對選中的節點呼叫 **`backends`** 的 `infer()`，回傳 `InferResult`（TTFT、成本、token 數）。
4. `sim` 更新佇列/負載、結算 reward 與指標；同一條請求流**並行**餵給基準線比較。
5. `server` 把即時狀態透過 **WebSocket** 推給 **`web`** 前端渲染。

**改方針的那條線**：使用者在前端打一句中文（「成本太高，多用本機」）→ `server` → **`control`** 解析成權重 `w` → 更新 `sim` 的當前 `w` → 下一步 `policy` 讀到新 `w`，**行為當場改變、零重訓**。

---

## 四層 / 六模組

| 層 | 模組 | 職責 | 對外介面 | 技術 |
|---|---|---|---|---|
| 前端 | `web` | 三面板即時儀表板、自我說明 | 連 WebSocket 渲染 | React + Vite + TS |
| 後端 | `server` | 跑即時模擬迴圈、推數據、接指令 | REST + WebSocket | FastAPI、asyncio |
| 後端 | `control` | 中文方針 → 驗證過的權重 `w` | `parse(text, w) → w` | 規則式 / Gemini 結構化輸出 + 安全網 |
| 強化學習 | `agent` | 訓練 policy + 提供推論；含基準線 | `policy.predict(obs) → action` | CleanRL PPO（權重條件化） |
| 強化學習 | `sim` | 請求流 + 三節點排隊模型 + Gym 介面 | `Env`(obs、Discrete(3)、reward) | Gymnasium、NumPy |
| 節點後端 | `backends` | 「給請求 → 回 TTFT/成本」 | `NodeBackend.infer(req) → InferResult` | 可插拔 + 自動降級 |

---

## 為什麼這樣切（乾淨邊界）

- **`sim` 不知道後端是真是假**——它只看到「一個節點回了 `InferResult`」。所以訓練用模擬、展示接真實，上層一行都不用改。
- **`server` 不碰 PPO 內部**——只呼叫 `policy.predict`，換演算法互不影響。
- **`web` 不知道模擬細節**——只渲染 `server` 串流來的狀態。
- **`control` 可整包抽換**——規則式或 Gemini 都實作同一個 `parse` 介面。

每個模組都能單獨寫測試（全專案 66 個測試）。

---

## 節點後端：真實 vs 模擬 + 優雅降級

三個後端都實作同一個 `NodeBackend` 介面，上層無感：

| 後端 | 真/模擬 | 角色 |
|---|---|---|
| `SimulatedBackend` | 🟤 模擬 | 公式排隊模型；訓練與線上高吞吐都靠它（校準過） |
| `OllamaBackend` | 🟢 真實 | 4070 邊緣真跑 llama3.2，量真實 TTFT |
| `GeminiBackend` | 🟢 真實 | 雲端，可插拔 |

**降級鏈**：Gemini 額度用盡 → 退回本機 Ollama → 再退回模擬。**服務永不中斷**，線上（無 GPU 的 GCP）也照跑。

---

*相關：核心觀念見 [concepts.md](concepts.md)；各階段實作筆記見 [m0](m0-scaffold.md)–[m7](m7-advanced.md)。*
