# edge-llm-router 設計規格

> 求職作品集。**RL 智慧 LLM 推論分流器**的展示平台。
> 定案日期：2026-07-24。作者：Jacob（陳品榤）。
> 本文件由腦力激盪逐段確認後彙整；實作計畫另由 writing-plans 產出。

---

## 1. 一句話 / 定位

一個**互動網頁展示平台**：很多使用者同時發 LLM 對話請求，作者訓練的 **RL agent 即時決定每個請求在「本機(手機) / 邊緣 / 雲」哪裡處理**，目標是「首字最快（TTFT）+ 最省成本」；使用者可用**中文對系統下方針**（例如「成本太高，多用本機」），一個**真的 LLM** 把白話翻成 RL 的優化權重，排程行為當場改變。

**它是什麼**：LLM 推論的「智慧負載平衡器 / 路由器」展示台（LLM serving / inference routing 職缺方向）。
**它不是**：聊天機器人（聊天只是流過管線的工作負載）、也不是真實可用的商用閘道（那是進階選項）。

**差異化賣點（面試三句話）**：
1. **位置/層級路由**（本機/邊緣/雲），接作者碩論的 computation offloading（VR 卸載 → LLM 分流，同結構換負載）。
2. **目標條件化 RL**：一句自然語言即時改優化目標、**零重訓**切換行為。
3. **Sim-to-Real 校準**：模擬器被真實硬體量測校準。

---

## 2. 目標 / 非目標（YAGNI）

**v1 目標**
- 一個 Gymnasium 模擬環境（請求流 + 三節點排隊模型）。
- CleanRL PPO 訓練出的分流 policy，**贏過基準線**（貪婪/輪詢/全雲/全本機）。
- 權重條件化：一個 policy 覆蓋整條「延遲 vs 成本」偏好光譜。
- FastAPI + WebSocket 即時儀表板（三面板）。
- LLM 控制層（Gemini 結構化輸出）把中文方針轉成權重。
- 邊緣(4070 Ollama)、雲端(Gemini) 真實後端 + 自動 fallback 全模擬。
- Sim-to-Real 校準 + gap 指標。
- Docker + pytest + GitHub Actions CI + 部署（GCP）。

**非目標（先不做）**
- 真實可用的商用 LLM 閘道。
- LSTM 循環策略（列為選配消融實驗）。
- Attention over queue（進階）。
- MLflow / 模型版本管理（進階）。
- Agent / RAG / fine-tuning（用不到）。

---

## 3. 真實度階梯與節點對應

| 節點 | 誰扮演 | 延遲 | 成本 | 併發容量 | 線上(GCP)可真跑？ |
|---|---|---|---|---|---|
| **本機 local（弱手機）** | 模擬（或超小 1B 模型），數字錨定公開 benchmark | 中，一擠就慢 | ≈0（電費） | 低 | ❌ 退模擬 |
| **邊緣 edge** | 🟢 **作者 4070 桌機真跑 Ollama**（桌機物理上就是邊緣 GPU 盒） | 真實量測 | 我標的(小) | 中 | ❌ 無 GPU 退模擬 |
| **雲端 cloud** | 🟢 **真 Gemini 免費層**（結構化串流） | 真實量測(含網路 RTT) | **我標的($X/次，明顯貴)** | 高 | ✅ 只要有網路+key |

**關鍵原則**
- **成本一律「自己標價」**：Gemini 免費層實際 0 元，但我們假裝雲端很貴（比本機貴約 100× 量級），否則 RL 會學到「全丟雲端」。→ 延遲=真、成本=模擬標價。
- **訓練/展示分離**：訓練 PPO 全用（校準過的）模擬（快、不打 API）；線上請求洪流大多走模擬（撐吞吐、守免費層額度）；**抽驗少量請求真打 Ollama/Gemini**，標「✅實測」秀在儀表板，同時當 Sim-to-Real 驗證儀。
- **可插拔後端**：`NodeBackend.infer(req)→InferResult`；`SimulatedBackend`/`OllamaBackend`/`GeminiBackend`；偵測不到 GPU/Ollama 自動 fallback。

---

## 4. MDP 設計

### Observation（約 15 維，全部正規化到有界範圍）
| 塊 | 欄位 | 維度 |
|---|---|---|
| 這個請求 | 估計輸入 token、估計輸出 token | 2 |
| 本機/邊緣/雲各自 | 使用率、排隊長度、預估等待時間 | 3×3 = 9 |
| 全域 | 累積成本、目前負載程度 | 2 |
| **方針 w** | w_lat、w_cost | 2 |

> `w_lat, w_cost ∈ [0,1]`, `w_lat + w_cost = 1`。餵兩個雖冗餘但好學、可擴到多目標。
> **不可洩上帝視角**：agent 只看真實環境也拿得到的量（估計 token、觀測佇列、最近量到的延遲），不得看模擬器內部真值/未來。token 估計要帶雜訊（真實只能估）。

### Action
`Discrete(3)`：逐請求選 本機/邊緣/雲。儀表板另把分流結果畫成**即時 ratio 曲線**（免費拿到論文 LocalRatio 敘事）。

### Reward（請求完成時結算，延遲獎勵交給 GAE）
```
r = -( w_lat · 正規化TTFT + w_cost · 正規化成本 ) - 懲罰(逾時/丟棄)
正規化值 = clip( 原始值 / 固定基準, 0, 1 )
```
- **固定基準（可解釋）**：TTFT 基準＝可接受上限（如 2 秒 SLA）；成本基準＝全用雲端每請求花費（最貴情況）。避免動態統計漂移。
- 逾時/塞爆丟棄 → 大扣分，逼 agent 別把便宜節點塞死。
- obs 特徵正規化亦用固定基準或 VecNormalize。

### Episode / 工作負載
- 請求隨時間到達，用**會變動的到達率**製造尖峰。
- 每請求：從分布抽輸入/輸出 token（短/中/長混合），文字從**預備 prompt 池**抽（給真實後端）。
- 一 episode ＝ 固定模擬時長或請求數；**訓練時每 episode 隨機抽一組 w**（權重條件化訓練）。

### 權重條件化（核心 DL 技術）
把 reward 權重當 observation 輸入、訓練時隨機抽 w，學成一整族 policy `π(a|s,w)` 與 `V(s,w)`（**goal-conditioned / UVFA / 多目標 RL scalarization**）。demo 時 LLM 改 w → 直接餵入 → **零重訓即時切換行為**。掃 w 可描出 **Pareto 前緣**（成本 X、TTFT Y），當殺手級圖表。

---

## 5. 節點模型（讓問題有意思的關鍵）

- **必含「忙→變慢」的負載相依延遲（排隊）**：同時處理越多越慢；否則分流無意義。本機便宜但一擠就爆、雲端貴但穩。
- TTFT 組成 ≈ 排隊等待 + 網路 RTT + prefill 時間；prefill ≈ `prefill_ms/token × 輸入token`。
- 延遲/成本公式**參數化**，供真實量測回填校準。

---

## 6. Sim-to-Real 縮 gap 策略

1. **系統辨識/校準**：量真（Ollama TTFT vs token、decode 率、併發惡化曲線、冷啟動、熱降頻；Gemini 延遲/RTT/限速）→ 回歸擬合 sim 參數。
2. **Domain Randomization**：每 episode 在真值±範圍隨機化（prefill 率±20~40%、RTT+jitter、容量±1~2、到達率/突發、token 估計誤差、限速機率、冷啟動尖峰）。**範圍取自實測變異，勿開太寬**（否則過度保守、到處平庸）。
3. **現實落差回饋迴圈**：訓練→部署→比對模擬預測 vs 真實量測→用 KL/Wasserstein 算分布落差→回填參數、調 DR→重訓。畫「**gap 隨迭代下降**」圖當鐵證。
4. **LLM 專屬雜訊**：TTFT 從擬合真實的分布抽樣、**match 長尾(p95/p99)**；token 估計餵雜訊；不洩上帝視角。
- **綜效**：DR × 權重條件化 = 雙重穩健 contextual policy；真實後端 = 內建驗證儀。

---

## 7. LLM 控制層

- **角色**：只把「中文方針」翻成權重，不回答任何聊天。
- **技術**：Gemini 免費層**結構化輸出 / function calling**——事先給 schema，逼它填 JSON。
  ```json
  { "latency_weight": 0~1, "cost_weight": 0~1, "note": "一句話解讀" }
  ```
- **安全網（絕不盲信 LLM）**：驗證格式 → 權重夾進 [0,1] 並正規化 → 吐垃圾就退回上一組權重。
- **可插拔**：`ControlLLM.parse(text, w)→w`；`GeminiControl`（v1），`OllamaControl`（進階，全離線敘事）。
- **待定小分岔**（實作模組時談）：LLM 直接吐權重 vs 只吐「意圖」由程式換算（後者較穩可控，傾向之）。

---

## 8. 系統架構 / 模組邊界

```
web/(前端) ──WS──► server/(FastAPI) ──► control/(LLM 控制層)
                       │        │
                       ▼        ▼
                    agent/    sim/(Gym env)
                   (policy)      │
                       └─────────┴──► backends/(節點後端介面)
                                       ├ SimulatedBackend
                                       ├ OllamaBackend(4070 邊緣)
                                       └ GeminiBackend(雲端)
```

| 模組 | 職責 | 對外介面 | 依賴 | 技術 |
|---|---|---|---|---|
| `backends/` | 給請求→回 TTFT/成本 | `NodeBackend.infer(req)→InferResult` | httpx、ollama、gemini SDK | 可插拔後端＋fallback |
| `sim/` | 請求流+三節點排隊+Gym | `Env`(obs 15、Discrete(3)、reward) | 只依賴 `backends` 介面 | Gymnasium、NumPy、DR |
| `agent/` | 訓練 policy + 推論 | `Policy.predict(obs)→action` | `sim`、torch | **CleanRL PPO**(改權重條件化)；基準線同介面 |
| `control/` | 中文方針→驗證權重 | `ControlLLM.parse(text,w)→w` | gemini SDK | 結構化輸出+安全網 |
| `server/` | 即時模擬迴圈、推數據、接指令 | REST + WebSocket | `agent`/`sim`/`control` | FastAPI、asyncio |
| `web/` | 三面板儀表板、自我說明 | 連 WS 渲染 | 只跟 `server` | HTML/JS（或輕框架） |
| `calibration/`,`eval/` | 量真校準+gap；Pareto+AI vs 基準線出圖 | 腳本 | `backends`/`agent`/`sim` | 產圖 |

**邊界乾淨處**：`sim` 不知後端真假；`server` 不碰 PPO 內部；`web` 不知模擬細節；`control` 可整包抽換。各模組可獨立測試。

**資料流（即時 demo）**：請求→env 產 obs(含當前 w)→`Policy.predict`→選節點→`backend.infer`→TTFT/成本→env 更新算指標→server WS 推→前端渲染；同一請求流並行餵基準線比較。使用者打指令→server→`control.parse`→驗證 w→env 當前 w 更新→下一步行為即變。

---

## 9. 前端（展示台）需求

觀眾多為面試官/評審，60 秒內要自明：
- 大標題 + 一句話副標（秒懂主題）。
- 自動開跑或明顯 `▶ 開始`。
- 三面板：左＝即時分流儀表板、中＝AI vs 基準線兩條線賽跑（圖例「藍=你的 AI，灰=傳統」+ 領先%）、右＝中文方針對話框（放好範例提示字）。
- 情境按鈕（尖峰、省錢模式、重置）。
- hover tooltip 解釋數字；抽驗請求標「✅實測」。

---

## 10. 基準線與殺手級圖表

- **基準線**：貪婪(最快節點)、輪詢 round-robin、全雲、全本機——皆實作 `Policy.predict` 同介面，跑同一請求流。
- **圖表**：① AI vs 基準線的 TTFT/成本曲線（誰贏一目了然）；② 掃 w 的 **Pareto 前緣**；③ **gap 隨迭代下降**（Sim-to-Real 鐵證）。

---

## 11. 技術棧（0 元鐵則）

Python 3.11（含中文 `-X utf8`）、Gymnasium、NumPy、PyTorch、**CleanRL**（PPO 單檔，vendored）、httpx（async 串流量 TTFT）、Ollama（4070 邊緣）、google-generativeai（Gemini 雲端+控制層）、FastAPI + WebSocket、Docker、pytest、GitHub Actions、部署 GCP（參考作者 CV 專案流程）。

---

## 12. 建置順序

1. `backends` 介面 + `SimulatedBackend`
2. `sim` 環境跑通（reset/step，隨機策略印 TTFT/成本）
3. `agent` CleanRL 訓練 + 基準線 → **先拿到「AI 贏」**
4. `server` + `web` 左/中面板動起來
5. `control` 右面板（中文→權重）
6. 校準接真 Ollama/Gemini + Docker/CI/GCP 部署
7. 進階：LSTM 消融、Pareto 圖、gap 圖、（可選）OllamaControl 全離線

---

## 13. 鐵則 / 約束

- 🚫 **成本 0 元**：全開源；不接付費 API；LLM 走本機 Ollama 或 Gemini 免費層。
- 🔐 **隱私 / 公開前檢查**：repo 不得含論文封存量化數據、公司機密、個人/內部路徑。**交接檔 `_交接_給新對話.md` 屬私人工作文件，不進版控**（.gitignore）。API key 走 `.env`，不進版控。
- 📄 本專案數字皆來自公開 benchmark 或自訂模擬，天然安全，不使用碩論量化結果。
- 🧠 教學風格：白話+比喻+記憶鉤+邊做邊教；術語配真解釋、不硬翻怪詞。

---

## 14. 待定 / 開放項目

- 控制層「吐權重 vs 吐意圖」——寫 `control/` 模組時定。
- 節點延遲/成本的具體起始數字——建 `SimulatedBackend` 時填合理值，之後校準。
- 本機(手機)錨定的公開 benchmark 來源——建模型時挑。
- 前端框架（純 HTML/JS vs 輕框架）——做 `web/` 時定。
