# edge-llm-router 實作計畫

> 依據設計規格 [`docs/superpowers/specs/2026-07-24-edge-llm-router-design.md`](../specs/2026-07-24-edge-llm-router-design.md) 展開的分階段實作計畫。
> 建立：2026-07-24。作者：Jacob（陳品榤 / RapkingJaco）。
> 用途：把「已定案的設計」拆成可一步步做、每步有明確驗收標準的路線圖。

---

## 0. 怎麼讀這份計畫

- **先結論**：照 7 個 Phase 走，每個 Phase 結束都有「跑得起來、看得到東西」的里程碑。**Phase 3（AI 贏基準線）是靈魂**，前面都在鋪路。
- 每個 Phase 有：**目標 → 要建的檔案/介面 → 做法重點 → 驗收標準（DoD）**。
- 📌＝待定項的建議決定　🧠＝教學鉤（新概念白話解釋）。

---

## 1. 環境現況（2026-07-24 實測，這台家用機）

| 工具 | 現況 | 計畫採用 |
|---|---|---|
| Python | 未安裝（只有 Windows Store 假指令） | **用 uv 管 Python（實際用 3.12）** |
| uv | ✅ 0.11.27 | 管 Python 版本 + venv + 套件 |
| Node.js | ✅ 在 PATH | 前端 Vite/React/TS build |
| Ollama | ✅ 0.32.1，已有 `llama3.2:latest`(2GB) | 邊緣後端真跑，模型現成 |
| GPU | ✅ RTX 4070 SUPER / 12GB | 跑 Ollama；PPO 小 MLP 用 CPU 就夠 |
| Docker | ✅ 29.6.1 | Phase 6 容器化 |
| gcloud | ✅ 在 PATH | Phase 6 GCP 部署 |

> Python 版本註記：原訂 3.11，但 uv 0.11.27 在此機安裝 3.11 卡在建 link 步驟；改用已就緒的 **3.12**，全部套件相容，對專案無影響。

🧠 **PPO 訓練用 CPU 就好**：policy 只是十幾維輸入的小 MLP，運算量極小；GPU 價值在跑 llama3.2。把 GPU 留給 Ollama，訓練走 CPU，少一堆 CUDA 地雷。

---

## 2. 全域決策（spec 第 14 節待定項就地拍板）

| 待定項 | 📌 決定 | 理由 |
|---|---|---|
| 套件/環境管理 | **uv + pyproject.toml，Python 3.12** | 這台沒 Python；uv 一條龍、lockfile 可重現 |
| 專案佈局 | **`src/edge_llm_router/` 套件佈局**，子模組職責分明，頂層 `config.py`/`metrics.py` | import 乾淨、可測試、當真套件寫 |
| 設定管理 | **`configs/default.yaml` + `config.load_config()`** | 可調參數集中一檔，校準只改一處 |
| 控制層「吐權重 vs 吐意圖」 | **吐意圖**（LLM 回 `{intent, magnitude}`，程式換算 w） | 較穩可控、好加安全網 |
| 前端框架 | **Vite + React + TypeScript** | 元件化、質感好、作品集加分 |
| 實驗追蹤 | **MLflow**（追 PPO 訓練 reward/超參/曲線） | 訓練可回溯；`mlruns/` 已 gitignore |
| 本機延遲錨定 | 建 `SimulatedBackend` 時挑公開行動端 LLM benchmark，寫死＋註明出處 | spec 允許公開 benchmark |
| 節點起始數字 | 填有物理意義的合理值，集中放 `configs/default.yaml` | Phase 6 校準改一處 |
| 成本標價倍率 | 雲端 ≈ 本機 100×、邊緣 ≈ 5~10× | 逼出「不能全丟雲」的張力 |

---

## 3. 套件清單（Phase 0 寫進 pyproject.toml）

```
# 核心 RL / 模擬
gymnasium, numpy, torch (CPU wheel，綁 PyTorch CPU index)
# 真實後端 / 網路
httpx, ollama
# 服務
fastapi, uvicorn[standard], websockets, pydantic
# 設定 / 追蹤 / 出圖
pyyaml, mlflow, matplotlib
# 開發
pytest, pytest-asyncio, ruff
```
- Gemini SDK（`google-genai` vs `google-generativeai`）延到 Phase 5 建 `control/` 時決定，屆時再加。
- 前端（`web/`，走 npm）：`react, react-dom, typescript, vite, @vitejs/plugin-react, oxlint` + 一個輕量圖表方案。

---

## 4. 目標目錄結構（做完 Phase 5 的樣子）

```
edge-llm-router/
├─ pyproject.toml / uv.lock / .python-version(3.12)
├─ .env.example
├─ CLAUDE.md                        # 專案給 Claude 的指示
├─ Dockerfile / .dockerignore / .gcloudignore / .gitattributes   # Phase 6
├─ configs/default.yaml             # 節點參數/正規化基準/DR 範圍/超參
├─ .github/workflows/ci.yml,deploy.yml   # pytest+ruff / GCP
├─ scripts/
│  ├─ check_env.py                  # 環境檢查
│  └─ random_rollout.py             # Phase 2 隨機策略跑一局
├─ src/edge_llm_router/
│  ├─ config.py                     # 讀 configs/default.yaml
│  ├─ metrics.py                    # TTFT/成本/reward/領先% 指標
│  ├─ backends/  base.py simulated.py ollama_backend.py gemini_backend.py
│  ├─ sim/       env.py workload.py
│  ├─ agent/     ppo.py policy.py baselines.py
│  ├─ control/   base.py gemini_control.py
│  └─ server/    app.py             # FastAPI + WebSocket
├─ web/                             # Vite+React+TS
│  ├─ package.json / vite.config.ts / tsconfig*.json / .oxlintrc.json
│  ├─ index.html
│  └─ src/  main.tsx App.tsx api.ts types.ts
│     └─ components/ RoutingDashboard.tsx RaceChart.tsx PolicyConsole.tsx
├─ eval/  compare.py pareto.py gap.py eval_gallery.py   # → JSON + HTML
├─ outputs/                         # eval 結果（大檔 gitignore）
├─ calibration/                     # Phase 6 量真校準
├─ docs/
│  ├─ foundations.md concepts.md tech-stack.md environment.md python-notes.md
│  ├─ m0-scaffold.md … m7-advanced.md   # 每 Phase 一份里程碑筆記
│  ├─ img/
│  └─ superpowers/{specs,plans}/
└─ tests/
```

🧠 **`src/` 佈局的好處**：把套件關進 `src/` 可避免測試時 import 到當前資料夾而非安裝的套件，也逼你當成真的可安裝套件在寫。

📄 **文件庫工作法**：每做完一個 Phase，同步寫一份 `docs/mN-*.md`（做了什麼、為什麼這樣設計、踩到什麼坑、學到的觀念）；跨 Phase 的通用觀念沉澱到 `concepts.md`/`foundations.md`。這對「邊做邊教」風格是核心，也是面試時的深度證據。

🖥 **前端要點**：`node_modules/`、`dist/` 一律 gitignore；三面板拆成 `RoutingDashboard`/`RaceChart`/`PolicyConsole` 各自單一職責；WebSocket client 抽成型別化的 `api.ts`（含重連）；明暗主題 + 無障礙 + 響應式；圖表用輕量方案（自繪 canvas 或單一小型圖表庫），守成本節制、少依賴。

---

## Phase 0 — 骨架 + 環境 + 工程模板（1 天）

**目標**：能 `uv run pytest` 綠燈、能 import 的空殼；工程模板一次到位。

**要建**：
- `uv init` → `pyproject.toml`、`.python-version`；第 3 節套件 `uv add`（torch CPU wheel）
- 第 4 節所有模組資料夾 + 空 `__init__.py`
- `configs/default.yaml`（先放骨架鍵）+ `src/.../config.py` 讀取
- `scripts/check_env.py`（檢查 Python/torch/ollama/gpu/gemini key，印報告）
- `.github/workflows/ci.yml`（`uv run ruff check` + `uv run pytest` + `check_env.py`）
- `CLAUDE.md`
- `docs/` 觀念檔骨架 + `m0-scaffold.md`
- `.env.example`、`tests/test_smoke.py`

**DoD**：
- [x] `uv run python -c "import edge_llm_router"` 無錯
- [x] `uv run pytest`、`uv run ruff check` 綠燈
- [x] `uv run python scripts/check_env.py` 印出環境報告
- [ ] CI 在 push 後綠燈（待首次 push）

---

## Phase 1 — `backends/` 介面 + SimulatedBackend（1~2 天）

**目標**：定義節點後端抽象；實作**會因忙碌變慢**的模擬後端。

**要建**：
- `backends/base.py`：`Request`、`InferResult`（`ttft_ms/cost/node/is_measured/dropped`）、`NodeBackend`(ABC) `infer(req)->InferResult` + `utilization/queue_len/est_wait_ms`
- `backends/simulated.py`：`SimulatedBackend`，參數全讀 `configs/default.yaml`
- `metrics.py`：TTFT/成本正規化、統計聚合（Phase 0 已建骨架）

**節點模型（spec 第 5 節落地）**：
```
TTFT ≈ 排隊等待 + 網路RTT + prefill;  prefill_ms ≈ prefill_ms_per_token × input_tokens
排隊等待 = f(當前併發 / 容量)   # 越接近容量非線性上升
```
起始參數（📌 放 YAML）：local RTT~0/prefill 高/容量低/成本~0；edge 中；cloud RTT 大/prefill 低/容量高/成本×100。

🧠 **為何一定要「忙→變慢」**：延遲固定的話最佳解永遠是「全丟最快的」，RL 無事可學。排隊模型製造「便宜節點一擠就爆」的張力，分流才有意義——這是問題的靈魂。

**DoD**：
- [x] 單元測試：越忙 TTFT 越大；cloud 成本 ≈ local 的 100×
- [x] 塞爆容量 → `dropped=True`/等待暴增
- [x] `SimulatedBackend` 介面與未來 Ollama/Gemini 一致（自己不知是假的）
- [x] `docs/m1-backends.md` 寫好

---

## Phase 2 — `sim/` Gym 環境跑通（2~3 天）

**目標**：合法 Gymnasium 環境，`reset()/step()` 跑通，隨機策略印 TTFT/成本。

**要建**：
- `sim/workload.py`：變動到達率（尖峰）、token 分布（短/中/長）、prompt 池
- `sim/env.py`：`RouterEnv`（obs Box ~15 維正規化、`Discrete(3)`、reward per spec）
  - **權重條件化**：每 episode `reset()` 隨機抽 `w=(w_lat,w_cost)`（和=1），塞進 obs 末 2 維
  - **不洩上帝視角**：obs 只放真實可得量；token 估計加雜訊
- `scripts/random_rollout.py`

🧠 **權重條件化（核心 DL 技術）**：把「你多在乎延遲 vs 成本」的權重 `w` 當**環境狀態**餵給 agent，訓練每局隨機換 w，學成**一整族** `π(a|s,w)`。同一網路改 w 行為就變 → demo 時 LLM 改 w **零重訓**切換「省錢/極速」模式。學名 goal-conditioned RL / 多目標 scalarization；一句話記：**把偏好當輸入，一網打盡整條偏好光譜**。

**DoD**：
- [x] `env_checker.check_env(RouterEnv())` 通過
- [x] `random_rollout.py` 跑完一局印平均 TTFT/總成本/丟棄數
- [x] 換不同 w，reward 懲罰項確實改變；obs 每維都在界內
- [x] `docs/m2-sim.md` 寫好

---

## Phase 3 — `agent/` PPO + 基準線 →「AI 贏」🏆（3~5 天，最重要）

**目標**：PPO policy 在**同一請求流**上、**跨整條 w 光譜**贏過所有基準線。核心證據。

🧠 **CleanRL 為何適合作品集**：PPO 寫成**單檔、無層層抽象**，面試可直接指程式講 advantage/clip；SB3 把細節藏起來，秀 RL 功力吃虧。

依賴：**P3-1 → P3-2 → P3-3 → P3-4**。前兩個不碰 PPO、快；後兩個是訓練（CPU、不用網路）。

### P3-1 — 基準線 + Policy 介面 ✅
- **要建**：`agent/policy.py`（`Policy.predict(obs)->action` 統一介面）、`agent/baselines.py`
  （`Greedy` 挑最快 / `RoundRobin` / `AllCloud` / `AllLocal`）、`agent/runner.py`（`run_episode`）
- **DoD**：
  - [x] 4 條基準線各跑完整局；AllLocal 96% 丟棄、AllCloud/greedy 高成本、round_robin 42% 丟棄
  - [x] `tests/test_baselines.py` 綠燈（全套 29 passed）

### P3-2 — Eval 對照（control group）✅
- **要建**：`edge_llm_router/evaluation.py`（引擎）+ `eval/compare.py`（薄 CLI）→ 同一請求流
  + 同一組 `w` 公平比 4 條基準線，跨 w_cost=0.1/0.5/0.9，產 JSON + HTML（進 `outputs/`）
- **DoD**：
  - [x] 對照表（每策略 drop率 / TTFT / 成本 / reward），看得出各自取捨
  - [x] 固定 workload + w 餵所有策略（公平性）
  - 洞察：基準線皆不看 w；greedy 靠燒錢勝出，w_cost=0.9 是 PPO 破口

### P3-3 — PPO 訓練（固定 w 先跑通）✅
- **要建**：`agent/ppo.py`（CleanRL 風格單檔 PPO）、MLflow（粗粒度）、`scripts/train_ppo.py`、
  存 checkpoint（`.pt`）
- **DoD**：
  - [x] MLflow 訓練曲線收斂（粗粒度約 40 點）
  - [x] PPO(固定 w=0.5) **贏過全部基準線**（-211.5 vs greedy -283.7，0% 丟棄、成本省 29%）；
    policy 可存/載

### P3-4 — 權重條件化 + 「AI 贏」🏆 ✅
- **要建**：`fixed_w=None` 訓練（完整權重條件化）、`PPOPolicy`、`scripts/train_wc.py`、
  `outputs/eval_ppo_vs_baselines.*`
- **DoD**（里程碑，標準要硬）：
  - [x] PPO **跨整條 w 光譜勝全部基準線**（w=0.1/0.5/0.9 分別贏 greedy ~8%/28%/39%）
  - [x] 重延遲 w → 82% 雲；重成本 w → 42% edge（權重條件化生效、零重訓切換）
  - [x] `docs/m3-agent.md` 寫好

**Phase 3 ✅ 完成**——RL 靈魂里程碑達成。

---

## Phase 4 — `server/` + `web/` 左中面板（3~4 天）

**目標**：瀏覽器看到**即時分流**與**AI vs 基準線賽跑**。依賴 P4-1 → P4-2 → P4-3。

### P4-1 — server：即時模擬迴圈 + WebSocket ✅
- **要建**：`server/simulation.py`（`LiveSimulation`）、`server/app.py`（FastAPI + WS +
  `/health`）、`RouterEnv.peek_utilizations()`、載入 `ppo_wc.pt`
- **DoD**：
  - [x] `LiveSimulation.tick()` 回傳雙 lane 快照；同請求流公平
  - [x] TestClient 連 WS 收得到快照、送 reset 有效；`tests` 綠燈（全套 42 passed）

### P4-2 — web：React 骨架 + WS client + 左面板 ✅
- **要建**：Vite+React+TS（`api.ts` 型別化 WS + 重連、`types.ts`）、`RoutingDashboard.tsx`；
  FastAPI 掛 `StaticFiles` 服務 `web/dist`
- **DoD**：
  - [x] `npm run build` 過、FastAPI 服務、瀏覽器連上後端 WS 即時更新（不用重整）
  - [x] 左面板顯示三節點負載、看得出壅塞（瀏覽器實測 localhost:8000，無 console error）

### P4-3 — web：中面板（AI vs 基準線賽跑）+ 情境按鈕 ✅
- **要建**：`RaceChart.tsx`（手繪 SVG 兩線 + 領先%）、`ScenarioControls.tsx`（尖峰/重置）、
  App 賽跑歷史
- **DoD**：
  - [x] 中面板兩線隨時間拉開、領先% 正確（瀏覽器實測 ▲30.7%，對上 Phase 3 ~28%）
  - [x] 按「尖峰」看得出壅塞（edge 衝 100%）；60 秒自明；瀏覽器實測；`docs/m4-server-web.md` 寫好

**Phase 4 ✅ 完成**——即時三面板儀表板（左/中）動起來。

---

## Phase 5 — `control/` 右面板：中文方針 → 權重（2~3 天）

**目標**：打中文（「成本太高，多用本機」），行為**零重訓當場改變**。

**已做（選項 1：規則版先行，真 Gemini 可插拔）**：
- `control/base.py`：`ControlLLM.parse`、`intent_to_weights`、安全網
- `control/rule_based.py`：`RuleBasedControl`（關鍵字→意圖→權重；離線免 key）
- `RouterEnv.set_w()` 即時改 w；`server` `policy` 指令 + 快照 `note`；`web` `PolicyConsole.tsx`
- 真 `GeminiControl` 為可插拔選項（放了 `GEMINI_API_KEY` 再實作，介面不變）

🧠 **「吐意圖」比「吐權重」穩**：讓 LLM 直接吐 `w_lat=0.83` 易亂給/超界/抖動；讓它只做擅長的「判斷語氣意圖」，把「意圖→數字」交給你的確定性程式，多一層可控可驗證——這就是「絕不盲信 LLM」。

🧠 **結構化輸出**：事先給 JSON schema **逼 LLM 只能填格子**，回來保證可 parse，不用正則去撈——把 LLM 接進正式系統的標準做法。

**DoD**：
- [x] 「成本太高，多用本機」→ w=(0.1,0.9)、更多 local/edge；「我要最快」→ w=(0.9,0.1)、更多 cloud（瀏覽器實測）
- [x] 亂輸入/空白不崩、維持原方針；全程**無重訓**
- [x] `docs/m5-control.md` 寫好

**Phase 5 ✅ 完成**（規則版；真 Gemini 待 key 可插拔接上）。

---

## Phase 6 — 校準 + 真實後端 + Docker/CI/GCP（4~6 天）

**目標**：接真 Ollama/Gemini、真實量測校準模擬、可部署。

**要建**：
- `backends/ollama_backend.py`（本機 llama3.2，async 串流量真 TTFT）、`backends/gemini_backend.py`（真延遲、成本仍標價）；偵測不到 → 自動 fallback 模擬
- 抽驗機制：洪流走模擬，少量真打標 `is_measured=True`，前端顯示「✅實測」
- `calibration/`：量真 → 回歸擬合 `configs/default.yaml` 參數；DR 範圍取自實測變異
- `eval/gap.py`：模擬 vs 真實分布落差（KL/Wasserstein）
- Docker + `.github/workflows/{ci,deploy}.yml` + GCP 部署（雲端節點線上真跑，本機/邊緣退模擬）

🧠 **Sim-to-Real 一句話**：真硬體跑不了幾百萬步訓練，所以在**校準過的模擬**裡訓練、再確保它夠像真實。DR 讓 agent 在「一整片可能的真實」裡訓練，真機值落在見過的範圍內就不水土不服。「gap 隨迭代下降」圖是你證明模擬越來越像真實的鐵證。

**分支進度**：
- **P6-1** ✅ `OllamaBackend` 真打 llama3.2 + `build_edge_backend` 自動 fallback（實測到真 TTFT）
- **P6-2** ✅ `calibration.py` 量真→擬合→gap；實測 gap 2185ms→61ms（縮小 97%），出圖
- **P6-3** ✅ `GeminiBackend` 可插拔 + **真呼叫已驗證**（放 key 後真打量到 ~3300ms；
  型號用 `gemini-flash-latest`；沒 key 自動 fallback）
- **P6-4** ✅ Dockerfile 多階段 build；**已上 GCP Cloud Run**（asia-east1，公開網址、WS 即時
  連線實測可用、閒置縮到零）：https://edge-llm-router-735815297154.asia-east1.run.app

**DoD**：
- [x] 真打 Ollama 拿到真 TTFT；停掉 Ollama 自動 fallback 不崩
- [x] 校準後 sim 預測 TTFT 分布與實測 gap 縮小（2185→61ms）
- [x] Gemini 後端接好、沒 key 自動 fallback（真抽驗待 key）
- [x] `docker build` 成功、container 跑起來服務前端 + WS（GCP push 待帳號）

**Phase 6 ✅ 完成**（sim-to-real 全到位；GCP 實際上線待 gcloud 帳號）。

---

## Phase 7 — 進階（有時間再做，各自獨立）

- [x] **P7-1** `eval/pareto.py`：掃 w 畫 Pareto 前緣（一 policy 掃出整條前緣、0% 丟棄、
  壓過 greedy/all_cloud 固定點）✅
- [x] **P7-2** `calibration/gap_iterations.py`：gap 隨校準迭代 2286→44ms 收斂（真 Ollama）✅
- [ ] LSTM 消融（選配；預期驗證「obs 已是充分統計量、記憶幫助有限」）
- [ ] OllamaControl（選配；控制層走本機 Ollama 全離線）
- [ ] `eval/eval_gallery.py`（選配；圖表匯總單頁）
- [x] `docs/m7-advanced.md` ✅

**Phase 7（P7-1、P7-2）✅ 完成**——殺手級圖表到位。

---

## 5. 里程碑總覽（面試故事線）

| 里程碑 | Phase | 一句話 |
|---|---|---|
| 環境會忙會塞 | 1–2 | 「三節點排隊推論環境」 |
| **AI 贏基準線** 🏆 | 3 | 「RL agent 分流打贏所有傳統策略」 |
| 即時看得到 | 4 | 「網頁看 AI vs 傳統即時賽跑」 |
| 一句話改行為 | 5 | 「中文下令，零重訓即時切換目標」 |
| 真機驗證 | 6 | 「邊緣真跑 Ollama，模擬被真實校準」 |
| 壓箱圖表 | 7 | 「Pareto 前緣 + Sim-to-Real gap 收斂」 |

---

## 6. 隱私 / 公開前檢查（每次 push 前）

- `.env`（GEMINI_API_KEY）不進版控（已 gitignore）。
- 交接檔 `_交接_繼續開發.md` **目前在版控中、會被 push**；repo 轉 public 前需確認/移除（含個人/內部資訊）。工作機版 `_交接_給新對話.md` 已 gitignore。
- `node_modules/`、`dist/`、`mlruns/` 大檔確認都在 .gitignore。
- 不使用碩論量化數據；數字皆來自公開 benchmark 或自訂模擬。

---

## 7. 下一步（立即可做）

**Phase 0（骨架 + 工程模板）✅ → Phase 1（backends）→ Phase 2（sim）**，先讓 `env.reset()/step()` 跑通、隨機策略印得出 TTFT/成本，再進 Phase 3 拚「AI 贏」。
