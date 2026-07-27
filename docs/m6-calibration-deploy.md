# m6 — 校準 + 真後端 + 部署（Phase 6）

> 里程碑筆記，分 P6-1 ~ P6-4。

## 直觀理解（白話）

前面全在模擬裡跑。Phase 6 把模擬接上真實：邊緣節點用你的 4070 真跑 Ollama，量它真實
的 TTFT，再**回頭校準模擬參數**，讓「模擬預測」逼近「真實量測」。差距（gap）縮小，就是
「我的模擬真的越來越像真實」的鐵證——這是作品集第三個賣點。
記憶鉤：**先在便宜的模擬裡練，再拿真機量一量、把模擬調到像真的。**

---

## P6-1 — 真 OllamaBackend + 自動 fallback ✅

- `backends/ollama_backend.py`：`OllamaBackend`（真打本機 Ollama，streaming 量首字 TTFT、
  `is_measured=True`、成本仍用標價）、`ollama_available()` 探測、`build_edge_backend()`
  偵測不到 Ollama 自動 fallback 回 `SimulatedBackend`。
- 介面與 `SimulatedBackend` 一致 → `sim` 不知真假。真後端只用於校準/抽驗，不進訓練主迴圈
  （真生成慢）。
- `tests/test_ollama_backend.py`：4 個（真打測試 Ollama 未啟動時自動略過，CI 友善）。
- **實測**：`llama3.2` 在 4070 真的量到 TTFT ✅。

## P6-2 — 校準 + gap 指標 ✅

- `edge_llm_router/calibration.py`：`measure_ollama`（各輸入長度量 TTFT）、`fit_edge_params`
  （線性回歸擬合 rtt + prefill/token）、`gap_wasserstein`（模擬 vs 真實分布距離）、`calibrate`
  （回校準前後參數與 gap）。
- `calibration/run_calibration.py`：量真 → 擬合 → 出圖（`outputs/calibration_gap.png`）+ 存量測
  （`calibration/measurements/`）。
- `tests/test_calibration.py`：4 個（合成資料，不需 Ollama）。

**實測結果（12 筆真 Ollama 量測）**

| | rtt | prefill/token | gap（Wasserstein） |
|---|---|---|---|
| 校準前（placeholder 猜的） | 8ms | 1.500 | 2185 ms |
| 校準後（擬合真實） | ~2496ms | 0.075 | **61 ms** |

**gap 縮小 97%** —— 模擬被真實量測校準到幾乎吻合。（真實發現：此機 llama3.2 首字有 ~2.5s
固定開銷、但 prefill 斜率很平；校準誠實地捕捉了這個特性。之後可加更多重複量測 / 熱 keep_alive
精修絕對值。）

**DoD**：真打拿到真 TTFT、fallback 不崩 ✅；校準後 gap 大幅縮小（有數字）✅

---

## P6-3 — Gemini 雲端後端（可插拔）✅ 真呼叫已驗證

- `backends/gemini_backend.py`：`GeminiBackend`（lazy import `google-genai`、streaming 量首字、
  成本標價、`is_measured=True`）、`gemini_available()`（有 key + SDK 才算可用）、
  `build_cloud_backend()` 沒 key/SDK 自動 fallback 回 `SimulatedBackend`。
- 已裝 `google-genai` + `python-dotenv`；`server/app.py` 與腳本 `load_dotenv()` 自動讀 `.env`。
- `scripts/probe_gemini.py`：真打驗證腳本。
- `tests/test_gemini_backend.py`：2 個（有 key 時 fallback 測試自動 skip）。

**真呼叫驗證（放了 `GEMINI_API_KEY` 後）**
- `probe_gemini.py` 真打 3 次成功，實測 TTFT **~3300ms**（含網路 RTT + 免費層），成本標價 0.01。
- **型號雷**：`gemini-1.5-flash` 對此 key 回 404、`gemini-2.0-flash` 連打易 429（免費層 RPM 低）；
  改用 **`gemini-flash-latest`** OK。預設已設為它。
- **安全**：`.env` 已 gitignore；key 請勿貼整行 `echo`（會把指令文字寫進檔）。

- **DoD**：介面接好、沒 key 自動 fallback ✅；有 key 真打量到雲端 TTFT ✅。

**降級鏈（graceful degradation）**
- `backends/fallback.py`：`FallbackBackend` 依序試多後端、回第一個成功的。
- `build_cloud_backend` 組成鏈：**真 Gemini →（額度沒了/失敗）本機 Ollama →（再不行）模擬**。
  抽驗/探測永不卡在 Google 額度。
- 實測：Gemini 429（prepay credits 用盡）→ 自動退 Ollama → 仍量到真 TTFT ~2.5s。
- 註：Gemini 額度是 Google 端狀態、非程式問題；用光也不影響 demo（demo 走模擬）。

## P6-4 — Docker（本機 build 驗證）✅

- `Dockerfile`：多階段（`node:20` build 前端 → `python:3.12` 裝 uv + `uv sync --frozen
  --no-dev` + 複製 configs/checkpoints/web dist，跑 uvicorn）。
- `.dockerignore`：排除 .venv/node_modules/dist/mlruns/docs/tests/.env 等；**保留 checkpoints**。
- 容器內無 GPU/key → LiveSimulation 走純模擬 + PPO（可無 GPU、無 key 部署）。

**驗證（`docker build -t edge-llm-router .` + `docker run -p 8080:8000`）**
- build 成功（container 內 torch 2.13.0+cpu、前端 dist、模型 checkpoint 都打包）。
- `/health` → `{"status":"ok","ai_loaded":true}`（PPO checkpoint 在容器內載入）✅
- `/` 服務 React 前端 ✅；瀏覽器連 `localhost:8080`：WS 即時連線、三面板動起來、
  AI 領先 ~25% ✅——**整包 self-contained、可無 GPU/無 key 部署**。

**指令**
```
docker build -t edge-llm-router .
docker run --rm -p 8080:8000 edge-llm-router   # 開 http://localhost:8080
```

**GCP Cloud Run 部署 ✅**
- `.gcloudignore`：上傳 Cloud Build 前排除 `.env`/`.venv`/`node_modules` 等，**保留 checkpoints**
  （雲端有真 PPO 模型）。
- `gcloud run deploy edge-llm-router --source . --region asia-east1 --allow-unauthenticated
  --port 8000 --memory 2Gi --timeout 3600` → 一鍵 build+deploy。
- 線上：**https://edge-llm-router-735815297154.asia-east1.run.app**（`/health` ai_loaded=true、
  WebSocket 即時儀表板實測可用、AI 領先 ~27%）。閒置縮到零、無 GPU/key 也照跑。

---

## 即時抽驗（線上真打真實後端）✅

把「線上少量請求真打、標✅實測」做進**運行中的系統**（不再只有離線腳本）：

- `LiveSimulation.real_sample(node)`：對真實後端抽打一次量真 TTFT。真實後端 lazy build
  （第一次抽驗才探測）：edge → `build_edge_backend`（Ollama）、cloud → `build_cloud_backend`
  （Gemini→Ollama→模擬 降級鏈）。
- `server/app.py`：WS 迴圈每 `SAMPLE_EVERY`(~10s) 用 **`asyncio.to_thread`** 在**背景執行緒**
  抽驗（edge/cloud 輪流），**不阻塞**高速模擬主迴圈（真生成要數秒）。
- 快照多帶 `measured`；前端 `RoutingDashboard` 顯示「**✅ 實測 · 真打後端 <node> <ttft>ms**」
  徽章（`is_measured=False` 時標「退回模擬」）。

**實測（本機 Ollama 在線）**：edge 抽驗 → 真打 4070 Ollama ✅ ~2.6s；cloud 抽驗 → Gemini
額度沒了 → 降級鏈自動退 Ollama → 仍 ✅實測 ~2.5s。**線上（GCP 無 GPU/Ollama/key）則誠實
標「退回模擬」**——同一套機制、優雅降級。

**設計要點**：真打慢，所以只在**背景**抽驗、且真後端不進 RL 的 obs/訓練路徑（避免拖慢、也
避免洩上帝視角）；主迴圈仍是校準過的模擬，抽驗只是「摸一下真實、秀給觀眾看」。

---

## Phase 6 總結

sim-to-real 全數到位：邊緣真跑 Ollama（P6-1）、校準把 gap 壓低 97%（P6-2）、Gemini 可插拔
（P6-3）、整包 Docker 化 + **上 GCP Cloud Run**（P6-4）、**即時抽驗真實後端進運行系統**。
作品集第三賣點「Sim-to-Real 校準」有了硬證據，且產出一個線上可玩的公開 demo。
剩 Phase 7 進階圖表為加分項。
