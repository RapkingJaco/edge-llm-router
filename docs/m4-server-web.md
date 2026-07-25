# m4 — server + web：即時儀表板（Phase 4）

> 里程碑筆記，分 P4-1 ~ P4-3 逐步累積。

## 直觀理解（白話）

Phase 3 證明了「AI 贏」，但只是命令列的數字。Phase 4 把它變成**打開瀏覽器就看到的即時
表演**：後端一邊跑模擬、一邊把「這個請求 AI 丟哪、傳統丟哪、誰領先」串到前端畫出來。
記憶鉤：**後端是導播（送畫面），前端是螢幕（放畫面）。**

---

## P4-1 — server：即時模擬迴圈 + WebSocket ✅

**做了什麼**
- `server/simulation.py`：`LiveSimulation` — 兩條 `RouterEnv` 並行，AI 用 PPO（載
  `ppo_wc.pt`，載不到退 greedy）、對照用 greedy。**同 seed → 同一份工作負載**，逐 tick
  推進一個請求，累積雙方 reward/成本/丟棄/TTFT，產生快照。episode 跑完自動換下一局。
- `server/app.py`：FastAPI + WebSocket `/ws`，async 迴圈每 ~0.06s 送一個快照；接指令
  `{"cmd":"reset"}` / `{"cmd":"peak","on":true}`。另有 `/health`。
- `tests/test_server.py`：6 個測試（LiveSimulation 邏輯 + WS 煙霧 via TestClient）。

**快照格式（給前端）**
```
{ t, w, peak, lead_pct, ai_loaded,
  ai_utils:{local,edge,cloud}, base_utils:{...},
  ai:{name,node,dropped,cum_reward,cost,served,drops,avg_ttft_ms},
  base:{...} }
```

**設計選擇**
- 兩條 env 同 seed → 公平（同一請求流），只是路由不同。
- `RouterEnv.peek_utilizations()`：只在要廣播時算使用率，不加到訓練每步的成本。
- 尖峰模式：拉高 `arrival_rate_base`（deepcopy config，不動原始），立即換一局生效。
- 每個 WS 連線一個獨立 `LiveSimulation`（多人看各自的）。

**DoD**：`tick()` 回雙 lane 快照、同請求流公平 ✅；TestClient 連 WS 收得到、reset 有效 ✅；
`/health`、`/ws` 就緒 ✅（全套 42 passed）

**啟動**：`uv run uvicorn edge_llm_router.server.app:app --reload`

---

## P4-2 — web：React 骨架 + WS client + 左面板 ✅

**做了什麼**
- Vite + React + TypeScript 骨架（`web/`）：`package.json`、`vite.config.ts`、`tsconfig.json`、
  `index.html`、`src/main.tsx`、`App.tsx`、`styles.css`。
- `src/types.ts`：對應 server 快照的型別。
- `src/api.ts`：`RouterSocket` 型別化 WS client + 斷線自動重連；同源走 host、dev(5173) 連 8000。
- `src/components/RoutingDashboard.tsx`：左面板——三節點即時使用率長條（滿載變紅）、當前
  請求高亮、AI lane 已處理/丟棄/成本/TTFT。
- `server/app.py`：有 `web/dist` 就由 FastAPI 掛 `StaticFiles` 服務前端（單 port demo/部署）。

**驗證（瀏覽器實測 http://localhost:8000）**
- WS「即時連線」、w 顯示 0.50/0.50、三節點使用率即時跳動、stats 累積（已處理 206、成本
  1.41、TTFT 240ms）、無 console error。
- cost 1.41 遠低於 greedy ~5 → 確認前端接的是真的省成本 PPO。

**踩到的坑**
- 這台 npm 是強化設定，預設**不跑 install script**（esbuild 的 postinstall 被擋，有 warning）；
  本次 `vite build` 仍成功。若日後 build 抱怨 esbuild 二進位缺失，用 `npm approve-scripts` 放行。
- `node_modules/`、`dist/` 已 gitignore；`package-lock.json` 進版控。

**DoD**：`npm run build` 型別檢查 + 打包過 ✅；連上後端 WS 即時更新（不用重整）✅；左面板顯
示三節點負載、看得出壅塞 ✅

**啟動**：`npm --prefix web run build` 後 `uv run uvicorn edge_llm_router.server.app:app`
→ 開 http://localhost:8000（dev 熱重載：另跑 `npm --prefix web run dev` 於 5173）

## P4-3 — web：中面板（AI vs 基準線賽跑）+ 情境按鈕 ✅

**做了什麼**
- `components/RaceChart.tsx`：手繪 SVG 兩線（藍=AI/灰=傳統），累積 reward 從 0 往下走、
  上緣=0=完美；大字顯示「AI 領先 X%」（綠=領先/紅=落後）+ 圖例。**不加圖表庫依賴**。
- `components/ScenarioControls.tsx`：製造尖峰（toggle）/重置，送 WS 指令。
- `App.tsx`：維護賽跑歷史（新局自動清空）、`send()` 送指令。

**驗證（瀏覽器實測）**
- 中面板顯示「▲ 30.7% AI 領先」——與 Phase 3 w=0.5 的 ~28% 一致（活的證據）。
- 按「製造尖峰」→ 鈕變「尖峰中 ✓」、立即換高到達率一局、edge 衝到 100%、領先%重算。
- 三面板 + 標題副標，60 秒自明；無 console error。

**DoD**：兩線隨時間拉開、領先%正確 ✅；按尖峰看得出壅塞 ✅；60 秒自明、瀏覽器實測 ✅；
`docs/m4-server-web.md` 寫好 ✅

---

## Phase 4 總結

命令列的「AI 贏」變成**打開瀏覽器就看到的即時表演**：左面板即時分流、中面板 AI vs 傳統
賽跑（活生生領先 ~30%）、情境按鈕製造尖峰。後端 FastAPI+WS 一個 port 服務前端 build。
下一步 Phase 5：右面板——用**中文下方針**即時改 w，讓行為零重訓當場切換。
