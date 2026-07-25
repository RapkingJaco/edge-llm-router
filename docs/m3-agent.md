# m3 — agent：PPO + 基準線 →「AI 贏」（Phase 3）

> 里程碑筆記，分 P3-1 ~ P3-4 逐步累積。

## 直觀理解（白話）

Phase 2 把分流做成一台「電動遊戲」，Phase 3 讓玩家上場：先寫幾個**規則寫死的傳統玩家**
（基準線），再訓練一個**會自己學**的玩家（PPO）。目標是證明「學出來的」明顯打敗「寫死的」，
而且能隨 w（在乎快 vs 在乎省）即時換打法。
記憶鉤：**基準線是對照組，PPO 是實驗組；差距 = 作品集要秀的成果。**

---

## P3-1 — 基準線 + Policy 介面 ✅

**做了什麼**
- `agent/policy.py`：`Policy` 抽象（`predict(obs)->action` + 可選 `reset()`）。PPO 與基準線
  都實作它，才能共用 runner / eval。
- `agent/baselines.py`：
  - `Greedy` — 最小預估等待，平手偏好較快節點（cloud>edge>local）
  - `RoundRobin` — 輪流本機→邊緣→雲
  - `AllCloud` / `AllLocal` — 永遠丟同一節點
- `agent/runner.py`：`run_episode(env, policy, seed)` → episode 摘要（同 seed = 同工作負載
  + 同 w，比較才公平）。
- `tests/test_baselines.py`：7 個測試。

**基準線表現（同一請求流，w=0.5/0.5，seed=42）**

| 策略 | 丟棄率 | 平均TTFT | 總成本 | 總reward |
|---|---|---|---|---|
| greedy | 0.0% | 224ms | 4.94 | -275 |
| all_cloud | 0.0% | 239ms | 5.07 | -284 |
| round_robin | 42.0% | 1125ms | 1.80 | -674 |
| all_local | 96.3% | 4929ms | 0.002 | -1229 |

**解讀**
- greedy ≈ all_cloud：greedy 節點空時挑最快 = cloud，幾乎全丟雲 → 0 丟棄、快，但**燒錢**。
- all_local 96% 丟棄：全塞容量 2 的本機幾乎全爆。
- **PPO 的機會**：w 偏省錢時，用 local/edge 聰明分流做到低丟棄又省錢——權重條件化的價值點。

**DoD**：4 條各跑完整局、行為符合預期 ✅；`tests/test_baselines.py` 綠燈 ✅（全套 29 passed）

---

## P3-2 — Eval 對照（control group）✅

**做了什麼**
- `edge_llm_router/evaluation.py`：評估引擎（`evaluate` / `to_console` / `render_html` /
  `save`）。放套件裡可 import、可測；`eval/compare.py` 只是薄 CLI（`eval` 這名字會遮蔽
  內建 `eval`，故邏輯不放那）。
- 跨 w_cost = 0.1 / 0.5 / 0.9、每個 5 個 seed，公平比 4 條基準線 → `outputs/eval_baselines.{json,html}`。
- `tests/test_evaluation.py`：4 個測試。

**結果（5 seed 平均；每策略在三個 w 下 drop/TTFT/成本相同，只 reward 隨 w 變）**

| 策略 | 丟棄率 | 平均TTFT | 總成本 | reward @0.1 | @0.5 | @0.9 |
|---|---|---|---|---|---|---|
| greedy | 0.0% | 214ms | 5.11 | **-101.7** | **-283.7** | **-465.7** |
| all_cloud | 0.0% | 224ms | 5.26 | -105.6 | -292.4 | -479.1 |
| round_robin | 40.7% | 1285ms | 1.87 | -750.7 | -690.4 | -630.0 |
| all_local | 95.9% | 4713ms | 0.002 | -1481 | -1271 | -1061 |

**洞察（PPO 的價值主張）**
- 基準線都**不看 w**：routing 固定，只 reward 隨 w 變。傳統策略無法隨偏好調整。
- greedy 每次贏，但靠「狂燒錢」（成本 5.11 ≈ 全丟雲）；round_robin 便宜卻 40% 丟棄輸掉。
- **破口在 w_cost=0.9**：greedy 掉到 -465（省不了錢）。會看 w 的 PPO 應能在「要省」時
  多用 local/edge、壓低成本又不爆 → 這是權重條件化的殺手級價值。

**DoD**：對照表看得出取捨 ✅；固定 workload + w 公平比較 ✅；JSON+HTML 產出 ✅

## P3-3 — PPO 訓練（固定 w 先跑通）✅

**做了什麼**
- `agent/ppo.py`：CleanRL 風格單檔 PPO（`PPOAgent` actor-critic MLP、`train()`、`PPOPolicy`、
  `save_agent`/`load_agent`）。自管數個 env + 立即重置，GAE done 記帳可控、跨 gymnasium
  版本穩定。訓練走 CPU。
- MLflow 記訓練曲線/超參（粗粒度，見下方踩坑）。
- `scripts/train_ppo.py`：固定 w=0.5 訓練 → 存 checkpoint → 和基準線比。
- `tests/test_ppo.py`：極短訓練煙霧測試。

**結果（200k timesteps，w=0.5/0.5，5 seed 平均）**

| 策略 | 丟棄率 | 平均TTFT | 總成本 | reward |
|---|---|---|---|---|
| **ppo** | 0.0% | 231ms | **3.62** | **-211.5** 🏆 |
| greedy | 0.0% | 214ms | 5.11 | -283.7 |
| all_cloud | 0.0% | 224ms | 5.26 | -292.4 |
| round_robin | 40.7% | 1285ms | 1.87 | -690.4 |
| all_local | 95.9% | 4713ms | 0.002 | -1271.2 |

**PPO 學到什麼**：維持 0% 丟棄（跟 greedy 一樣穩），但成本 3.62 vs greedy 5.11 —— **省 29%**。
它學會「cloud 不是唯一解」：edge/local 有餘裕時分過去省錢，又精準地不塞爆。greedy 只會無腦
衝最快（全雲燒錢），PPO 會權衡。**PPO 已在 w=0.5 贏過全部基準線。**

**踩到的坑**
- **MLflow 拖慢訓練**：檔案後端每次 log 都 fsync（Windows 慢），每更新都寫 → 20k 從 10s
  暴增到 97s。改成**粗粒度記錄**（約 40 個點），200k 訓練回到 ~98s。
- 純訓練速度 ~1960 steps/s（CPU，num_envs=4）。

**DoD**：訓練收斂、reward 上升（MLflow 有曲線）✅；PPO 贏隨機/輪詢/全本機、且**贏過
greedy** ✅；checkpoint 可存/載 ✅

> 看訓練曲線：`uv run mlflow ui` 後開瀏覽器（experiment: edge-llm-router-ppo）。

---

## P3-4 — 權重條件化 + 「AI 贏」🏆 ✅

**做了什麼**
- 用 `fixed_w=None`（每局隨機 w）訓練 → 權重條件化 policy（`train()` 程式不變，只換參數）。
- `scripts/train_wc.py`：訓練 → 存 `ppo_wc.pt` → 跨 w 光譜比 PPO vs 基準線 → 印 w 響應。
- `outputs/eval_ppo_vs_baselines.{json,html}`。

**結果：PPO 在每個 w 都贏，且越在乎成本贏越多（reward，越接近 0 越好，5 seed 平均）**

| w_cost | PPO | greedy | all_cloud | round_robin | all_local |
|---|---|---|---|---|---|
| 0.1 要快 | **-93.1** | -101.7 | -105.6 | -750.7 | -1481.2 |
| 0.5 平衡 | **-203.3** | -283.7 | -292.4 | -690.4 | -1271.2 |
| 0.9 要省 | **-286.2** | -465.7 | -479.1 | -630.0 | -1061.1 |

贏的幅度：w=0.1 贏 ~8% → w=0.5 贏 ~28% → w=0.9 贏 ~39%。

**w 響應（同一 policy、零重訓，只改 w，單局分流分布）**

| 方針 | local | edge | cloud |
|---|---|---|---|
| w_cost=0.1 要快 | 0% | 18% | 82% |
| w_cost=0.9 要省 | 4% | 42% | 54% |

告訴它「要快」→ 猛用雲；「要省」→ 把 42% 移到 edge。這就是權重條件化的殺手級展示：
**一個網路覆蓋整條偏好光譜，改 w 即時換行為、不用重訓。**

**DoD**：PPO 跨整條 w 光譜勝全部基準線 ✅；重延遲→偏快節點、重成本→偏便宜節點 ✅；
`docs/m3-agent.md` 寫好 ✅

---

## Phase 3 總結

RL 靈魂里程碑達成：從「隨機亂丟 40% 丟棄、reward -700」的地板，訓練出「0% 丟棄、會看偏好
分流、跨 w 光譜打贏所有傳統策略」的 PPO agent。下一步 Phase 4：把它接上 FastAPI +
WebSocket + React 前端，讓這一切**即時看得到**。
