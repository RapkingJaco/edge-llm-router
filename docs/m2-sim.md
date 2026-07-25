# m2 — sim Gymnasium 環境（Phase 2）

> 里程碑筆記：做了什麼、為什麼這樣設計、踩到什麼坑。

## 直觀理解（白話）

**一句話**：Phase 1 蓋了世界的物理，Phase 2 把它包成一台標準「電動遊戲」（Gymnasium
環境），讓 RL agent 能來玩、來學。記憶鉤：**Phase 1 是物理引擎，Phase 2 是裝上標準手把
的遊戲機。**

**電動類比**：一個 RL 環境就是一台遊戲，每一格畫面（step）發生三件事：

```
   看畫面 ──►  按按鈕  ──► 拿分數
(observation) (action)   (reward)
     ▲                        │
     └──────  下一格畫面  ◄────┘
```

- observation：當下狀態（一串數字）；action：決策（按哪顆鈕）；reward：好壞分數。
- episode：一整局（≈ 60 秒一波流量、約 500 個請求）。
- agent 靠玩很多局、看哪種按法拿高分來學。這在 RL 叫 **MDP**（Markov 決策過程）；
  Phase 2 就是把分流問題寫成一個 MDP。

**一步 = 一個請求**：每來一個請求，agent 就選一個節點送。step = 處理一個請求，一局幾百步。

**看畫面（observation 15 維）**：這個請求多大（2）＋三節點各自多忙（9）＋全域花多少錢/
多塞（2）＋你現在在乎什麼 w（2）。
- 重要細節「不能偷看答案」：輸出長度只給 agent **帶雜訊的估計**，不給真值——真實世界裡
  請求還沒生成完，你不知道它會多長。讓 agent 偷看真值，模擬裡很強、接真機就破功。這叫
  **不洩上帝視角**，是之後 Sim-to-Real 能成立的誠實前提。

**按按鈕（action）**：`Discrete(3)`，本機/邊緣/雲三選一。

**拿分數（reward）**：`-(w_lat×正規化延遲 + w_cost×正規化成本) - 丟棄懲罰`。負號是因為
延遲/成本越低越好，分數越接近 0 越好；w 決定這局在乎快還是省；丟棄大扣分。

**權重條件化（核心賣點）**：每局隨機抽一組 w 塞進畫面，逼 agent 學「看 w 隨機應變」而非
單一策略。這樣同一個 policy 之後改 w 就能**零重訓**切換行為。

**random rollout 數字**：隨機亂按 → 丟棄率 40%+、reward ~-700，因為它不看畫面、硬塞爆
節點。這是一個爛到有明顯進步空間的地板；PPO 學會看負載分流就能大幅超越。
記憶鉤：**隨機是地板，PPO 要當天花板，中間的差距就是作品集要秀的「AI 贏多少」。**

## 做了什麼

- `sim/workload.py`：請求到達流程 + token 分布
  - 非齊次卜瓦松過程（thinning 細化法），中段高斯突波製造**尖峰**
  - token 短/中/長三段混合（短多長少）
  - 給定 seed 可重現整局工作負載
- `sim/env.py`：`RouterEnv(gym.Env)`
  - observation 15 維、全部正規化到 [0,1]、動作 `Discrete(3)`
  - reward 依當前權重 `w` 用 `metrics.scalarize_reward` 結算；丟棄加大懲罰
  - **權重條件化**：每局 `reset()` 隨機抽 `w`，放進 obs 末兩維
- `scripts/random_rollout.py`：隨機策略跑一整局的摘要
- `tests/test_sim.py`：7 個測試（含 gymnasium `check_env` 官方檢查）

## 一步 = 一個請求（事件驅動）

到達序列在 `reset()` 一次產生；每個 `step(action)` 處理「當前到達的請求」：
路由到選定節點 → `backend.infer(req, now=到達時間)` → 算 reward → 前進到下一個到達。
episode 在請求用完時 `truncated=True` 結束。節點的排隊狀態隨 `now` 前進自然累積壅塞。

## observation 15 維（不洩上帝視角）

| 塊 | 內容 | 維 |
|---|---|---|
| 這個請求 | input token（已知）、output token **帶雜訊估計** | 2 |
| 每節點 ×3 | 使用率、佇列長/容量、預估等待/丟棄門檻 | 9 |
| 全域 | 累積成本/最壞情況、當前負載程度 | 2 |
| 方針 | w_lat、w_cost | 2 |

- output token 真值只給模擬器算服務時間；agent 只拿到 `真值 × (1 + N(0, 0.2))` 的估計。
- 正規化基準皆固定（token/512、等待/丟棄門檻、成本/全雲最壞值），避免動態統計漂移。

## reward

`r = -(w_lat·正規化TTFT + w_cost·正規化成本) - 逾時懲罰`（丟棄時 TTFT 破表→正規化=1、
另加 `timeout_penalty`）。越接近 0 越好。

## 隨機策略 rollout（要被 PPO 打敗的基準）

| seed | w(延遲,成本) | 丟棄率 | 平均TTFT | 總reward |
|---|---|---|---|---|
| 0 | 0.64 / 0.36 | 40.4% | 1142ms | -685 |
| 1 | 0.51 / 0.49 | 44.2% | 1395ms | -747 |
| 2 | 0.26 / 0.74 | 42.2% | 1186ms | -652 |

> 隨時重跑：`uv run python scripts/random_rollout.py [seed]`

**解讀**：隨機亂丟 → 40%+ 請求被塞爆丟棄。PPO 只要學會「看負載分流、別硬塞便宜節點」
就能大幅改善——這就是 Phase 3 的空間。

## 踩到的坑 / 設計選擇

- **Request 正名**：`output_tokens_est` → `output_tokens`（真值）。估計值 + 雜訊移到 env
  建 obs 時才加，避免洩上帝視角。
- **只用 truncation 不用 termination**：固定長度工作負載，episode 靠請求用完結束。
- **成本正規化用「全雲最壞值」**：`cloud_cost × 請求數`，可解釋成「花了最壞情況的幾成」。

## 驗收

- ruff ✅ / pytest ✅ 22 passed（含 `check_env` 官方檢查、本 Phase 7 個）
- `check_env` 通過 ✅；隨機策略跑得完整局並印摘要 ✅；obs 恆在 [0,1] ✅；換 w reward 變 ✅

## 下一步

Phase 3：`agent/` — CleanRL PPO（吃這個環境）+ 基準線（貪婪/輪詢/全雲/全本機），
目標**拿到「AI 贏基準線」**。這是整個作品集的靈魂里程碑。
