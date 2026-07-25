# m1 — backends 介面 + SimulatedBackend（Phase 1）

> 里程碑筆記：做了什麼、為什麼這樣設計、踩到什麼坑。

## 直觀理解（白話）

**一句話**：Phase 1 還沒有任何 AI，是先把「世界的物理」蓋出來——一個模擬「請求跑到某
節點會怎樣」的模型。有了合理的世界，之後的 RL agent 才有東西可學。
記憶鉤：**Phase 1 是蓋舞台，還沒有演員（agent）。**

**餐廳類比**：三個節點像三種餐廳。
- local ＝ 2 個爐子的路邊攤：免費，但人一多就大排長龍，等到受不了的客人直接走（丟棄）。
- cloud ＝ 16 個爐子的大飯店：又快又穩，客人來都能馬上服務，但收費是路邊攤的 100 倍。
- edge ＝ 中間。

**TTFT 是什麼**：time-to-first-token，你按送出到看到第一個字冒出來的時間，是 LLM 服務
體驗最關鍵的指標。組成 ≈ 排隊等待 + 網路來回（RTT）+ prefill（讀完你的輸入）。

**demo 那張圖在說什麼**（同一瞬間灌 8 個請求，壓力測試）：
- local（容量 2）：前 2 個 770ms 還行；第 3 個起要排隊等 7000+ms、超過 4000ms 丟棄門檻
  → 全被丟掉。
- cloud（容量 16）：8 個全部 171ms，穩，但每個都貴 100×。

**為什麼「排隊」是靈魂**：延遲若固定不變，最佳解永遠是「全丟最快的」，RL 沒得學。加上
「忙→變慢→塞爆」後才有取捨（省錢 vs 快），才值得用 RL——沒有這個張力，整個專案不成立。

## 做了什麼

- `backends/base.py`：後端契約
  - `Request`（估計輸入/輸出 token + 可選 prompt 文字）
  - `InferResult`（node / ttft_ms / cost / dropped / is_measured / full_ms）
  - `NodeState`（utilization / queue_len / est_wait_ms）——給 observation 讀
  - `NodeBackend`(ABC)：`infer(req, now)` / `state(now)` / `reset()`
- `backends/simulated.py`：`SimulatedBackend`（G/G/c 排隊模型）+ `build_nodes(config)`
- `configs/default.yaml`：新增 `sim.drop_wait_ms`（塞爆丟棄門檻）
- `tests/test_backends.py`：7 個測試（忙→變慢、塞爆丟棄、成本比、reset、util 界內…）

## 排隊模型（讓問題有意思的關鍵）

節點有 `capacity` 個平行服務槽，各記「何時空出」（`_free_at`）。新請求：
1. 貪婪挑**最早空出**的槽；`start = max(now, 該槽空出時間)`，`wait = start - now`。
2. `wait > drop_wait_ms` → **入場即拒**（dropped，不佔槽），逼 agent 別把便宜節點塞死。
3. 否則佔槽整段生成：`service = prefill + decode`，`prefill = prefill_ms/token × 輸入token`，
   `decode = decode_ms/token × 輸出token`。
4. `TTFT = wait + RTT + prefill`（首字在 prefill 完成後出來）。

時間單位一律 **ms**；`now` 是模擬時鐘（真實後端會忽略、直接量 wall-clock）。

🧠 **為何一定要「忙→變慢」**：延遲固定的話最佳解永遠是「全丟最快的」，RL 無事可學。
排隊讓「便宜節點一擠就爆」，分流才有意義。

## 實測 demo（同瞬間連發 8 個請求）

> 隨時重跑：`uv run python scripts/demo_backends.py`


| 節點 | 行為 |
|---|---|
| local（cap 2） | 前 2 個 770ms；第 3 個起排隊等 7168ms > 4000ms 門檻 → **全丟棄** |
| cloud（cap 16） | 8 個全 171ms、每個 cost 0.01（local 的 100×） |

## 設計選擇

- **`infer(req, now)` 帶時鐘**：模擬需要時鐘算排隊；真實後端同介面但忽略 `now`。介面一致 →
  模擬/真打可無縫替換（`sim` 不知後端真假）。
- **丟棄 = 入場即拒、不佔槽**：避免「無限延遲」的殭屍請求；製造清楚的過載訊號。
- **成本 flat per request**：v1 用 `cost_per_request`（自訂標價）；日後要按 token 計可再改。

## 驗收

- ruff ✅ / pytest ✅ 15 passed（含本 Phase 7 個）
- 越忙 TTFT 越高 ✅；塞爆丟棄 ✅；cloud ≈ local×100 ✅；reset 清空 ✅

## 下一步

Phase 2：`sim/` Gymnasium 環境——請求流 + 用這些後端組出 `reset()/step()`，
obs ~15 維、`Discrete(3)`、reward 依權重 w。
