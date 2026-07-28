# m7 — 進階圖表（Phase 7）

> 加分項。本輪做 P7-1（Pareto 前緣）、P7-2（gap 隨迭代下降）。LSTM 消融、OllamaControl
> 全離線列為未做的選配。

## P7-1 — Pareto 前緣 ✅

**做了什麼**
- `evaluation.pareto_points()`：取某策略跨 w 的 (cost, ttft) 點。
- `eval/pareto.py`：載 `ppo_wc.pt`，掃 w_cost = 0.0…1.0（11 點），出 `outputs/pareto.{json,png}`。

**結果（權重條件化 PPO，w_cost 0→1）**

| w_cost | 0.0 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| 成本 | 4.76 | 3.97 | 3.53 | 3.23 | 3.02 |
| TTFT(ms) | 212 | 204 | 239 | 434 | 918 |
| 丟棄率 | 0% | 0% | 0% | 0% | 0% |

**一個 policy 掃出整條「成本 vs TTFT」前緣**，全程 0% 丟棄。基準線各只是一個固定點
（greedy≈(5.21,214)、all_cloud≈(5.36,224)），PPO 前緣壓在它們左下方——同樣的速度更便宜、
還能往更省的操作點延伸。round_robin/all_local 因 40%/96% 丟棄不是實際選項。

🧠 **為什麼這是殺手級圖表**：傳統做法只能「選一個點」；權重條件化的 PPO 給你**一整條可
即時滑動的取捨曲線**，配上 Phase 5 的中文控制＝「用講的沿著前緣滑」。

## P7-2 — gap 隨迭代下降 ✅

**做了什麼**
- `calibration/gap_iterations.py`：每輪多收一批真實 Ollama 量測、重新擬合，在**獨立
  held-out 真實量測**上算 Wasserstein gap。出 `outputs/gap_iterations.{json,png}`。

**結果（真 Ollama，log 軸）**

| 迭代 | 0（未校準） | 1 | 2 | 3 | 6 |
|---|---|---|---|---|---|
| gap(ms) | 2286 | 72 | 47 | 49 | 44 |

gap 從 2286ms（placeholder 亂猜）→ 第一次擬合就崩到 72ms → 隨資料增加收斂到 ~44ms。
誠實說明：迭代＝「校準資料量增加」，大跌在「猜→擬合」，之後靠更多資料穩定（非重訓迴圈）。

🧠 **這證明什麼**：模擬不是憑感覺調的——拿真機量測回歸校準，gap 收斂到幾十 ms，代表
「在模擬裡訓練的策略，數字可信」。這是 Sim-to-Real 的硬證據。

## OllamaControl（全離線真 LLM 控制）✅
- `control/ollama_control.py`：用本機 Ollama（llama3.2）+ `format=json` 逼吐
  `{intent, magnitude}`，few-shot prompt（範例 8/8 正確），失敗自動退規則版。
- `build_control(config)` 依 `control.provider` 選 ollama/rule；GCP 無 Ollama 自動退規則版。
- server 把 policy 指令丟 `to_thread`（真 LLM 數秒不卡迴圈）。瀏覽器實測：🦙 Ollama 解讀、
  w 當場變、不凍。**控制層不再只有規則版，是真的離線 LLM。**

## 未做（選配）
- LSTM 消融：驗證「observation 已是充分統計量、記憶幫助有限」的對照實驗。
- Domain Randomization：每 episode 在真值±範圍隨機化參數，訓練對「真實落差」更穩。
