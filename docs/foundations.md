# 基礎（RL / LLM serving）

> 打底用的觀念筆記，邊做邊補。

## RL 基礎
- MDP：state / action / reward / transition。
- PPO：policy gradient + clip 限制每步更新幅度；GAE 估 advantage。
- on-policy：用當前 policy 蒐集資料再更新。

## LLM serving / inference routing
- TTFT（time-to-first-token）：首字延遲，互動體驗關鍵指標。
- prefill vs decode：prefill 處理輸入 prompt（≈ 每 token 固定成本），decode 逐 token 產生。
- 排隊：併發越高、等待越久，非線性惡化。

*(建置中)*
