# m5 — control：中文方針 → 權重（Phase 5）

> 里程碑筆記：做了什麼、為什麼、踩什麼坑。

## 直觀理解（白話）

前面訓好的 PPO 是「會看 w 隨機應變」的 agent。Phase 5 做的是**讓人用一句中文改 w**：
打「成本太高，多用本機」→ 轉成 w＝(延遲0.1, 成本0.9) → 餵進 obs → PPO 當場改分流，
**完全不用重訓**。傳統策略（greedy）無視 w，所以只有 AI 會變——對照更明顯。
記憶鉤：**w 是「旋鈕」，中文是「用講的轉旋鈕」，PPO 立刻照新旋鈕跑。**

## 核心設計：只吐意圖，不讓 LLM 吐數字

`ControlLLM.parse(text, current_w) → ControlResult`：LLM/規則只判斷**意圖**
（cheaper/faster/balanced）+ **強度**，再由確定性程式 `intent_to_weights` 換算成 w。
這層是「絕不盲信 LLM」的安全網——LLM 只做語意判斷，數字由我們控制、可夾範圍、可驗證。

**安全網**：看不懂（無關鍵字）或空白 → 維持原方針、不亂動。

## 做了什麼（P5-1 ~ P5-3）

- `control/base.py`：`ControlLLM` 介面、`ControlResult`、`intent_to_weights`、`normalize_weights`。
- `control/rule_based.py`：`RuleBasedControl`——關鍵字比對（省成本 vs 低延遲 vs 平衡、
  強語氣加碼）。離線、免 key，永遠是可用 fallback。
- `RouterEnv.set_w()`：即時改 w（含當前 episode），obs 下一步就反映 → PPO 零重訓改行為。
- `server`：`LiveSimulation.set_policy(text)` 套用新 w；WS 指令 `{"cmd":"policy","text":...}`；
  快照多帶 `note`（一句話解讀）。
- `web`：右面板 `PolicyConsole`——輸入框 + 範例 chip + 顯示解讀；即時改 header 的 w。
- 測試：`tests/test_control.py` 8 個。

## 驗證（瀏覽器實測）

| 下的方針 | w 變成 | AI 分流 | 解讀 |
|---|---|---|---|
| 成本太高，多用本機 | 延遲 0.1 / 成本 0.9 | local/edge 100%、cloud↓、成本 1.22 | 偏向省成本（強）|
| 我要最快 | 延遲 0.9 / 成本 0.1 | 多用 cloud、成本升到 2.68 | 偏向低延遲（強）|

同一個 policy，只改中文 → 行為當場翻轉、AI 全程領先 ~25-32%。傳統策略不受影響。

## 可插拔：真 Gemini（之後）

`GeminiControl(ControlLLM)` 是可插拔選項：用 Gemini 免費層**結構化輸出**吐
`{intent, magnitude, note}`，同樣經 `intent_to_weights` + 安全網。放了 `GEMINI_API_KEY`
再實作即可，介面不變（規則版永遠當 fallback）。SDK 選 `google-genai`（新）vs
`google-generativeai`（舊）屆時定。

## DoD

「成本太高多用本機」→ 偏成本、更多 local/edge ✅；「我要最快」→ 偏延遲、更多 cloud ✅；
亂輸入不崩、維持原方針 ✅；全程無重訓 ✅；`docs/m5-control.md` 寫好 ✅
