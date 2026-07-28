import { useState } from "react";
import type { Command, NodeName, Snapshot } from "../types";

const NODE_LABEL: Record<NodeName, string> = {
  local: "本機 local",
  edge: "邊緣 edge",
  cloud: "雲 cloud",
};
const EXAMPLES = [
  "你好",
  "幫我規劃今天的行程",
  "翻譯這句話成英文",
  "寫一篇 500 字介紹邊緣運算的文章",
];

export function QueryProbe({ snapshot, send }: {
  snapshot: Snapshot | null;
  send: (c: Command) => void;
}) {
  const [text, setText] = useState("");
  const r = snapshot?.classify;
  const submit = (t: string) => {
    const v = t.trim();
    if (v) send({ cmd: "classify", text: v });
  };

  return (
    <section className="panel probe">
      <h2>試打一個請求</h2>
      <p className="panel-sub">
        打一句話，看它有多大、AI Agent 會把它調度到哪個節點（只是預測，不會實際送出）。
      </p>
      <form
        className="policy-form"
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="例如：幫我規劃今天的行程"
          aria-label="請求輸入"
        />
        <button type="submit">試算</button>
      </form>
      <p className="chips-hint">點範例把它帶進輸入框，再按「試算」：</p>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" type="button" onClick={() => setText(ex)}>
            {ex}
          </button>
        ))}
      </div>
      {r && (
        <div className="classify">
          <div className="classify-line">「{r.text}」</div>
          <div className="classify-meta">
            估 <b>{r.input_tokens}</b> 輸入 tokens · 約 <b>{r.output_est}</b> 輸出 · 複雜度 <b>{r.complexity}</b>
          </div>
          <table className="classify-table">
            <thead>
              <tr><th>節點</th><th>首字 TTFT</th><th>完整回應</th><th>成本</th><th></th></tr>
            </thead>
            <tbody>
              {(["local", "edge", "cloud"] as NodeName[]).map((k) => (
                <tr key={k} className={k === r.node ? "picked" : ""}>
                  <td className={`route-${k}`}>{NODE_LABEL[k]}</td>
                  <td>{Math.round(r.per_node[k].ttft_ms)} ms</td>
                  <td>{Math.round(r.per_node[k].total_ms)} ms</td>
                  <td>{r.per_node[k].cost.toFixed(4)}</td>
                  <td>{k === r.node ? "★ AI Agent 選" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="classify-hint">
            空載預測。<b>首字 TTFT</b>＝網路 + 讀輸入；<b>完整回應</b>再加輸出生成（長輸出時雲端
            decode 快才划算）。AI Agent 依你目前的<b>優化目標</b>平衡「回應時間 vs 成本」挑最划算的
            ——<b>改上面的優化目標</b>（省 → local、快 → cloud）會換節點；各半時 edge 最全能。
          </p>
        </div>
      )}
    </section>
  );
}
