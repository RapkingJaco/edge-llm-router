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
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" type="button" onClick={() => { setText(ex); submit(ex); }}>
            {ex}
          </button>
        ))}
      </div>
      {r && (
        <div className="classify">
          <div className="classify-line">「{r.text}」</div>
          <div className="classify-meta">
            約 {r.input_tokens} tokens · 複雜度 <b>{r.complexity}</b>
          </div>
          <div className="classify-route">
            → AI Agent 調度到 <b className={`route-${r.node}`}>{NODE_LABEL[r.node]}</b>
          </div>
        </div>
      )}
    </section>
  );
}
