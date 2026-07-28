import { useState } from "react";
import type { Command, Snapshot } from "../types";

const EXAMPLES = [
  "成本太高，多用本機",
  "我要最快",
  "平衡就好",
  "盡量省錢",
  "延遲優先，別讓使用者等",
  "預算有限",
];

export function PolicyConsole({ snapshot, send }: {
  snapshot: Snapshot | null;
  send: (c: Command) => void;
}) {
  const [text, setText] = useState("");

  return (
    <section className="panel">
      <h2>優化目標（自然語言）</h2>
      <p className="panel-sub">
        用一句話設定要偏「快」還是「省」；AI Agent 零重訓即時調整（傳統排程不受影響）。
      </p>
      <form
        className="policy-form"
        onSubmit={(e) => {
          e.preventDefault();
          const v = text.trim();
          if (v) send({ cmd: "policy", text: v });
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="例如：成本太高，多用本機"
          aria-label="優化目標輸入"
        />
        <button type="submit">送出</button>
      </form>
      <p className="chips-hint">點一下把範例帶進輸入框，可再修改後送出：</p>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" type="button" onClick={() => setText(ex)}>
            {ex}
          </button>
        ))}
      </div>
      {snapshot?.note && <p className="note">解讀：{snapshot.note}</p>}
    </section>
  );
}
