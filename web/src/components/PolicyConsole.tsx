import { useState } from "react";
import type { Command, Snapshot } from "../types";

const EXAMPLES = ["成本太高，多用本機", "我要最快", "平衡就好"];

export function PolicyConsole({ snapshot, send }: {
  snapshot: Snapshot | null;
  send: (c: Command) => void;
}) {
  const [text, setText] = useState("");
  const submit = (t: string) => {
    const v = t.trim();
    if (v) send({ cmd: "policy", text: v });
  };

  return (
    <section className="panel">
      <h2>中文方針</h2>
      <p className="panel-sub">一句話下指令 → 轉成權重 → AI 行為零重訓當場改（傳統策略不受影響）。</p>
      <form
        className="policy-form"
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
          setText("");
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="例如：成本太高，多用本機"
          aria-label="中文方針輸入"
        />
        <button type="submit">送出</button>
      </form>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" type="button" onClick={() => submit(ex)}>
            {ex}
          </button>
        ))}
      </div>
      {snapshot?.note && <p className="note">解讀：{snapshot.note}</p>}
    </section>
  );
}
