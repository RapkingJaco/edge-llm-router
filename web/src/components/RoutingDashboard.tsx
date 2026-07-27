import type { NodeName, Snapshot, Utils } from "../types";

const NODES: { key: NodeName; label: string; hint: string }[] = [
  { key: "local", label: "本機 local", hint: "便宜 · 一擠就爆" },
  { key: "edge", label: "邊緣 edge", hint: "4070 · 中庸" },
  { key: "cloud", label: "雲 cloud", hint: "穩 · 貴 100×" },
];

function NodeBar({ name, label, hint, util, active }: {
  name: NodeName;
  label: string;
  hint: string;
  util: number;
  active: boolean;
}) {
  const pct = Math.round(util * 100);
  const danger = util >= 0.85;
  return (
    <div className={`node ${active ? "node-active" : ""}`}>
      <div className="node-head">
        <span className="node-label">{label}</span>
        <span className="node-pct">{pct}%</span>
      </div>
      <div className="bar">
        <div
          className={`bar-fill ${danger ? "bar-danger" : ""} bar-${name}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="node-hint">{hint}</div>
    </div>
  );
}

export function RoutingDashboard({ snapshot }: { snapshot: Snapshot | null }) {
  const utils: Utils = snapshot?.ai_utils ?? { local: 0, edge: 0, cloud: 0 };
  const current = snapshot?.ai.node;
  return (
    <section className="panel">
      <h2>即時分流（AI）</h2>
      <p className="panel-sub">PPO 逐請求決定丟哪個節點；滿載會塞爆丟棄。</p>
      <div className="nodes">
        {NODES.map((n) => (
          <NodeBar key={n.key} name={n.key} label={n.label} hint={n.hint}
            util={utils[n.key]} active={current === n.key} />
        ))}
      </div>
      {snapshot && (
        <div className="stats">
          <div><span>已處理</span><b>{snapshot.ai.served}</b></div>
          <div><span>丟棄</span><b className={snapshot.ai.drops ? "warn" : ""}>{snapshot.ai.drops}</b></div>
          <div><span>成本</span><b>{snapshot.ai.cost.toFixed(2)}</b></div>
          <div><span>平均 TTFT</span><b>{Math.round(snapshot.ai.avg_ttft_ms)}ms</b></div>
        </div>
      )}
      {snapshot?.measured && (
        <div className={`sample ${snapshot.measured.is_measured ? "real" : ""}`}>
          {snapshot.measured.is_measured ? "✅ 實測" : "抽驗"}
          <span className="sample-node"> {snapshot.measured.node}</span>
          {snapshot.measured.is_measured
            ? <> · 真打後端 <b>{Math.round(snapshot.measured.ttft_ms)}ms</b></>
            : <> · 該節點退回模擬</>}
        </div>
      )}
    </section>
  );
}
