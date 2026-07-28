import type { Snapshot } from "../types";

export interface RacePoint {
  ai: number;
  base: number;
}

// 手繪 SVG 兩線賽跑（不加圖表庫依賴）。累積 reward 皆從 0 往下走，AI 線在上＝領先。
export function RaceChart({ history, snapshot }: { history: RacePoint[]; snapshot: Snapshot | null }) {
  const W = 460;
  const H = 200;
  const pad = 20;
  const pts = history.length ? history : [{ ai: 0, base: 0 }];
  const values = pts.flatMap((p) => [p.ai, p.base]);
  const minV = Math.min(0, ...values);
  const span = minV < 0 ? -minV : 1;
  const n = pts.length;
  const x = (i: number) => pad + (n <= 1 ? 0 : (i / (n - 1)) * (W - 2 * pad));
  const y = (v: number) => pad + ((0 - v) / span) * (H - 2 * pad);
  const line = (key: "ai" | "base") =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(" ");

  const lead = snapshot?.lead_pct ?? 0;
  const ahead = lead >= 0;

  return (
    <section className="panel">
      <h2>AI Agent vs 傳統排程</h2>
      <p className="panel-sub">累積 reward，越高＝越接近 0＝越好。藍＝AI Agent，灰＝傳統排程。</p>
      <div className="lead">
        <span className={`lead-num ${ahead ? "up" : "down"}`}>
          {ahead ? "▲" : "▼"} {Math.abs(lead).toFixed(1)}%
        </span>
        <span className="lead-txt">{ahead ? "AI 提升" : "AI 落後"}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="race" preserveAspectRatio="none" role="img"
        aria-label={`AI 相對傳統領先 ${lead.toFixed(1)}%`}>
        <line x1={pad} y1={pad} x2={W - pad} y2={pad} className="axis0" />
        <path d={line("base")} className="line-base" />
        <path d={line("ai")} className="line-ai" />
      </svg>
      <div className="race-legend">
        <span><i className="sw sw-ai" />AI Agent</span>
        <span><i className="sw sw-base" />傳統排程</span>
        <span className="axis-note">上緣＝0（完美）</span>
      </div>
    </section>
  );
}
