import type { Snapshot } from "../types";

export function SummaryPanel({ snapshot }: { snapshot: Snapshot }) {
  const { ai, base, lead_pct, n_total, w } = snapshot;
  const aiWin = lead_pct >= 0;
  return (
    <section className="summary">
      <div className="summary-head">
        <h2>本次結果（{n_total} 筆請求）</h2>
        <span className={`summary-verdict ${aiWin ? "win" : "lose"}`}>
          {aiWin ? `AI 提升 ▲ ${lead_pct.toFixed(1)}%` : `AI 落後 ${Math.abs(lead_pct).toFixed(1)}%`}
        </span>
      </div>
      <table className="summary-table">
        <thead>
          <tr><th>指標</th><th>AI Agent</th><th>傳統排程</th></tr>
        </thead>
        <tbody>
          <tr><td>已處理</td><td>{ai.served}</td><td>{base.served}</td></tr>
          <tr><td>丟棄</td><td className={ai.drops ? "warn" : ""}>{ai.drops}</td><td className={base.drops ? "warn" : ""}>{base.drops}</td></tr>
          <tr><td>平均 TTFT</td><td>{Math.round(ai.avg_ttft_ms)} ms</td><td>{Math.round(base.avg_ttft_ms)} ms</td></tr>
          <tr><td>總成本</td><td>{ai.cost.toFixed(2)}</td><td>{base.cost.toFixed(2)}</td></tr>
          <tr className="row-reward"><td>總 reward</td><td>{ai.cum_reward.toFixed(1)}</td><td>{base.cum_reward.toFixed(1)}</td></tr>
        </tbody>
      </table>
      <p className="summary-note">
        reward 越接近 0 越好（把「快、省、別塞爆」揉成一分）；本次方針 w＝延遲 {w[0]} / 成本 {w[1]}
      </p>
    </section>
  );
}
