import { useState } from "react";
import type { Mode } from "../types";

const COUNTS = [100, 200, 300, 500];

export function RunControls({ mode, progress, nTotal, onRun, onReset }: {
  mode: Mode;
  progress: number;
  nTotal: number;
  onRun: (n: number, peak: boolean) => void;
  onReset: () => void;
}) {
  const [n, setN] = useState(200);
  const [peak, setPeak] = useState(false);
  const running = mode === "running";
  const pct = nTotal ? Math.round((progress / nTotal) * 100) : 0;

  return (
    <div className="run">
      <div className="run-row">
        <span className="run-label">請求數：</span>
        {COUNTS.map((c) => (
          <button
            key={c}
            className={`chip ${n === c ? "chip-on" : ""}`}
            disabled={running}
            onClick={() => setN(c)}
          >
            {c}
          </button>
        ))}
        <label className="peak-toggle">
          <input type="checkbox" checked={peak} disabled={running}
            onChange={(e) => setPeak(e.target.checked)} /> 尖峰
        </label>
        <button className="btn btn-run" disabled={running} onClick={() => onRun(n, peak)}>
          {running ? "跑動中…" : "▶ 開始"}
        </button>
        <button className="btn" onClick={onReset}>重置</button>
      </div>
      {mode !== "idle" && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${pct}%` }} />
          <span className="progress-txt">
            {mode === "finished" ? `完成 ${nTotal} 筆` : `${progress} / ${nTotal}`}
          </span>
        </div>
      )}
    </div>
  );
}
