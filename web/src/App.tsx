import { useEffect, useRef, useState } from "react";
import { RouterSocket } from "./api";
import { PolicyConsole } from "./components/PolicyConsole";
import { RaceChart, type RacePoint } from "./components/RaceChart";
import { RoutingDashboard } from "./components/RoutingDashboard";
import { RunControls } from "./components/RunControls";
import { SummaryPanel } from "./components/SummaryPanel";
import type { Command, Snapshot } from "./types";

const MAX_POINTS = 600;

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<RacePoint[]>([]);
  const [connected, setConnected] = useState(false);
  const sockRef = useRef<RouterSocket | null>(null);

  useEffect(() => {
    const sock = new RouterSocket(
      (snap) => {
        setSnapshot(snap);
        setHistory((prev) => {
          if (snap.mode === "idle") return [];
          const base = snap.progress <= 1 ? [] : prev; // 新的一輪從頭畫
          const next = [...base, { ai: snap.ai.cum_reward, base: snap.base.cum_reward }];
          return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
        });
      },
      setConnected,
    );
    sock.connect();
    sockRef.current = sock;
    return () => sock.close();
  }, []);

  const send = (cmd: Command) => sockRef.current?.send(cmd);
  const w = snapshot?.w;
  const mode = snapshot?.mode ?? "idle";

  return (
    <div className="app">
      <header className="hero">
        <div>
          <h1>edge-llm-router</h1>
          <p>RL 智慧 LLM 推論分流器 — AI 逐請求決定在 本機/邊緣/雲 哪裡跑，優化延遲 + 成本</p>
        </div>
        <div className="status">
          <span className={`dot ${connected ? "on" : "off"}`} />
          {connected ? "即時連線" : "連線中…"}
          {w && <span className="wtag">方針 w＝延遲 {w[0].toFixed(2)} / 成本 {w[1].toFixed(2)}</span>}
        </div>
      </header>

      <RunControls
        mode={mode}
        progress={snapshot?.progress ?? 0}
        nTotal={snapshot?.n_total ?? 0}
        onRun={(n, peak) => {
          setHistory([]);
          send({ cmd: "run", n, peak });
        }}
        onReset={() => {
          setHistory([]);
          send({ cmd: "reset" });
        }}
      />

      {mode === "idle" && (
        <p className="hint">選一個請求數、按 <b>▶ 開始</b>，AI 會逐請求分流跑到底，最後給你統計。</p>
      )}
      {mode === "finished" && snapshot && <SummaryPanel snapshot={snapshot} />}

      <main className="grid">
        <RoutingDashboard snapshot={snapshot} />
        <RaceChart history={history} snapshot={snapshot} />
        <PolicyConsole snapshot={snapshot} send={send} />
      </main>
    </div>
  );
}
