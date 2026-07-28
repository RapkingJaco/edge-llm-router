// 對應 server/simulation.py 的快照格式。

export type NodeName = "local" | "edge" | "cloud";
export type Mode = "idle" | "running" | "finished";

export interface LaneView {
  name: string;
  node: NodeName | null; // 最新請求送去哪；idle 時為 null
  dropped: boolean;
  cum_reward: number;
  cost: number;
  served: number;
  drops: number;
  avg_ttft_ms: number;
}

export type Utils = Record<NodeName, number>;

export interface Measured {
  node: NodeName;
  ttft_ms: number;
  is_measured: boolean;
  dropped: boolean;
  backend: string;
  t: number;
}

export interface Snapshot {
  mode: Mode;
  n_total: number;
  progress: number;
  w: [number, number];
  peak: boolean;
  note: string;
  measured: Measured | null;
  ai_loaded: boolean;
  lead_pct: number;
  ai_utils: Utils;
  base_utils: Utils;
  ai: LaneView;
  base: LaneView;
}

export type Command =
  | { cmd: "run"; n: number; peak: boolean }
  | { cmd: "reset" }
  | { cmd: "policy"; text: string };
