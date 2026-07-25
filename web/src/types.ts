// 對應 server/simulation.py 的快照格式。

export type NodeName = "local" | "edge" | "cloud";

export interface LaneView {
  name: string;
  node: NodeName;
  dropped: boolean;
  cum_reward: number;
  cost: number;
  served: number;
  drops: number;
  avg_ttft_ms: number;
}

export type Utils = Record<NodeName, number>;

export interface Snapshot {
  t: number;
  w: [number, number];
  peak: boolean;
  note: string;
  episode_over: boolean;
  ai_loaded: boolean;
  lead_pct: number;
  ai_utils: Utils;
  base_utils: Utils;
  ai: LaneView;
  base: LaneView;
}

export type Command =
  | { cmd: "reset" }
  | { cmd: "peak"; on: boolean }
  | { cmd: "policy"; text: string };
