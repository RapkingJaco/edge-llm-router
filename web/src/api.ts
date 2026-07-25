// 型別化 WebSocket client：自動連線 + 斷線重連。
import type { Command, Snapshot } from "./types";

// 同源時走目前 host（FastAPI 服務 build）；dev（vite 5173）時連後端 8000。
function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const host = location.port === "5173" ? `${location.hostname}:8000` : location.host;
  return `${proto}://${host}/ws`;
}

export class RouterSocket {
  private ws: WebSocket | null = null;
  private closed = false;

  constructor(
    private onSnapshot: (s: Snapshot) => void,
    private onStatus: (connected: boolean) => void,
  ) {}

  connect(): void {
    this.ws = new WebSocket(wsUrl());
    this.ws.onopen = () => this.onStatus(true);
    this.ws.onmessage = (e) => this.onSnapshot(JSON.parse(e.data) as Snapshot);
    this.ws.onclose = () => {
      this.onStatus(false);
      if (!this.closed) setTimeout(() => this.connect(), 1000); // 重連
    };
    this.ws.onerror = () => this.ws?.close();
  }

  send(cmd: Command): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(cmd));
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }
}
