"""SimulatedBackend：c 個平行服務槽的排隊模型（G/G/c，貪婪指派最早空出的槽）。

核心是**忙 → 變慢**：每個請求佔用一個服務槽整段生成時間（prefill + decode），
槽都在忙就得等最早空出的那個；等待超過門檻（塞爆）就丟棄。這製造「便宜節點一擠
就爆」的張力，分流才有意義。

TTFT ≈ 排隊等待 + 網路 RTT + prefill。時間單位皆毫秒（ms）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import InferResult, NodeBackend, NodeState, Request


@dataclass
class SimulatedBackend(NodeBackend):
    """單一模擬節點。參數由 ``configs/default.yaml`` 帶入（見 ``build_nodes``）。"""

    name: str
    rtt_ms: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    capacity: int
    cost_per_request: float
    drop_wait_ms: float
    # 每個服務槽「何時空出」的時間戳（ms）。長度 = capacity。
    _free_at: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"{self.name}: capacity 需 >= 1")
        self.reset()

    def reset(self) -> None:
        self._free_at = [0.0] * self.capacity

    def _service_ms(self, req: Request) -> float:
        """整段生成佔用節點的時間：prefill（隨輸入）+ decode（隨輸出）。"""
        prefill = self.prefill_ms_per_token * req.input_tokens
        decode = self.decode_ms_per_token * req.output_tokens
        return prefill + decode

    def state(self, now: float = 0.0) -> NodeState:
        busy = sum(1 for f in self._free_at if f > now)
        all_busy = busy >= self.capacity
        est_wait = max(0.0, min(self._free_at) - now) if all_busy else 0.0
        return NodeState(
            utilization=busy / self.capacity,
            queue_len=busy,
            est_wait_ms=est_wait,
        )

    def infer(self, req: Request, now: float = 0.0) -> InferResult:
        # 貪婪指派：挑最早空出的槽。
        slot = min(range(self.capacity), key=lambda i: self._free_at[i])
        start = max(now, self._free_at[slot])
        wait = start - now
        prefill_ms = self.prefill_ms_per_token * req.input_tokens

        # 塞爆：排隊等待超過門檻 → 入場即拒（不佔槽），逼 agent 別把便宜節點塞死。
        if wait > self.drop_wait_ms:
            return InferResult(
                node=self.name,
                ttft_ms=wait + self.rtt_ms + prefill_ms,
                cost=0.0,
                dropped=True,
            )

        service = self._service_ms(req)
        self._free_at[slot] = start + service
        return InferResult(
            node=self.name,
            ttft_ms=wait + self.rtt_ms + prefill_ms,
            cost=self.cost_per_request,
            full_ms=wait + service,
        )


def build_nodes(config: dict[str, Any]) -> dict[str, SimulatedBackend]:
    """從設定 dict 建三個模擬節點：``{"local":..., "edge":..., "cloud":...}``。"""
    drop_wait_ms = config.get("sim", {}).get("drop_wait_ms", 4000.0)
    nodes: dict[str, SimulatedBackend] = {}
    for name, nc in config["nodes"].items():
        nodes[name] = SimulatedBackend(
            name=name,
            rtt_ms=nc["rtt_ms"],
            prefill_ms_per_token=nc["prefill_ms_per_token"],
            decode_ms_per_token=nc["decode_ms_per_token"],
            capacity=nc["capacity"],
            cost_per_request=nc["cost_per_request"],
            drop_wait_ms=drop_wait_ms,
        )
    return nodes
