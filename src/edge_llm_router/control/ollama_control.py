"""OllamaControl：用本機 Ollama（llama3.2）把中文方針轉成意圖 → 權重。全離線、免 Gemini/額度。

真 LLM 版的「吐意圖」控制層：用 Ollama 的 `format=json` 逼 llama3.2 吐
`{intent, magnitude}`，再由確定性程式 `intent_to_weights` 換算 w（同一套安全網）。
任何失敗（Ollama 掛 / JSON 壞 / intent 超出範圍）→ **退回規則版**，服務不中斷。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..backends.ollama_backend import ollama_available
from .base import ControlLLM, ControlResult, WeightPair, intent_to_weights
from .rule_based import RuleBasedControl

_BASE_URL = "http://localhost:11434"
_VALID = {"cheaper", "faster", "balanced"}
_LABEL = {"cheaper": "省成本", "faster": "低延遲", "balanced": "平衡"}
_PROMPT = (
    "你是把中文方針轉成 LLM 推論優化偏好的分類器。只輸出 JSON。\n"
    "分類：faster=要更快(低延遲,願多花錢)、cheaper=要更省成本(可慢一點)、balanced=兩者兼顧。\n"
    "magnitude 是強度 0~1（越極端越接近 1；平衡用 0.5）。\n"
    "範例：\n"
    '「成本太高，多用本機」→ {{"intent":"cheaper","magnitude":0.8}}\n'
    '「盡量省錢」→ {{"intent":"cheaper","magnitude":0.9}}\n'
    '「我要最快」「太慢了受不了」→ {{"intent":"faster","magnitude":0.9}}\n'
    '「延遲重要一點」→ {{"intent":"faster","magnitude":0.6}}\n'
    '「平衡就好」「兼顧」→ {{"intent":"balanced","magnitude":0.5}}\n'
    "現在分類這句方針：「{text}」\n"
    "只回 JSON："
)


class OllamaControl(ControlLLM):
    """真 LLM（llama3.2）控制層；失敗自動退回規則版。"""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = _BASE_URL,
        timeout: float = 20.0,
        fallback: ControlLLM | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._fallback = fallback or RuleBasedControl()

    def parse(self, text: str, current_w: WeightPair) -> ControlResult:
        t = text.strip()
        if not t:
            return self._fallback.parse(text, current_w)
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "prompt": _PROMPT.format(text=t),
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
            data = json.loads(resp.json()["response"])
            intent = str(data.get("intent", "")).strip().lower()
            magnitude = max(0.0, min(1.0, float(data.get("magnitude", 0.6))))
            if intent not in _VALID:
                raise ValueError(f"bad intent: {intent!r}")
        except Exception:  # noqa: BLE001 — 任何問題退回規則版（安全網）
            return self._fallback.parse(text, current_w)

        w = intent_to_weights(intent, magnitude if intent != "balanced" else 0.0)
        note = f"🦙 Ollama：偏向{_LABEL[intent]} → w＝延遲 {w[0]} / 成本 {w[1]}"
        return ControlResult(w=w, note=note, intent=intent, magnitude=magnitude)


def build_control(config: dict[str, Any]) -> ControlLLM:
    """依 config 的 control.provider 選控制層。

    provider=ollama 且 Ollama 在線 → OllamaControl（附規則版 fallback）；
    否則（含 GCP 沒 Ollama）→ 規則版。
    """
    provider = config.get("control", {}).get("provider", "rule")
    if provider == "ollama" and ollama_available():
        return OllamaControl(fallback=RuleBasedControl())
    return RuleBasedControl()
