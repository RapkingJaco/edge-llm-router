"""規則版控制層：關鍵字 → 意圖 → 權重。離線、免 API key，永遠是可用的 fallback。"""

from __future__ import annotations

from .base import ControlLLM, ControlResult, WeightPair, intent_to_weights

# 偏向「省成本」的線索（含「多用本機/邊緣」這種等價說法）。
_CHEAPER = (
    "成本", "省", "便宜", "貴", "花費", "錢", "省錢", "少花", "預算",
    "本機", "邊緣", "cost", "cheap", "budget", "local", "edge",
)
# 偏向「低延遲/快」的線索（含「多用雲」）。
_FASTER = (
    "快", "延遲", "慢", "速度", "即時", "反應", "首字", "順",
    "雲", "latency", "fast", "speed", "cloud",
)
_BALANCED = ("平衡", "均衡", "兼顧", "一半", "折衷", "balance")
_STRONG = ("太", "非常", "很", "極", "最", "儘量", "盡量", "大幅", "超", "務必", "狠")


def _describe(intent: str, magnitude: float, w: WeightPair) -> str:
    label = {"cheaper": "省成本", "faster": "低延遲", "balanced": "平衡"}[intent]
    deg = "強" if magnitude >= 0.7 else "適中"
    return f"偏向{label}（{deg}）→ w＝延遲 {w[0]} / 成本 {w[1]}"


class RuleBasedControl(ControlLLM):
    """關鍵字比對版。看不懂就維持原方針（安全網）。"""

    name = "rule"

    def parse(self, text: str, current_w: WeightPair) -> ControlResult:
        t = text.strip()
        if not t:
            return ControlResult(current_w, "空白輸入，維持原方針", "unknown", 0.0)

        cheaper = sum(k in t for k in _CHEAPER)
        faster = sum(k in t for k in _FASTER)
        has_balanced = any(k in t for k in _BALANCED)
        magnitude = 0.8 if any(k in t for k in _STRONG) else 0.55

        if cheaper == 0 and faster == 0:
            if has_balanced:
                intent = "balanced"
            else:
                return ControlResult(current_w, f"看不懂「{t}」，維持原方針", "unknown", 0.0)
        elif cheaper == faster:
            intent = "balanced"
        elif cheaper > faster:
            intent = "cheaper"
        else:
            intent = "faster"

        w = intent_to_weights(intent, magnitude if intent != "balanced" else 0.0)
        return ControlResult(w, _describe(intent, magnitude, w), intent, magnitude)
