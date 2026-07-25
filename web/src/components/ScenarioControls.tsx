export function ScenarioControls({ peakOn, onPeak, onReset }: {
  peakOn: boolean;
  onPeak: (on: boolean) => void;
  onReset: () => void;
}) {
  return (
    <div className="controls">
      <span className="controls-label">情境：</span>
      <button className={`btn ${peakOn ? "btn-active" : ""}`} onClick={() => onPeak(!peakOn)}>
        {peakOn ? "尖峰中 ✓" : "製造尖峰"}
      </button>
      <button className="btn" onClick={onReset}>重置</button>
    </div>
  );
}
