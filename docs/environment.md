# 開發環境

## 這台家用機（2026-07-24 實測）

| 工具 | 版本 / 狀態 |
|---|---|
| Python | 由 uv 管理，實際用 **3.12.13** |
| uv | 0.11.27 |
| torch | 2.13.0**+cpu**（綁 PyTorch CPU index） |
| Node.js | 在 PATH（前端 Vite/React/TS） |
| Ollama | 0.32.1，已有模型 `llama3.2:latest`(2GB) |
| GPU | RTX 4070 SUPER / 12GB（只給 Ollama） |
| Docker | 29.6.1 |
| gcloud | 在 PATH |

> Python 版本註記：原訂 3.11，但 uv 在此機安裝 3.11 卡在建 minor-version link 步驟；
> 改用已就緒的 3.12，全部套件相容，對專案無影響。

## 常用指令（一律走 uv，勿用 pip/conda）

```bash
uv sync --dev                      # 建 venv + 裝全部套件（含開發工具）
uv run python scripts/check_env.py # 環境就緒檢查
uv run ruff check                  # lint（--fix 自動修）
uv run pytest -q                   # 跑測試
```

## GPU / CPU 分工

- **PPO 訓練用 CPU**：policy 是十幾維輸入的小 MLP，運算量極小，CPU 足夠且避開 CUDA 版本地雷。
- **GPU 只給 Ollama**（Phase 6 邊緣節點真跑）。
- 因此 `torch` 綁 PyTorch **CPU** 官方 index（見 `pyproject.toml` 的 `[tool.uv.sources]`），
  確保 Windows 本機與 Linux CI 都拿 CPU wheel、不誤抓 2GB+ CUDA 包。

## 中文輸出

Windows console 預設非 UTF-8，Python 印中文易亂碼。腳本開頭已用
`sys.stdout.reconfigure(encoding="utf-8")` 保險；必要時也可用 `python -X utf8`。
