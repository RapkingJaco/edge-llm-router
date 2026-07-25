# Python / 工具筆記

> 開發過程踩到的雷與慣用法。

## uv
- `uv sync --dev`：建 venv + 裝套件（含 dev group）。
- `uv run <cmd>`：在專案 venv 裡跑。
- `uv add <pkg>` / `uv add --dev <pkg>`：加相依。
- torch 綁特定 index：`[[tool.uv.index]]` + `[tool.uv.sources]`（見 pyproject）。

## 中文輸出
Windows console 預設非 UTF-8，印中文易亂碼。用 `sys.stdout.reconfigure(encoding="utf-8")`
或 `python -X utf8`。

## gitignore 眉角
`.env.*` 會一併忽略 `.env.example`，需加 `!.env.example` 例外。

*(建置中)*
