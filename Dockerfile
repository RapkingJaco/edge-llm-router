# syntax=docker/dockerfile:1

# ── 前端 build ──────────────────────────────────────────────
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ── 後端 ────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
ENV UV_LINK_MODE=copy

# 先裝相依（layer 快取）
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# 再放設定 / 模型 / 前端產物
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/
COPY --from=web /web/dist ./web/dist

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "edge_llm_router.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
