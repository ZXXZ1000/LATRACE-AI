# =============================================================================
# LATRACE Memory Service - Minimal Runtime Dockerfile
# =============================================================================
# This image builds and runs only the standalone `modules.memory` service.
#
# It does not bundle Qdrant or Neo4j into the same container.
# For the recommended self-hosted setup, use:
#   docker compose up --build
#
# Build:
#   docker build -t latrace-memory .
#
# Run against external dependencies:
#   docker run --rm -p 8000:8000 --env-file .env latrace-memory
# =============================================================================

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --frozen --no-dev

COPY modules/__init__.py modules/__init__.py
COPY modules/memory modules/memory

RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/modules /app/modules
COPY README.md LICENSE NOTICE ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "modules.memory.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
