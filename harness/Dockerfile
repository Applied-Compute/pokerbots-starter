# Build from this directory:
#   docker build -t poker-harness .
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml ./
COPY src ./src
RUN uv sync --no-dev --no-editable

FROM python:3.12-slim

WORKDIR /app
COPY --from=build /app/.venv .venv
COPY shard_router.py run_sharded.sh ./

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8765 POKER_HARNESS_RUNS=/tmp/runs

USER 65532:65532
EXPOSE 8765
ENTRYPOINT ["./run_sharded.sh"]
