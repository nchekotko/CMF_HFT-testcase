FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=600 \
    PYTHONPATH=/app/src

# uv handles environment + lockfile reproducibility
RUN pip install --no-cache-dir uv==0.4.30

WORKDIR /app

# Cache dependency layer (lockfile copied if present)
COPY pyproject.toml ./
COPY uv.lock* ./
RUN uv sync --no-install-project --extra dev

# Project sources
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY Makefile README.md ./

# Default: run V2 backtest (assumes parquet files mounted at /app/data).
CMD ["uv", "run", "python", "scripts/run_backtest.py", "--config", "configs/backtest_v2.yaml"]
