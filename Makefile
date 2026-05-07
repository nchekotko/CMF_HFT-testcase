.PHONY: install convert calibrate backtest backtest-v1 backtest-v2 backtest-v3 sweep test lint typecheck report all clean

# `uv sync --no-install-project` only installs dependencies; the package itself
# is found by setting PYTHONPATH=src — keeps the install step network-free
# w.r.t. the build backend.
PY = PYTHONPATH=src uv run python
PYTEST = PYTHONPATH=src uv run pytest
RUFF = uv run ruff
MYPY = PYTHONPATH=src uv run mypy

install:
	uv sync --no-install-project --extra dev

convert:
	$(PY) scripts/convert_to_parquet.py

calibrate:
	$(PY) scripts/calibrate.py --config configs/backtest_v2.yaml --out results/calibration

backtest-v1:
	$(PY) scripts/run_backtest.py --config configs/backtest_v1.yaml

backtest-v2:
	$(PY) scripts/run_backtest.py --config configs/backtest_v2.yaml

backtest-v3:
	$(PY) scripts/run_backtest.py --config configs/backtest_v3.yaml

backtest: backtest-v1 backtest-v2

sweep:
	$(PY) scripts/run_sweep.py --config configs/sweep.yaml

report:
	$(PY) scripts/generate_report.py --results-root results --out results/report

test:
	$(PYTEST) --cov=src/cmf_mm --cov-report=term-missing

lint:
	$(RUFF) check src scripts tests

typecheck:
	$(MYPY) --strict src/cmf_mm

all: convert backtest report

clean:
	rm -rf results/* data/*.parquet .pytest_cache .mypy_cache .ruff_cache
