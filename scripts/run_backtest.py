"""Run a single backtest from a YAML config."""

from __future__ import annotations

import argparse

from cmf_mm.config import load_backtest_config
from cmf_mm.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_backtest_config(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
