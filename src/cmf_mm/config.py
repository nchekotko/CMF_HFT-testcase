"""Pydantic-validated config models.

Configs are validated at the entry point — invalid YAML files fail fast,
not three hours into a sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    trades_parquet: str
    lob_parquet: str
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)


class StrategyConfig(BaseModel):
    name: Literal["as_vanilla", "as_microprice", "as_asymmetric"]
    gamma: float = Field(gt=0.0)
    order_size: float = Field(gt=0.0)
    max_inventory: float = Field(gt=0.0)
    quote_refresh_min_interval_us: int = Field(default=100_000, ge=0)
    alpha: float | None = None  # only for as_asymmetric

    @model_validator(mode="after")
    def _check_alpha(self) -> StrategyConfig:
        if self.name == "as_asymmetric" and self.alpha is None:
            raise ValueError("as_asymmetric requires alpha")
        return self


class CalibrationConfig(BaseModel):
    sigma_sample_stride: int = Field(default=1, ge=1)
    intensity_n_bins: int = Field(default=20, ge=5)
    T_horizon_seconds: int = Field(default=3600)


class ExecutionConfig(BaseModel):
    partial_fills: bool = False
    fee_bps: float = 0.0


class OutputConfig(BaseModel):
    results_dir: str = "results/default"
    sample_every_n_events: int = Field(default=1, ge=1)


class BacktestConfig(BaseModel):
    name: str = "backtest"
    data: DataConfig
    strategy: StrategyConfig
    calibration: CalibrationConfig
    execution: ExecutionConfig = ExecutionConfig()
    output: OutputConfig = OutputConfig()


class SweepConfig(BaseModel):
    base: BacktestConfig
    grid: dict[str, list[Any]]
    n_jobs: int = -1


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at the top level")
    return data


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig.model_validate(load_yaml(path))


def load_sweep_config(path: str | Path) -> SweepConfig:
    return SweepConfig.model_validate(load_yaml(path))
