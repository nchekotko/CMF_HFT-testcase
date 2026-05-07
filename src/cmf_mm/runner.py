"""Glue between config and engine — used by both CLI scripts and the sweep.

A `run_from_config` call performs a full pipeline:
    1. Compute the chronological train/test split from the LOB parquet.
    2. Calibrate σ and (A, k) on the train half.
    3. Build the configured strategy with the calibrated parameters.
    4. Stream the test half through the event loop.
    5. Pickle the BacktestSummary + CalibrationResult into the output dir.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from .calibration.pipeline import CalibrationResult, calibrate_all
from .config import BacktestConfig, StrategyConfig
from .data.loader import stream_events
from .data.splits import SplitBoundaries, chronological_split
from .engine.event_loop import BacktestConfig as EngineConfig
from .engine.event_loop import BacktestResult, run_backtest
from .strategy.as_asymmetric import ASAsymmetricParams, ASAsymmetricStrategy
from .strategy.as_microprice import ASMicropriceStrategy
from .strategy.avellaneda_stoikov import ASParams, AvellanedaStoikovStrategy
from .strategy.base import Strategy
from .utils.logging import get_logger

LOG = get_logger("cmf_mm.runner")


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    summary_path: Path
    calibration_path: Path
    split: SplitBoundaries
    calibration: CalibrationResult


def _build_strategy(cfg: StrategyConfig, sigma: float, k: float, T_horizon_us: int) -> Strategy:
    p = ASParams(
        gamma=cfg.gamma,
        sigma=sigma,
        k=k,
        T_horizon_us=T_horizon_us,
        order_size=cfg.order_size,
        max_inventory=cfg.max_inventory,
        quote_refresh_min_interval_us=cfg.quote_refresh_min_interval_us,
    )
    if cfg.name == "as_vanilla":
        return AvellanedaStoikovStrategy(p)
    if cfg.name == "as_microprice":
        return ASMicropriceStrategy(p)
    if cfg.name == "as_asymmetric":
        ap = ASAsymmetricParams(
            gamma=p.gamma,
            sigma=p.sigma,
            k=p.k,
            T_horizon_us=p.T_horizon_us,
            order_size=p.order_size,
            max_inventory=p.max_inventory,
            quote_refresh_min_interval_us=p.quote_refresh_min_interval_us,
            alpha=cfg.alpha or 0.0,
        )
        return ASAsymmetricStrategy(ap)
    raise ValueError(f"unknown strategy: {cfg.name}")


def run_from_config(cfg: BacktestConfig) -> RunArtifacts:
    out_dir = Path(cfg.output.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Computing chronological split (train_ratio=%.3f)", cfg.data.train_ratio)
    split = chronological_split(cfg.data.lob_parquet, train_ratio=cfg.data.train_ratio)
    LOG.info(
        "Train: [%d, %d), test: [%d, %d)",
        split.train_start_ts, split.train_end_ts,
        split.test_start_ts, split.test_end_ts,
    )

    LOG.info("Calibrating σ and (A, k) on train period")
    calibration = calibrate_all(
        trades_parquet=cfg.data.trades_parquet,
        lob_parquet=cfg.data.lob_parquet,
        train_start_ts=split.train_start_ts,
        train_end_ts=split.train_end_ts,
        n_intensity_bins=cfg.calibration.intensity_n_bins,
        sigma_sample_stride=cfg.calibration.sigma_sample_stride,
    )
    LOG.info(
        "σ=%.6g/s, A=%.6g, k=%.6g, R²=%.3f",
        calibration.volatility.sigma_per_sec,
        calibration.intensity.A,
        calibration.intensity.k,
        calibration.intensity.r_squared,
    )

    sigma = calibration.volatility.sigma_per_sec
    k = max(calibration.intensity.k, 1e-6)
    T_horizon_us = max(cfg.calibration.T_horizon_seconds * 1_000_000, 0)

    strategy = _build_strategy(cfg.strategy, sigma=sigma, k=k, T_horizon_us=T_horizon_us)

    engine_cfg = EngineConfig(
        config_name=cfg.name,
        partial_fills=cfg.execution.partial_fills,
        fee_bps=cfg.execution.fee_bps,
        sample_every_n_events=cfg.output.sample_every_n_events,
    )

    LOG.info("Running backtest on test period")
    events = stream_events(
        cfg.data.trades_parquet,
        cfg.data.lob_parquet,
        start_ts=split.test_start_ts,
        end_ts=split.test_end_ts,
    )
    result: BacktestResult = run_backtest(events, strategy, engine_cfg)
    LOG.info(
        "Done: %d trades, %d fills, total PnL=%.6g, inv std=%.4g",
        result.summary.n_trades,
        result.summary.n_fills,
        result.summary.pnl.final(),
        result.summary.inventory.std,
    )

    summary_path = out_dir / "summary.pkl"
    calibration_path = out_dir / "calibration.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(result.summary, f)
    with open(calibration_path, "wb") as f:
        pickle.dump(calibration, f)

    return RunArtifacts(
        summary_path=summary_path,
        calibration_path=calibration_path,
        split=split,
        calibration=calibration,
    )
