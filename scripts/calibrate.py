"""Standalone calibration runner — useful for diagnostics independent of a strategy."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from cmf_mm.calibration.pipeline import calibrate_all
from cmf_mm.config import load_backtest_config
from cmf_mm.data.splits import chronological_split
from cmf_mm.reports.figures import plot_intensity_calibration
from cmf_mm.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone σ + (A, k) calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/calibration")
    args = parser.parse_args()

    cfg = load_backtest_config(args.config)
    log = get_logger()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = chronological_split(cfg.data.lob_parquet, train_ratio=cfg.data.train_ratio)
    log.info("Train window: [%d, %d) μs", split.train_start_ts, split.train_end_ts)

    cal = calibrate_all(
        trades_parquet=cfg.data.trades_parquet,
        lob_parquet=cfg.data.lob_parquet,
        train_start_ts=split.train_start_ts,
        train_end_ts=split.train_end_ts,
        n_intensity_bins=cfg.calibration.intensity_n_bins,
        sigma_sample_stride=cfg.calibration.sigma_sample_stride,
    )

    log.info("σ_per_sec=%.6g  A=%.6g  k=%.6g  R²=%.3f",
             cal.volatility.sigma_per_sec, cal.intensity.A,
             cal.intensity.k, cal.intensity.r_squared)

    with open(out / "calibration.pkl", "wb") as f:
        pickle.dump(cal, f)
    plot_intensity_calibration(cal.intensity, out / "intensity_calibration.png")


if __name__ == "__main__":
    main()
