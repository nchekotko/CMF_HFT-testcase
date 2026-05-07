from __future__ import annotations

import math

import numpy as np
import polars as pl

from cmf_mm.calibration.intensity import calibrate_intensity
from cmf_mm.calibration.volatility import estimate_sigma


def test_intensity_recovers_known_params(synthetic_intensity_data):
    """Known A=10, k=2 — the calibrator should recover both within ±10%."""
    res = calibrate_intensity(
        trades_parquet=synthetic_intensity_data["trades_path"],
        lob_parquet=synthetic_intensity_data["lob_path"],
        n_bins=20,
    )
    assert res.n_observations > synthetic_intensity_data["n"] * 0.95
    assert math.isclose(res.k, synthetic_intensity_data["k_true"], rel_tol=0.10)
    # A is more sensitive to binning; allow ±20%
    assert math.isclose(res.A, synthetic_intensity_data["A_true"], rel_tol=0.20)
    assert res.r_squared > 0.95


def test_volatility_recovers_known_sigma(tmp_path):
    """Generate a Brownian mid with known σ_per_sec, recover it."""
    rng = np.random.default_rng(0)
    sigma_true = 0.5  # per second
    n = 50_000
    dt_us = 1_000  # 1 ms steps
    dt_s = dt_us / 1e6
    increments = rng.normal(0.0, sigma_true * math.sqrt(dt_s), size=n)
    log_mid = np.cumsum(increments)
    mid = np.exp(log_mid) * 100.0
    ts = (np.arange(n) * dt_us).astype(np.int64)

    lob_path = tmp_path / "lob.parquet"
    pl.DataFrame({
        "ts": ts,
        "ask_px_0": mid + 0.01,
        "ask_sz_0": np.full(n, 1.0),
        "bid_px_0": mid - 0.01,
        "bid_sz_0": np.full(n, 1.0),
    }).write_parquet(lob_path)

    res = estimate_sigma(lob_path)
    assert res.n_observations > 0
    assert math.isclose(res.sigma_per_sec, sigma_true, rel_tol=0.05)
