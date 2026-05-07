"""σ calibration from mid-price log-returns.

We compute σ in **per-second** units and let the strategy multiply by the
relevant horizon. This is the most stable convention: it is independent of
sampling frequency and trivially composes with the AS time-to-horizon term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class VolatilityResult:
    sigma_per_sec: float
    n_observations: int
    mean_dt_sec: float


def estimate_sigma(
    lob_parquet: str | Path,
    start_ts: int | None = None,
    end_ts: int | None = None,
    sample_stride: int = 1,
) -> VolatilityResult:
    """Estimate σ in per-second units from LOB mid-price log-returns.

    Resampling stride lets us subsample large LOB streams without changing
    the per-second variance estimate (we divide by the realized inter-sample
    Δt, not by stride).
    """
    lf = pl.scan_parquet(lob_parquet)
    if start_ts is not None:
        lf = lf.filter(pl.col("ts") >= start_ts)
    if end_ts is not None:
        lf = lf.filter(pl.col("ts") < end_ts)
    df = lf.select("ts", "ask_px_0", "bid_px_0").collect()
    if df.height < 2:
        return VolatilityResult(sigma_per_sec=0.0, n_observations=0, mean_dt_sec=0.0)

    if sample_stride > 1:
        df = df.gather_every(sample_stride)

    ts = df["ts"].to_numpy()
    mid = 0.5 * (df["ask_px_0"].to_numpy() + df["bid_px_0"].to_numpy())
    log_mid = np.log(mid)
    dlog = np.diff(log_mid)
    dt = np.diff(ts) / 1e6  # μs → seconds
    mask = dt > 0
    dlog = dlog[mask]
    dt = dt[mask]
    if dlog.size == 0:
        return VolatilityResult(sigma_per_sec=0.0, n_observations=0, mean_dt_sec=0.0)

    # Variance per second: average of (Δlog)² / Δt
    var_per_sec = float(np.mean(dlog * dlog / dt))
    return VolatilityResult(
        sigma_per_sec=float(np.sqrt(var_per_sec)),
        n_observations=int(dlog.size),
        mean_dt_sec=float(np.mean(dt)),
    )
