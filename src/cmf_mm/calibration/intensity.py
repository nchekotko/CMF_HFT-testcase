"""λ(δ) = A · exp(−k · δ) calibration via log-linear regression.

Procedure:
  1. For each trade in the calibration window, find the most recent LOB
     snapshot; compute the side-aware distance δ from the mid:
       buy-aggressor:  δ = trade.price − mid          (positive ⇒ ask side)
       sell-aggressor: δ = mid − trade.price          (positive ⇒ bid side)
     We pool both sides — the AS model assumes symmetric arrivals. The
     fitted A is therefore the *pooled* density: it equals the sum of the
     per-side densities at δ=0. Only k enters the optimal-spread formula,
     so the pooled-vs-per-side distinction is cosmetic for the strategy.
  2. Bin δ into log-spaced buckets, compute rate per bucket as
     count / total_window_seconds.
  3. Log-linear fit log λ(δ) = log A − k · δ with weighted least squares
     (weights ∝ √count to dampen near-zero buckets).

We deliberately drop the smallest δ bucket if it contains zero observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class IntensityResult:
    A: float
    k: float
    r_squared: float
    n_observations: int
    bucket_centers: tuple[float, ...]
    bucket_rates: tuple[float, ...]


def _join_trades_with_mid(
    trades_parquet: str | Path,
    lob_parquet: str | Path,
    start_ts: int | None,
    end_ts: int | None,
) -> pl.DataFrame:
    trades_lf = pl.scan_parquet(trades_parquet).select("ts", "side", "price")
    lob_lf = pl.scan_parquet(lob_parquet).select(
        "ts",
        ((pl.col("ask_px_0") + pl.col("bid_px_0")) * 0.5).alias("mid"),
    )
    if start_ts is not None:
        trades_lf = trades_lf.filter(pl.col("ts") >= start_ts)
        lob_lf = lob_lf.filter(pl.col("ts") >= start_ts)
    if end_ts is not None:
        trades_lf = trades_lf.filter(pl.col("ts") < end_ts)
        lob_lf = lob_lf.filter(pl.col("ts") < end_ts)
    trades_df = trades_lf.collect().sort("ts")
    lob_df = lob_lf.collect().sort("ts")
    # Backward asof: each trade gets the most recent LOB snapshot at-or-before its ts.
    return trades_df.join_asof(lob_df, on="ts", strategy="backward")


def calibrate_intensity(
    trades_parquet: str | Path,
    lob_parquet: str | Path,
    start_ts: int | None = None,
    end_ts: int | None = None,
    n_bins: int = 20,
    delta_floor_quantile: float = 0.01,
    delta_cap_quantile: float = 0.99,
) -> IntensityResult:
    """Calibrate (A, k) from real or synthetic data.

    ``delta_floor_quantile`` / ``delta_cap_quantile`` clip extreme δ values
    that otherwise dominate the regression with a few outliers.
    """
    df = _join_trades_with_mid(trades_parquet, lob_parquet, start_ts, end_ts)
    df = df.drop_nulls("mid")
    if df.height == 0:
        return IntensityResult(A=0.0, k=0.0, r_squared=0.0, n_observations=0,
                               bucket_centers=(), bucket_rates=())

    side = df["side"].to_numpy()
    price = df["price"].to_numpy()
    mid = df["mid"].to_numpy()
    ts = df["ts"].to_numpy()

    delta = np.where(side == "buy", price - mid, mid - price)
    delta = delta[delta > 0]  # drop crossing trades or wrongly-classified ticks
    if delta.size < n_bins * 5:
        return IntensityResult(A=0.0, k=0.0, r_squared=0.0, n_observations=int(delta.size),
                               bucket_centers=(), bucket_rates=())

    lo = float(np.quantile(delta, delta_floor_quantile))
    hi = float(np.quantile(delta, delta_cap_quantile))
    lo = max(lo, 1e-12)
    if hi <= lo:
        hi = lo * 10.0
    edges = np.geomspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    counts, _ = np.histogram(delta, bins=edges)

    window_seconds = (int(ts.max()) - int(ts.min())) / 1e6
    if window_seconds <= 0:
        return IntensityResult(A=0.0, k=0.0, r_squared=0.0, n_observations=int(delta.size),
                               bucket_centers=(), bucket_rates=())
    # rate density per second per unit-δ. Required to make log λ linear in δ
    # under λ(δ) = A·exp(-k·δ); using raw counts on log-spaced bins biases k.
    rates = counts / (window_seconds * widths)

    mask = counts > 0
    if mask.sum() < 3:
        return IntensityResult(A=0.0, k=0.0, r_squared=0.0, n_observations=int(delta.size),
                               bucket_centers=tuple(centers.tolist()),
                               bucket_rates=tuple(rates.tolist()))
    x = centers[mask]
    y = np.log(rates[mask])
    w = np.sqrt(counts[mask])

    # Weighted log-linear fit: y = a − k·x  (a = log A)
    W = np.diag(w)
    X = np.column_stack([np.ones_like(x), -x])
    XtW = X.T @ W
    beta, *_ = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)
    a, k = float(beta[0]), float(beta[1])
    A = float(np.exp(a))

    y_hat = a - k * x
    ss_res = float(np.sum(w * (y - y_hat) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return IntensityResult(
        A=A,
        k=k,
        r_squared=r2,
        n_observations=int(delta.size),
        bucket_centers=tuple(centers.tolist()),
        bucket_rates=tuple(rates.tolist()),
    )
