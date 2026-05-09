"""Quantify the AS-vs-book-spread regime mismatch on the test split.

This is the centerpiece of the defense narrative: AS prescribes a half-spread
that is much wider than the natural top-of-book spread, so all three strategies
sit off-touch and V2's micro-price shift is below the fill quantum.
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

# UTF-8 stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    cal = pickle.load(open("results/v1/calibration.pkl", "rb"))
    sigma = cal.volatility.sigma_per_sec
    A = cal.intensity.A
    k = cal.intensity.k

    # Backtest parameters (from configs/backtest_v*.yaml)
    gamma = 200.0
    T_minus_t = 1.0  # infinite-horizon limit (T_horizon_seconds=0 → 1s constant)

    # AS spread formula: δ_a + δ_b = γ·σ²·(T-t) + (2/γ)·log(1 + γ/k)
    risk_term = gamma * sigma**2 * T_minus_t
    spread_term = (2.0 / gamma) * math.log(1.0 + gamma / k)
    full_spread = risk_term + spread_term
    half_spread = 0.5 * full_spread

    print("=" * 72)
    print("AS QUOTE GEOMETRY  (γ=200, T-t=1s, σ/k from train)")
    print("=" * 72)
    print(f"  σ                 = {sigma:.6e}")
    print(f"  k                 = {k:.6e}")
    print(f"  γ·σ²·(T-t)        = {risk_term:.6e}   (inventory risk term)")
    print(f"  (2/γ)·log(1+γ/k)  = {spread_term:.6e}   (spread term, ≥ 2/γ·log(1)·... ≈ 0)")
    print(f"  Full AS spread    = {full_spread:.6e}")
    print(f"  AS HALF-SPREAD    = {half_spread:.6e}")

    print()
    print("Sanity: Stoikov lower bound 1/k =", f"{1.0/k:.6e}")

    # Natural book spread on test split
    lob = pl.read_parquet("data/lob.parquet")
    # Find chronological split: same as runner — sort by timestamp, take last 30%
    lob = lob.sort("ts")
    n = lob.height
    test_start = lob["ts"][int(n * 0.7)]
    test = lob.filter(pl.col("ts") >= test_start)

    bid_top = test["bid_px_0"].to_numpy()
    ask_top = test["ask_px_0"].to_numpy()
    bid_qty_top = test["bid_sz_0"].to_numpy()
    ask_qty_top = test["ask_sz_0"].to_numpy()
    mid = 0.5 * (bid_top + ask_top)
    spread = ask_top - bid_top

    # Volume-weighted micro-price (V2 reference)
    micro = (bid_qty_top * ask_top + ask_qty_top * bid_top) / (bid_qty_top + ask_qty_top)
    micro_dev = np.abs(micro - mid)

    print()
    print("=" * 72)
    print("NATURAL TOP-OF-BOOK SPREAD  (TEST split, last 30%)")
    print("=" * 72)
    print(f"  N snapshots       = {len(spread):,}")
    print(f"  mid (mean)        = {np.mean(mid):.6e}")
    print(f"  mid (last)        = {mid[-1]:.6e}")
    print(f"  spread mean       = {np.mean(spread):.6e}")
    print(f"  spread median     = {np.median(spread):.6e}")
    print(f"  spread p99        = {np.percentile(spread, 99):.6e}")

    print()
    print("=" * 72)
    print("REGIME MISMATCH  (the punchline)")
    print("=" * 72)
    ratio_full = full_spread / np.mean(spread)
    ratio_half = half_spread / np.mean(spread)
    print(f"  AS full spread    = {full_spread:.6e}")
    print(f"  Book spread mean  = {np.mean(spread):.6e}")
    print(f"  RATIO (full/book) = {ratio_full:.2f}x")
    print(f"  RATIO (half/book) = {ratio_half:.2f}x")
    print(f"  → AS quotes sit ~{ratio_half:.0f}× the book spread away from mid.")

    print()
    print("=" * 72)
    print("WHY V2 ≡ V1  (micro-price deviation vs AS half-spread)")
    print("=" * 72)
    print(f"  |p_micro - mid| mean   = {np.mean(micro_dev):.6e}")
    print(f"  |p_micro - mid| median = {np.median(micro_dev):.6e}")
    print(f"  AS half-spread         = {half_spread:.6e}")
    quantum_ratio = half_spread / np.mean(micro_dev)
    print(f"  → AS half-spread is ~{quantum_ratio:.0f}× larger than typical micro-price shift.")
    print(f"    The reference-price shift V2 introduces is below the fill quantum.")


if __name__ == "__main__":
    main()
