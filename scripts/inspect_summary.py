"""Pretty-print existing BacktestSummary + CalibrationResult for the defense walkthrough."""

from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np

# Ensure UTF-8 stdout on Windows consoles (default cp1251 chokes on σ).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def fmt(x: float, w: int = 12) -> str:
    return f"{x:>{w}.6g}"


def show(run_dir: Path) -> None:
    name = run_dir.name
    summary = pickle.load(open(run_dir / "summary.pkl", "rb"))
    cal = pickle.load(open(run_dir / "calibration.pkl", "rb"))

    print(f"\n{'='*72}\n{name.upper()}  ({summary.config_name})\n{'='*72}")

    # Calibration
    print(f"\n[Calibration on TRAIN]")
    print(f"  σ (per √s)        = {cal.volatility.sigma_per_sec:.6g}")
    print(f"  σ n observations  = {cal.volatility.n_observations:,}")
    print(f"  σ mean Δt (s)     = {cal.volatility.mean_dt_sec:.4f}")
    print(f"  A                 = {cal.intensity.A:.6g}")
    print(f"  k                 = {cal.intensity.k:.6g}")
    print(f"  intensity R²      = {cal.intensity.r_squared:.4f}")
    print(f"  intensity n obs   = {cal.intensity.n_observations:,}")

    # Pnl
    pnl = summary.pnl
    inv = summary.inventory
    dec = summary.decomposition
    print(f"\n[P&L on TEST]")
    print(f"  Total PnL         = {pnl.final():.6f}")
    print(f"  Spread capture    = {dec.spread_capture:.6f}")
    print(f"  Inventory PnL     = {dec.inventory_pnl:.6f}")
    invariant = dec.spread_capture + dec.inventory_pnl
    err = abs(invariant - dec.total)
    print(f"  Invariant check   = {invariant:.6f}  (|err|={err:.2e}, tol={dec.tol})")

    # Inventory
    print(f"\n[Inventory]")
    print(f"  mean              = {inv.mean:.4f}")
    print(f"  std               = {inv.std:.4f}")
    print(f"  max |q|           = {inv.max_abs:.0f}")
    print(f"  time-weighted avg = {inv.time_weighted_avg:.4f}")
    print(f"  final             = {inv.final:.4f}")

    # Activity
    dur_h = summary.duration_seconds / 3600.0
    print(f"\n[Activity]")
    print(f"  duration          = {dur_h:.2f} h ({summary.duration_seconds:.0f} s)")
    print(f"  n trades (test)   = {summary.n_trades:,}")
    print(f"  n fills           = {summary.n_fills:,}")
    print(f"  fill rate (per tr)= {summary.fill_rate*100:.4f}%")
    print(f"  turnover (notional)= {summary.turnover:.2f}")

    # PnL trajectory snapshot
    if pnl.total_pnl is not None and len(pnl.total_pnl) > 0:
        tp = pnl.total_pnl
        print(f"\n[PnL trajectory snapshot]")
        idx_q = [int(len(tp) * q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        idx_q[-1] = len(tp) - 1
        for q, i in zip([0, 25, 50, 75, 100], idx_q):
            print(f"  P{q:>3}  i={i:>7}  total={tp[i]:>10.4f}")
        print(f"  PnL  min={tp.min():.4f}  max={tp.max():.4f}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    for sub in ("v1", "v2", "v3"):
        d = root / sub
        if (d / "summary.pkl").exists():
            show(d)
