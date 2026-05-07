"""Plot generation. matplotlib-only, no seaborn dependency."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..calibration.intensity import IntensityResult
from ..metrics.summary import BacktestSummary

FIGSIZE = (12, 8)
DPI = 100


def _ts_to_seconds(ts: np.ndarray) -> np.ndarray:
    if ts.size == 0:
        return np.asarray(ts.astype(np.float64))
    return np.asarray((ts - ts[0]).astype(np.float64) / 1e6)


def plot_pnl_comparison(
    summaries: Mapping[str, BacktestSummary],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name, s in summaries.items():
        x = _ts_to_seconds(s.pnl.timestamps)
        ax.plot(x, s.pnl.total_pnl, label=name)
    ax.set_xlabel("Time since start of test period (s)")
    ax.set_ylabel("Total PnL")
    ax.set_title("Out-of-sample PnL comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_inventory_comparison(
    summaries: Mapping[str, BacktestSummary],
    inventory_series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name, (ts, inv) in inventory_series.items():
        ax.plot(_ts_to_seconds(ts), inv, label=name, alpha=0.8)
    ax.set_xlabel("Time since start of test period (s)")
    ax.set_ylabel("Inventory")
    ax.set_title("Inventory time series")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_intensity_calibration(intensity: IntensityResult, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if not intensity.bucket_centers:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path)
        plt.close(fig)
        return
    x = np.asarray(intensity.bucket_centers)
    y = np.asarray(intensity.bucket_rates)
    mask = y > 0
    ax.scatter(x[mask], y[mask], label="empirical rate", color="steelblue")
    if intensity.A > 0 and intensity.k != 0:
        x_line = np.linspace(x[mask].min(), x[mask].max(), 200)
        ax.plot(x_line, intensity.A * np.exp(-intensity.k * x_line),
                label=f"fit A·exp(−k·δ), A={intensity.A:.3g}, k={intensity.k:.3g}",
                color="firebrick")
    ax.set_yscale("log")
    ax.set_xlabel("δ (distance from mid)")
    ax.set_ylabel("λ(δ) (events/s)")
    ax.set_title(f"Order arrival intensity calibration  (R² = {intensity.r_squared:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decomposition_bars(
    summaries: Mapping[str, BacktestSummary],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    names = list(summaries.keys())
    sc = [s.decomposition.spread_capture for s in summaries.values()]
    ip = [s.decomposition.inventory_pnl for s in summaries.values()]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, sc, width, label="Spread capture", color="seagreen")
    ax.bar(x + width / 2, ip, width, label="Inventory P&L", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("P&L")
    ax.set_title("P&L decomposition by strategy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
