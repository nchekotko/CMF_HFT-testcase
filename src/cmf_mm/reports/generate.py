"""Report generation entry point.

Loads pickled BacktestSummary objects from each strategy's results dir, plus
a calibration result, and emits figures + tables under a single output dir.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ..calibration.pipeline import CalibrationResult
from ..metrics.summary import BacktestSummary
from .figures import (
    plot_decomposition_bars,
    plot_intensity_calibration,
    plot_pnl_comparison,
)
from .tables import summary_latex, summary_markdown


def load_summary(path: str | Path) -> BacktestSummary:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, BacktestSummary):
        raise TypeError(f"{path} did not contain a BacktestSummary")
    return obj


def load_calibration(path: str | Path) -> CalibrationResult:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, CalibrationResult):
        raise TypeError(f"{path} did not contain a CalibrationResult")
    return obj


def generate_report(
    summary_paths: dict[str, str | Path],
    calibration_path: str | Path | None,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: load_summary(p) for name, p in summary_paths.items()}

    plot_pnl_comparison(summaries, out_dir / "pnl_comparison.png")
    plot_decomposition_bars(summaries, out_dir / "decomposition_bars.png")
    if calibration_path is not None:
        cal = load_calibration(calibration_path)
        plot_intensity_calibration(cal.intensity, out_dir / "intensity_calibration.png")

    md = summary_markdown(summaries)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.tex").write_text(summary_latex(summaries), encoding="utf-8")
