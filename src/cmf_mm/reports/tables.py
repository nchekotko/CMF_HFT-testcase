"""Markdown / LaTeX summary tables."""

from __future__ import annotations

from collections.abc import Mapping

from ..metrics.summary import BacktestSummary

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _per_event_sharpe(s: BacktestSummary) -> float:
    if s.duration_seconds <= 0:
        return 0.0
    n = s.pnl.total_pnl.size
    if n < 2:
        return 0.0
    periods_per_year = n * SECONDS_PER_YEAR / s.duration_seconds
    return s.pnl.sharpe(periods_per_year)


def summary_markdown(summaries: Mapping[str, BacktestSummary]) -> str:
    header = (
        "| Strategy | Total PnL | Sharpe (ann) | Max DD | Inv mean | Inv std | "
        "Spread capt | Inv PnL | N fills | Fill rate |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for name, s in summaries.items():
        rows.append(
            f"| {name} "
            f"| {s.pnl.final():.6g} "
            f"| {_per_event_sharpe(s):.3f} "
            f"| {s.pnl.max_drawdown():.6g} "
            f"| {s.inventory.mean:.4g} "
            f"| {s.inventory.std:.4g} "
            f"| {s.decomposition.spread_capture:.6g} "
            f"| {s.decomposition.inventory_pnl:.6g} "
            f"| {s.n_fills} "
            f"| {s.fill_rate:.4f} |"
        )
    return "\n".join(rows)


def summary_latex(summaries: Mapping[str, BacktestSummary]) -> str:
    rows: list[str] = []
    rows.append(r"\begin{tabular}{lrrrrrrrrr}")
    rows.append(r"\hline")
    rows.append(
        r"Strategy & Total PnL & Sharpe & Max DD & Inv mean & Inv std & "
        r"Spread capt & Inv PnL & N fills & Fill rate \\"
    )
    rows.append(r"\hline")
    for name, s in summaries.items():
        rows.append(
            f"{name} & {s.pnl.final():.4g} & {_per_event_sharpe(s):.3f} & "
            f"{s.pnl.max_drawdown():.4g} & {s.inventory.mean:.4g} & "
            f"{s.inventory.std:.4g} & {s.decomposition.spread_capture:.4g} & "
            f"{s.decomposition.inventory_pnl:.4g} & {s.n_fills} & "
            f"{s.fill_rate:.4f} \\\\"
        )
    rows.append(r"\hline")
    rows.append(r"\end{tabular}")
    return "\n".join(rows)
