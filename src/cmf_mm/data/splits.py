"""Chronological train/test split.

Random splits are explicitly forbidden — they leak information across the
time axis and overstate out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


def chronological_split(
    parquet_path: str | Path,
    train_ratio: float = 0.7,
    ts_col: str = "ts",
) -> SplitBoundaries:
    """Returns split boundaries based on the timestamp range of ``parquet_path``.

    Train interval is ``[train_start_ts, train_end_ts)``; test interval is
    ``[test_start_ts, test_end_ts)``. The split point is placed so that
    ``train_ratio`` of the wall-clock duration is in train.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    df = pl.scan_parquet(parquet_path).select(
        pl.col(ts_col).min().alias("min"),
        pl.col(ts_col).max().alias("max"),
    ).collect()
    t_min = int(df["min"][0])
    t_max = int(df["max"][0])
    if t_max <= t_min:
        raise ValueError(f"Degenerate timestamp range in {parquet_path}: [{t_min}, {t_max}]")

    split_ts = t_min + int((t_max - t_min) * train_ratio)
    return SplitBoundaries(
        train_start_ts=t_min,
        train_end_ts=split_ts,
        test_start_ts=split_ts,
        test_end_ts=t_max + 1,
    )
