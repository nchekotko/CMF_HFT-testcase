"""Read the project's real L2 order-book parquet into L2 snapshots.

Bridges the existing HW1-HW2 dataset (``data/lob.parquet`` — flat 25-level L2
snapshots, see :mod:`cmf_mm.data.schema`) into the :mod:`simlob` world. The rows
are turned into :data:`~cmf_mm.simlob.adapters.L2Snapshot` tuples, which feed the
:class:`~cmf_mm.simlob.adapters.L2ToL3Converter` to drive a HistoricalLOB — i.e.
``lob.parquet -> L2 snapshots -> synthetic L3 -> HistoricalLOB``. This is the
"represents the real market" path; the production path swaps the reader for the
Databento MBO adapter without touching the core.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from ..data.schema import LOB_LEVELS
from .adapters import L2Snapshot

DEFAULT_LOB_PATH = Path("data/lob.parquet")


def _level_columns(n_levels: int) -> list[str]:
    cols = ["ts"]
    for i in range(n_levels):
        cols += [f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"]
    return cols


def iter_lob_snapshots(
    path: str | Path = DEFAULT_LOB_PATH,
    *,
    n_levels: int = LOB_LEVELS,
    limit: int | None = None,
    stride: int = 1,
) -> Iterator[L2Snapshot]:
    """Yield ``(ts, bids, asks)`` snapshots from the L2 parquet, in file order.

    ``bids``/``asks`` are lists of ``(price, size)`` with empty/NaN levels
    dropped. ``limit`` caps the number of *rows scanned*; ``stride`` keeps every
    ``stride``-th snapshot (1 = all). Uses a lazy scan so a slice does not load
    the whole 1M-row file.
    """
    lf = pl.scan_parquet(path).select(_level_columns(n_levels))
    if limit is not None:
        lf = lf.head(limit)
    df = lf.collect()

    ask_px = [f"ask_px_{i}" for i in range(n_levels)]
    ask_sz = [f"ask_sz_{i}" for i in range(n_levels)]
    bid_px = [f"bid_px_{i}" for i in range(n_levels)]
    bid_sz = [f"bid_sz_{i}" for i in range(n_levels)]

    for idx, row in enumerate(df.iter_rows(named=True)):
        if idx % stride != 0:
            continue
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        for pxc, szc in zip(bid_px, bid_sz, strict=True):
            px, sz = row[pxc], row[szc]
            if px is not None and sz is not None and sz > 1e-12 and px > 0.0:
                bids.append((float(px), float(sz)))
        for pxc, szc in zip(ask_px, ask_sz, strict=True):
            px, sz = row[pxc], row[szc]
            if px is not None and sz is not None and sz > 1e-12 and px > 0.0:
                asks.append((float(px), float(sz)))
        yield (int(row["ts"]), bids, asks)
