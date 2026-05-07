"""Streaming event loader.

Reads trades.parquet and lob.parquet and yields a chronologically merged
stream of ``TradeEvent`` and ``LOBEvent`` objects. We use a two-pointer merge
on already-sorted parquet files; the source data is sorted by timestamp.

For tie-breaking we emit the trade first when timestamps coincide: trades
are executed against the *pre-update* book, and the LOB snapshot at the
same timestamp reflects the *post-trade* state. Processing the trade first
attributes the fill against the same book the matcher saw, then the book
updates to the new state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from ..types import Event, LOBEvent, TradeEvent


def _read_trades(path: str | Path, start_ts: int | None, end_ts: int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(path)
    if start_ts is not None:
        lf = lf.filter(pl.col("ts") >= start_ts)
    if end_ts is not None:
        lf = lf.filter(pl.col("ts") < end_ts)
    return lf.select("ts", "side", "price", "amount").collect()


def _read_lob(path: str | Path, start_ts: int | None, end_ts: int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(path)
    if start_ts is not None:
        lf = lf.filter(pl.col("ts") >= start_ts)
    if end_ts is not None:
        lf = lf.filter(pl.col("ts") < end_ts)
    return lf.select("ts", "ask_px_0", "ask_sz_0", "bid_px_0", "bid_sz_0").collect()


def stream_events(
    trades_path: str | Path,
    lob_path: str | Path,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[Event]:
    """Merge trades + LOB events in chronological order.

    Implementation: load both into memory as polars DataFrames (the LOB has
    only ~1M rows; trades has ~22M, both sorted by timestamp), then iterate
    via two pointers. Emits LOB event first when timestamps tie.
    """
    trades_df = _read_trades(trades_path, start_ts, end_ts)
    lob_df = _read_lob(lob_path, start_ts, end_ts)

    t_ts = trades_df["ts"].to_numpy()
    t_side = trades_df["side"].to_numpy()
    t_price = trades_df["price"].to_numpy()
    t_size = trades_df["amount"].to_numpy()

    l_ts = lob_df["ts"].to_numpy()
    l_ap = lob_df["ask_px_0"].to_numpy()
    l_as = lob_df["ask_sz_0"].to_numpy()
    l_bp = lob_df["bid_px_0"].to_numpy()
    l_bs = lob_df["bid_sz_0"].to_numpy()

    n_t = len(t_ts)
    n_l = len(l_ts)
    i = j = 0

    while i < n_t or j < n_l:
        take_lob = j < n_l and (i >= n_t or l_ts[j] < t_ts[i])
        if take_lob:
            yield LOBEvent(
                ts=int(l_ts[j]),
                bid_px=float(l_bp[j]),
                bid_sz=float(l_bs[j]),
                ask_px=float(l_ap[j]),
                ask_sz=float(l_as[j]),
            )
            j += 1
        else:
            yield TradeEvent(
                ts=int(t_ts[i]),
                price=float(t_price[i]),
                size=float(t_size[i]),
                aggressor_side="buy" if t_side[i] == "buy" else "sell",
            )
            i += 1
