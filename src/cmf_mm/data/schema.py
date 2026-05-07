"""Schema definitions for the input CSVs and the canonical parquet format.

The dataset uses microsecond timestamps. We preserve μs throughout the engine.

Trades CSV columns:
    , local_timestamp, side, price, amount

LOB CSV columns:
    , local_timestamp, asks[0].price, asks[0].amount, bids[0].price,
    bids[0].amount, ..., asks[24].price, asks[24].amount, bids[24].price,
    bids[24].amount

The parquet representation drops the unnamed index column and renames
``local_timestamp`` to ``ts`` for brevity. LOB levels are kept flat
(``ask_px_0``, ``ask_sz_0``, ...) to keep ingestion linear-time without
list/struct wrapping.
"""

from __future__ import annotations

import polars as pl

LOB_LEVELS: int = 25  # depth in raw lob.csv


def trades_csv_schema() -> dict[str, pl.DataType]:
    return {
        "": pl.Int64(),  # unnamed index column from pandas-style export
        "local_timestamp": pl.Int64(),
        "side": pl.String(),
        "price": pl.Float64(),
        "amount": pl.Float64(),
    }


def lob_csv_schema(n_levels: int = LOB_LEVELS) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "": pl.Int64(),
        "local_timestamp": pl.Int64(),
    }
    for i in range(n_levels):
        schema[f"asks[{i}].price"] = pl.Float64()
        schema[f"asks[{i}].amount"] = pl.Float64()
        schema[f"bids[{i}].price"] = pl.Float64()
        schema[f"bids[{i}].amount"] = pl.Float64()
    return schema


def trades_parquet_columns() -> list[str]:
    return ["ts", "side", "price", "amount"]


def lob_parquet_columns(n_levels: int = LOB_LEVELS) -> list[str]:
    cols = ["ts"]
    for i in range(n_levels):
        cols.extend([f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"])
    return cols
