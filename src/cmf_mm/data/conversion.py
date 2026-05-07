"""CSV → parquet conversion.

Uses polars' streaming engine so a 1 GB CSV does not need to fit in RAM.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .schema import LOB_LEVELS, lob_csv_schema, trades_csv_schema


def convert_trades(csv_path: str | Path, parquet_path: str | Path) -> None:
    """Stream-convert trades.csv to parquet.

    Drops the unnamed index column, renames ``local_timestamp`` to ``ts``,
    and keeps ``side`` as a UTF-8 column with values restricted to
    {"buy", "sell"} (the dataset uses these aggressor labels directly).
    """
    csv_path, parquet_path = Path(csv_path), Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(csv_path, schema=trades_csv_schema()).select(
        pl.col("local_timestamp").alias("ts"),
        pl.col("side"),
        pl.col("price"),
        pl.col("amount"),
    )
    lf.sink_parquet(parquet_path, compression="zstd", compression_level=3)


def convert_lob(
    csv_path: str | Path,
    parquet_path: str | Path,
    n_levels: int = LOB_LEVELS,
) -> None:
    """Stream-convert lob.csv to parquet, flattening the asks[i]/bids[i] columns.

    The output column order is ``ts, ask_px_0, ask_sz_0, bid_px_0, bid_sz_0, ...``.
    """
    csv_path, parquet_path = Path(csv_path), Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    selectors: list[pl.Expr] = [pl.col("local_timestamp").alias("ts")]
    for i in range(n_levels):
        selectors.append(pl.col(f"asks[{i}].price").alias(f"ask_px_{i}"))
        selectors.append(pl.col(f"asks[{i}].amount").alias(f"ask_sz_{i}"))
        selectors.append(pl.col(f"bids[{i}].price").alias(f"bid_px_{i}"))
        selectors.append(pl.col(f"bids[{i}].amount").alias(f"bid_sz_{i}"))

    lf = pl.scan_csv(csv_path, schema=lob_csv_schema(n_levels)).select(selectors)
    lf.sink_parquet(parquet_path, compression="zstd", compression_level=3)
