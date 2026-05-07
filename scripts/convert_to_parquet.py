"""One-off: stream-convert the source CSVs to zstd-compressed parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from cmf_mm.data.conversion import convert_lob, convert_trades
from cmf_mm.utils.logging import get_logger
from cmf_mm.utils.timing import timed


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV → parquet converter")
    parser.add_argument("--trades-csv", default="trades.csv")
    parser.add_argument("--lob-csv", default="lob.csv")
    parser.add_argument("--trades-parquet", default="data/trades.parquet")
    parser.add_argument("--lob-parquet", default="data/lob.parquet")
    args = parser.parse_args()

    log = get_logger()
    Path("data").mkdir(parents=True, exist_ok=True)

    with timed(f"convert_trades({args.trades_csv})", sink=log):
        convert_trades(args.trades_csv, args.trades_parquet)
    with timed(f"convert_lob({args.lob_csv})", sink=log):
        convert_lob(args.lob_csv, args.lob_parquet)


if __name__ == "__main__":
    main()
