"""Parameter sweep using joblib.

The grid is specified in YAML using dotted keys (e.g. ``strategy.gamma``).
We expand the cartesian product, override the base config for each cell,
fan out via joblib, then collect a pareto-friendly summary CSV.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import polars as pl
from joblib import Parallel, delayed

from cmf_mm.config import BacktestConfig, load_sweep_config
from cmf_mm.runner import run_from_config


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _expand_grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        merged = json.loads(json.dumps(base))  # deep copy via json round-trip
        for k, v in zip(keys, combo, strict=True):
            _set_dotted(merged, k, v)
        # Each cell writes to its own results dir
        cell_id = "__".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo, strict=True))
        merged.setdefault("output", {})["results_dir"] = f"results/sweep/{cell_id}"
        merged["name"] = f"sweep::{cell_id}"
        out.append(merged)
    return out


def _run_cell(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = BacktestConfig.model_validate(cfg_dict)
    art = run_from_config(cfg)
    import pickle
    with open(art.summary_path, "rb") as f:
        s = pickle.load(f)
    return {
        "name": cfg.name,
        "strategy": cfg.strategy.name,
        "gamma": cfg.strategy.gamma,
        "T_horizon_seconds": cfg.calibration.T_horizon_seconds,
        "total_pnl": s.pnl.final(),
        "max_dd": s.pnl.max_drawdown(),
        "inv_std": s.inventory.std,
        "inv_max_abs": s.inventory.max_abs,
        "spread_capture": s.decomposition.spread_capture,
        "inventory_pnl": s.decomposition.inventory_pnl,
        "n_fills": s.n_fills,
        "fill_rate": s.fill_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/sweep/sweep_results.csv")
    args = parser.parse_args()

    sweep = load_sweep_config(args.config)
    base_dict = sweep.base.model_dump()
    cells = _expand_grid(base_dict, sweep.grid)

    rows = Parallel(n_jobs=sweep.n_jobs, verbose=10)(
        delayed(_run_cell)(c) for c in cells
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)
    print(f"Sweep wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
