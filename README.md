# CMF Market Making Backtester

Event-driven backtester for single-asset limit-order-book market making, with
three variants of the Avellaneda–Stoikov strategy compared on a held-out test
period of a real crypto LOB dataset (≈21.86M trades, ≈1.04M LOB snapshots).

**Headline (out-of-sample, last 30% chronological split):**

| Strategy | Total PnL | Spread capture | Inventory P&L | Inv std | N fills |
|---|---:|---:|---:|---:|---:|
| V1 vanilla | −4.965 | +1.376 | −6.341 | 29.96 | 52,111 |
| V2 micro-price | −4.961 | +1.376 | −6.338 | 29.92 | 52,133 |
| V3 asymmetric | −14.74 | −0.264 | −14.48 | 45.34 | 154,629 |

The full discussion of *why* V1 and V2 are operationally indistinguishable on
this dataset (regime mismatch between AS-prescribed half-spread and natural
book spread) is in [REPORT.md](REPORT.md).

## Quickstart

```bash
# Build & run end-to-end (CSV → parquet → calibrate → backtest → report)
docker compose up convert        # one-off CSV → parquet
docker compose up backtester     # default: V2 (microprice) on test split

# Or locally:
make install
make convert      # writes data/trades.parquet, data/lob.parquet
make backtest     # runs V1 + V2 on the held-out test split
make report       # builds figures + summary tables under results/report/
```

A full walk-through is in [REPORT.md](REPORT.md). The improvement plan with
academic references is in [ROADMAP.md](ROADMAP.md).

## What's inside

```
src/cmf_mm/
├── data/             # CSV→parquet conversion, streaming event loader, splits
├── lob/              # OrderBookState, micro-price estimators
├── engine/           # event loop, matcher, order manager
├── strategy/         # base ABC + AS vanilla / AS micro-price / AS asymmetric
├── calibration/      # σ, A, k from training data
├── metrics/          # PnL series, inventory stats, P&L decomposition
└── reports/          # matplotlib figures, markdown/LaTeX tables
```

## Strategies

| Variant | Reference price | Spread | Skew |
|---|---|---|---|
| **V1** AS vanilla | mid | symmetric | none |
| **V2** AS micro-price | volume-weighted micro-price | symmetric | none |
| **V3** AS asymmetric (extension) | micro-price | symmetric | imbalance-driven (α·I) |

V1 and V2 differ **only** in the reference price — same calibration, same
parameters, same execution rules. This makes the comparison clean.

## P&L decomposition

Total P&L is split into:

* **Spread capture** — the bid–ask edge earned on each fill, measured against
  the engine-tracked mid at fill time.
* **Inventory P&L** — mark-to-market on inventory as the mid moves.

The identity `spread_capture + inventory_pnl ≡ total` is enforced as a class
invariant in `metrics/decomposition.py` and verified in `tests/test_metrics.py`.

## Reproducibility

* All deps pinned via `uv.lock`.
* Train/test split is **chronological** (first 70% by wall-clock).
* Calibration is fit on the train half, frozen, and applied to the test half.
* Strategies see only one event at a time — no look-ahead.

## Running tests

```bash
make test         # pytest with coverage
make lint         # ruff
make typecheck    # mypy --strict on src/cmf_mm
```

## Data

The dataset (~1 GB CSVs) lives at the repo root as `trades.csv` and `lob.csv`.
The schema is documented in `src/cmf_mm/data/schema.py`. Timestamps are in
microseconds; LOB has 25 levels per side (only the top is used by the engine).
