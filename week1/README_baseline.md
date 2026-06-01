# Week 1 baseline — liquidation signal for filtering Binance maker trades

We passively collect Binance taker flow as a maker. For each trade we decide keep (0) or
filter (1) per horizon τ ∈ {30, 120, 300}s so that maker PnL on the *kept* trades beats
PnL on *all* trades, while keeping ≥ $500k/day of clipped turnover.

## Two deliverables
| file | what |
|---|---|
| [ml_baseline.py](ml_baseline.py) | **the submission** — weighted gradient-boosting model. `classify_trades(trades, bbo, liq_binance, liq_bybit) -> {τ: 0/1 array}` |
| [baseline.py](baseline.py) | heuristic baseline + shared data loading / markout-PnL / metrics helpers |
| [eval_signals.py](eval_signals.py) | diagnostic comparison of fixed-sign signals (showed they don't generalize) |

## Key finding (why ML)
A fixed-sign liquidation rule (e.g. "filter taker trades that run *into* recent liq
pressure") is **non-stationary**: at τ=300s it scores train −0.47 / val +0.36 — the sign of
the relationship flips between Dec and Feb, so any fixed threshold nets ≈ 0 out-of-sample.
See `eval_signals.py`. So we let a model learn a regime-conditional, non-linear rule.

## ML baseline
- **Per-τ** `HistGradientBoostingRegressor` predicting maker markout-PnL, `sample_weight =
  w_i = min(notional, 100k)`, labels winsorised to ±50 bps.
- **Features** (all causal, from the four input frames): `side_sign`, `log_notional`,
  `log_cluster` (sweeper size), `spread_bps`, `hour`, `vol_60s` (realised vol),
  `mom_5s/mom_60s` aligned to maker side, and Bybit/Binance liq pressure (aligned + magnitude,
  a few lookbacks). Realised vol + recent momentum act as **regime proxies** so the model can
  adapt instead of committing to one sign. **Per-horizon set:** τ=30 also gets trade-volume-
  imbalance (VI) over {5,30,120}s (raw + volume-normalised, from a causal 1s flow grid); τ=120/300
  use base only (VI is a ~10–30s signal that hurts at the long horizon).
- **Filter**: drop the trades with the lowest predicted PnL. The kept fraction per τ is chosen
  by **forward-chaining time-series CV inside the train window** (no peeking at validation).
  `keep = 1.0` is in the grid, so when a horizon has no edge the model filters *nothing*
  (Score floor 0) rather than hurting.

## Results (train = Dec'25–Jan'26 CV, val = Feb'26 held out, 14 days)

Production retrain with the **per-horizon** feature set (τ30 = base+VI, τ120/300 = base).

**BTC**
| τ | keep | CV Score | **val Score** | PnL_all → PnL_kept (val) | turnover/day |
|---|---|---|---|---|---|
| 30s | 0.40 | +0.448 | **+0.337** | −0.125 → **+0.212** | $6.8B |
| 120s | 1.00 | ≤0 (no edge) | **0.000** | −0.167 (no filter) | $15.5B |
| 300s | 0.80 | +0.167 | **+0.155** | −0.135 → **+0.019** | $12.1B |

**ETH**
| τ | keep | CV Score | **val Score** | PnL_all → PnL_kept (val) | turnover/day |
|---|---|---|---|---|---|
| 30s | 0.40 | +0.410 | **+1.083** | +0.175 → **+1.257** | $6.0B |
| 120s | 0.40 | +0.211 | **+1.161** | +0.053 → **+1.214** | $5.9B |
| 300s | 0.40 | +0.106 | **+1.433** | +0.115 → **+1.548** | $5.7B |

The model is positive on **both** train (CV) and val: BTC at τ=30/300, ETH at all three horizons —
it survives the Dec→Feb regime change that broke the fixed-sign rule. τ=120 is a **dead horizon for
BTC** (no edge → no-filter, Score floor 0) but **alive for ETH** (+1.16). Turnover ≫ $500k/day
everywhere (~4 orders of magnitude of headroom).

Caveat: ETH val Score ≫ its CV because Feb's baseline maker-PnL was *positive* for ETH (a
favourable regime); the signal's sign is real (positive in both CV and val) but its magnitude
is regime-dependent — don't expect +1.0 bps on the hidden test.

## Run
```bash
python ml_baseline.py train BTC          # build, time-series CV, fit -> ml_artifacts_BTC.joblib, eval on val
python ml_baseline.py train ETH          # same for ETH
python ml_baseline.py train ETH noeval   # train only, skip the in-process eval (lower peak RAM)
python ml_baseline.py eval  BTC          # load artifacts, evaluate on val only (memory-safe sampled)
python baseline.py                       # heuristic baseline + turnover-constraint check
```
On a low-RAM box (this one has < 0.5 GB free), set `POLARS_MAX_THREADS=1` to cut peak memory and
just re-run on a crash: both `train` and `eval` **checkpoint per day** to `_traincache_<SYM>/` and
`_evalcache_<SYM>/` and skip days already done, so repeated runs resume and converge. Delete those
dirs to force a clean rebuild.
`classify_trades` auto-selects the per-symbol model from the trades' ticker (handles mixed-symbol
input too); BTC is the fallback if a symbol-specific artifact is missing.

## Notes / conventions
- `timestamp` = int64 microseconds UTC (the only time axis).
- `side` in trades is the **taker** side (`buy` ⇒ maker sold); in liquidations it's the
  liquidation-order side. Bybit events are shifted **+200 ms** (cross-exchange visibility).
- `pnl_i(τ) = -s_i·(mid(t+τ)-p)/p·1e4 + 0.5` bps (forward-filled Binance mid; trade excluded
  if t+τ is beyond available BBO). `w_i = min(notional, 100k)`.
- Full tables are ~14 GB. Everything loads **one day at a time**; features are built per row
  block via `FeatureContext`, and both train/eval checkpoint per day, so peak memory stays low
  and a transient spike from other apps costs at most one day's rework.
- Both symbols evaluated on held-out Feb'26 (above). Next: cross-symbol features and a
  less-favourable validation window to pressure-test ETH's regime-inflated magnitudes.
