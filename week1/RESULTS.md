# Week 1 — Liquidation signal: results

Filter Binance maker trades per horizon τ ∈ {30, 120, 300}s so maker PnL on the **kept**
trades beats PnL on **all** trades, keeping ≥ $500k/day clipped turnover.
`Score(τ) = PnL_kept(τ) − PnL_all(τ)`, higher is better.

## TL;DR
- A **fixed-sign** liquidation rule does **not** generalize — the liq→return relationship
  flips sign between Dec and Feb (regime non-stationarity). Net OOS edge ≈ 0.
- A **weighted gradient-boosting** model (predict markout-PnL, filter the worst) generalizes:
  positive Score on both train-CV and held-out Feb. **ETH is positive at all three horizons**
  (τ30/120/300); **BTC at τ30 and τ300** (τ120 has no edge → no-filter).
- τ=120 is symbol-dependent: a **dead horizon for BTC** (no feature set gives edge, so CV picks
  no-filter, Score floor 0) but **alive for ETH** (val +1.16, CV picks keep 0.40).
- **Order-flow imbalance (VI)** is a ~10–30s signal: it lifts Score at τ=30 (+0.08 OOS) but
  hurts at τ=300 (−0.15). Production uses VI **only at τ=30** (per-horizon feature set).
- Turnover constraint is non-binding: kept turnover runs 6–15 B$/day vs the 500k$/day floor.

## Data
- Binance trades / booktickers (BBO) / liquidations + Bybit liquidations, `perp:btcusdt` &
  `perp:ethusdt`. `timestamp` = int64 µs UTC.
- Range **2025-11-01 → 2026-04-28** (~6 months, 179 days). ~14 GB total; trades alone are
  0.8–1.4 B rows/symbol. Split: train Dec'25–Jan'26, val Feb'26, hidden test later.
- Conventions: trade `side` = taker (buy ⇒ maker sold); Bybit events usable only after +200 ms
  (cross-exchange latency). `w_i = min(notional, 100k)`;
  `pnl_i(τ) = −s_i·(mid(t+τ)−p)/p·1e4 + 0.5` bps (forward-filled BBO mid; excluded if t+τ past BBO).

## Model
Per-τ `HistGradientBoostingRegressor`, `sample_weight = w_i`, labels winsorised ±50 bps,
predicts markout-PnL; filter = drop the lowest-predicted-PnL trades. Kept-fraction per τ chosen
by forward-chaining time-series CV inside train (grid includes keep=1.0 ⇒ no-filter when no edge).
Features (causal): side, log_notional, log_cluster (sweeper size), spread, hour, realised vol,
momentum 5/60s (aligned), Bybit/Binance liq pressure (aligned + magnitude), and — at τ=30 only —
trade-volume-imbalance VI over {5,30,120}s (aligned, raw + volume-normalised, from a causal 1s flow grid).

## Headline results (val = Feb'26, 14 days held out), Score in bps
Production retrain, per-horizon feature set (τ30 = base+VI, τ120/300 = base). `keep` chosen by
train-CV; turnover scaled back to full-day from the per-day subsample.

| τ | BTC (keep) | BTC Score | ETH (keep) | ETH Score |
|---|---|---|---|---|
| 30s  | 0.40 | **+0.337** | 0.40 | **+1.083** |
| 120s | 1.00 | **0.000** (no-filter) | 0.40 | **+1.161** |
| 300s | 0.80 | **+0.155** | 0.40 | **+1.433** |

BTC PnL_all is negative at every τ (−0.13/−0.17/−0.14 bps — pure adverse selection); the filter
turns τ30/τ300 net-positive on kept trades. ETH PnL_all is mildly positive in Feb and the filter
lifts kept-PnL to +1.26/+1.21/+1.55. Turnover 5.7–15 B$/day everywhere (≫ 500k floor).

**Caveat:** ETH's large magnitudes ride a favourable Feb regime (positive baseline maker-PnL);
the *sign* is real (positive in both CV and val) but don't expect ~+1 bps on the hidden test. The
robust, regime-surviving claim is BTC τ30/τ300 and the fact that both symbols stay positive OOS.

## Experiments

### 1. Fixed-sign signal is non-stationary (why ML)
At τ=300 the liquidation-reversal rule scores **train −0.47 / val +0.36**; the momentum rule is the
exact mirror (+0.44 / −0.50). Whichever sign you fix helps in one regime and hurts equally in the
other → ≈0 OOS. (`eval_signals.py`)

### 2. Horizon term structure (BTC, val), base features
| τ | 1s | 5s | 10s | 30s | 120s | 300s |
|---|---|---|---|---|---|---|
| val Score | +0.231 | +0.305 | **+0.355** | +0.256 | 0.000 | +0.242 |

Edge does **not** die at short τ — it peaks ~τ=10s. (BTC spread ≈ 0.013 bps, so there is no
half-spread to capture; PnL is pure adverse-selection, which the model ranks well.) (`explore_horizons.py`)

### 3. Trade-volume-imbalance ablation (BTC, val): base vs base+VI
| τ | 1s | 5s | 10s | 30s | 120s | 300s |
|---|---|---|---|---|---|---|
| Δ Score from VI | +0.010 | +0.015 | +0.021 | **+0.084** | 0.000 | **−0.153** |

Inverted-U: VI helps up to τ=30, nothing at 120, hurts at 300 → VI is a 10–30s order-flow signal.
Best single config seen: **τ=10s + VI = +0.376**. (`explore_volimb.py`)

## Files
- `ml_baseline.py` — submission (`classify_trades`) + training/eval; per-horizon feature sets.
- `baseline.py` — heuristic baseline + shared data-loading / markout-PnL / metrics.
- `eval_signals.py`, `explore_horizons.py`, `explore_volimb.py` — diagnostics/experiments.
- `ml_artifacts_BTC.joblib`, `ml_artifacts_ETH.joblib` — trained models + keep-fractions + VI map.
- `README_baseline.md` — how to run.

## Engineering notes
14 GB data on a 7.4 GB box: everything streams one day at a time; eval subsamples per day
(`gather_every`) with full-day cluster/flow grids streamed separately; never materialises a full
multi-million-row day. `classify_trades` handles single- or mixed-symbol input.

## Next
1. ~~Confirm the VI-at-τ30 upgrade on ETH (retrain).~~ **Done** — ETH retrained per-horizon;
   val τ30 **+1.083**, τ120 **+1.161**, τ300 **+1.433** (all keep 0.40). VI helps τ30 as on BTC.
2. τ=120 is dead **only for BTC** (no-filter). For ETH it's a live horizon — worth probing why
   (ETH liquidation cascades persist longer?). For BTC τ120, try OFI from BBO (queue-imbalance).
3. Cross-symbol features (BTC liqs → ETH and vice versa); regime-robust evaluation (rolling windows).
4. ETH magnitudes are regime-inflated — re-check on a less-favourable window before trusting levels.
