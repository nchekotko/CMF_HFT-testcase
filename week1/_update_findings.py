"""Replace findings cell with version that includes strategy headline."""
import json
from pathlib import Path

PATH = Path("week1_exploration.ipynb")
nb = json.load(open(PATH, encoding="utf-8"))

new_body = """---
## 18. Findings log

**Date range:** 2025-11-01 00:00 → 2026-04-28 23:59 UTC (179 days, full 6 months).

**Volumes:**
- binance trades BTC = 804 M rows (4.5 M/day) · ETH = 1.37 B rows (7.7 M/day)
- binance bbo    BTC = 203 M · ETH = 220 M (≈1.2 M/day each)
- binance liq    BTC = 236 K (1.3 K/day) · ETH = 271 K (1.5 K/day)
- bybit   liq    BTC = 438 K (2.4 K/day) · ETH = 302 K (1.7 K/day) — **Bybit ≈ 2× Binance on BTC**

**Timestamp unit:** µs UTC ✓ (magnitude 1.76 × 10¹⁵, decodes to plausible dates).

**`side` semantics:**
- trades: **taker side ✓** (in a 1-hour BTC window, 99.3 % of `side=buy` trades sit above mid; 99.3 % of `side=sell` below mid)
- liquidations: liquidation-order side; in bear regime, sell-liqs > buy-liqs ✓ (e.g. Binance BTC: 131 k sell vs 105 k buy; Bybit BTC: 280 k sell vs 158 k buy)

**Duplicate timestamps (trades):**
- BTC: 80 % of rows share their µs with another row (uniq ratio 0.21)
- ETH: 84 % (uniq ratio 0.16)
- BBO and Binance liqs: 100 % unique. Bybit liqs: ~98 % unique
- Interpretation: one taker event splits into N maker-fills inside a single µs. The cluster size of this split turns out to be a powerful signal (see Strategy backtests).

**Trade size distribution:**
- BTC: median \\$232, p99 \\$49 k, p99.9 \\$146 k, p99.99 \\$500 k — clip at \\$100 k catches **0.23 %** of trades
- ETH: median \\$41, p99 \\$27 k, p99.9 \\$86 k, p99.99 \\$300 k — clip catches **0.07 %**
- Side balance ≈ 50/50 (as expected for taker side)
- 6-month gross taker volume: BTC \\$2.4 T, ETH \\$2.17 T

**BBO spread:**
- BTC: median **0.013 bps** (p99 0.016) — essentially always 1 tick
- ETH: median **0.037 bps** (p99 0.055) — wider in bps because tick is bigger fraction of price
- 0 crossed / 0 locked across the full window
- median bid/ask size: BTC 4 / 4 BTC; ETH 62.9 / 62.3 ETH

**Liquidation magnitudes:**
- Biggest single liq seen: Binance BTC \\$12.6 M, Binance ETH \\$12.1 M; Bybit max ≈ \\$3.5 M

**BBO around a Binance liquidation (median signed move):**
- t ± 100 ms: 0 bps; t + 1 s: 0; t + 10 s: ±0.04
- **t + 60 s: −0.84 bps for buy-liqs, +0.46 bps for sell-liqs** — mean-reversion *opposite* to liq direction over the next minute. Crucially, this is what flipped Strategy 1 from \\"momentum\\" to \\"reversal\\".

**Bybit → Binance impulse response:**
- Monotonically rising through +2 s. The claimed 200 ms is the information-availability delay we must respect, not the impulse-response peak.

**Cross-exchange liquidation alignment:**
- BTC: 43 % of Binance liqs have a same-side Bybit liq within ±2 s. Median Δt = −56 ms (Bybit fires first).
- ETH: 38 % matched, median Δt = −96 ms.
- After the +200 ms info delay, Bybit liq becomes visible ≈ 144–104 ms *after* the matching Binance liq. Bybit can't anticipate concurrent Binance liqs, but is a useful seconds-long directional pressure signal.

**Surprises:**
- Trades-duplicate-µs ratio of 80 % on BTC, 84 % on ETH — single events become many maker-fill rows.
- Bybit reports more BTC liquidations than Binance over the same period.

---

## 19. Strategy backtest results (BTC, τ = 120 s, val out-of-sample)

| strategy | val_score (bps) | kept_frac | turn/day (USD) |
|---|---:|---:|---:|
| baseline (no filter) | 0.000 | 1.00 | 9.6 B |
| S1 — Bybit momentum (ORIG, *wrong sign*) | −0.186 | 0.94 | 8.6 B |
| **S1b — Bybit reversal, 5 s lookback** | **+0.078** | 0.94 | 8.9 B |
| S2 — Binance reversal, 60 s | −0.044 | 0.72 | 6.8 B |
| S3 — filter LARGE (>\\$500 k) | +0.019 | 1.00 | 9.5 B |
| S3b — filter SMALL (<\\$50 k) | −0.377 | 0.01 | 1.9 B |
| **S4 — cluster size ≥ 50** | **+0.106** | 0.77 | 6.6 B |
| **S5 — ensemble (S1b ∨ S2 ∨ S4)** | **+0.308** | 0.54 | 4.5 B |

Constraint \\$500 k/day kept turnover is satisfied with 9000× headroom for the ensemble.

**Takeaways:**
1. **Cluster size (S4) is the strongest single signal.** Same-µs same-side groupings of ≥ 50 maker-fills mark an urgent sweeper; the maker on the other side is adversely selected. Out-of-sample +0.11 bps at τ=120 s, +0.16 bps at τ=300 s.
2. **Bybit-liquidation pressure is a reversal signal, not momentum.** With a 5 s lookback (after the mandated 200 ms gate), filtering the side opposite to recent Bybit pressure adds +0.08–0.19 bps to maker PnL on val.
3. **Big trades are not toxic** in this market — institutional flow is sliced via algos, so single large takes tend to come from emotional retail (mean-revert favourable to the maker). Filtering them HURTS.
4. **Ensemble works:** S1b ∨ S2 ∨ S4 with thresholds picked on train gives **+0.31 bps val score at τ = 120 s** — ~1.6× the baseline maker PnL on top of the 0.5 bps rebate floor, while keeping 54 % of trade count and \\$4.5 B/day turnover.

**Next-week directions:**
- Re-run all backtests on ETH; check that S4 cluster generalises.
- Build a gradient-boosted classifier with all 6 features (S1b/S2/S3/S4/spread/hour) and weighted samples; predict per-trade per-τ profitability.
- Add cross-symbol features (BTC liqs → ETH trades and vice versa).
- Regime-aware filter: realised-vol bucketed thresholds, since baseline itself shifts train ↔ val.
"""

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "## 18. Findings log" in "".join(c["source"]):
        c["source"] = new_body.splitlines(keepends=True)
        print(f"updated findings cell at index {i}")
        break

json.dump(nb, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
