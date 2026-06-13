"""Append Strategy backtest cells (15-18) to week1_exploration.ipynb."""
import json
from pathlib import Path

PATH = Path("week1_exploration.ipynb")
nb = json.load(open(PATH, encoding="utf-8"))

def md(id_, src):
    return {"cell_type":"markdown","id":id_,"metadata":{},"source":src.splitlines(keepends=True)}

def code(id_, src):
    return {"cell_type":"code","id":id_,"execution_count":None,"metadata":{},"outputs":[],
            "source":src.splitlines(keepends=True)}

new_cells = []

# === Section 15 — framework ===
new_cells.append(md("s15", """## 15. Strategy backtests — first cuts

Now that exploration is done, try the four hypothesised filters on a sample. For each Binance trade:

1. compute **markout PnL** = $-s_i \\cdot (m(t+\\tau) - p_i)/p_i \\cdot 10^4 + 0.5$ bps, for $\\tau \\in \\{30, 120, 300\\}$ s
2. compute **weight** $w_i = \\min(\\text{notional}_i, 100{,}000)$
3. compute one binary filter $f_i$ per strategy
4. **Score**$(\\tau) = \\text{PnL}_\\text{kept} - \\text{PnL}_\\text{all}$ (want > 0); **constraint**: $\\sum (1-f_i) w_i / \\text{days} \\geq \\$500{,}000$

Split (per `description.md`):
- **train**: 2025-12-01 → 2026-01-31 (2 mo)
- **val**:   2026-02-01 → 2026-02-28 (1 mo)

To keep Colab runtime sane: sample ~100k trades from train, ~50k from val (uniform downsample). BTC only for now."""))

new_cells.append(code("c-bt-setup", """\
TRAIN_LO = int(datetime(2025,12,1, tzinfo=timezone.utc).timestamp() * 1_000_000)
TRAIN_HI = int(datetime(2026, 2,1, tzinfo=timezone.utc).timestamp() * 1_000_000)
VAL_LO   = TRAIN_HI
VAL_HI   = int(datetime(2026, 3,1, tzinfo=timezone.utc).timestamp() * 1_000_000)
TRAIN_DAYS = 62
VAL_DAYS   = 28
SYM = "BTC"
TURNOVER_MIN = 500_000  # USD per day
TAUS_S = [30, 120, 300]
TAUS_US = [t * 1_000_000 for t in TAUS_S]
CLIP = 100_000.0
print(f"train: {TRAIN_DAYS}d  {fmt_ts(TRAIN_LO)} .. {fmt_ts(TRAIN_HI)}")
print(f"val:   {VAL_DAYS}d  {fmt_ts(VAL_LO)} .. {fmt_ts(VAL_HI)}")"""))

new_cells.append(code("c-bt-sample", """\
def sample_trades(sym: str, t_lo: int, t_hi: int, target: int) -> pl.DataFrame:
    n_total = scan("binance","trades",sym).filter(
        (pl.col("timestamp") >= t_lo) & (pl.col("timestamp") < t_hi)
    ).select(pl.len()).collect(engine="streaming").item()
    step = max(1, n_total // target)
    df = (
        scan("binance","trades",sym)
        .filter((pl.col("timestamp") >= t_lo) & (pl.col("timestamp") < t_hi))
        .gather_every(step)
        .collect(engine="streaming")
        .sort("timestamp")
    )
    return df

train = sample_trades(SYM, TRAIN_LO, TRAIN_HI, target=100_000)
val   = sample_trades(SYM, VAL_LO,   VAL_HI,   target=50_000)
print(f"sampled {SYM}: train={len(train):,}  val={len(val):,}")"""))

new_cells.append(code("c-bt-markout", """\
def markout_pnl_bps(sample: pl.DataFrame, sym: str, tau_us: int, chunk: int = 5000) -> np.ndarray:
    \"\"\"Return per-trade markout PnL in bps (incl. +0.5 rebate). NaN if t+tau outside BBO range.\"\"\"
    ts    = sample["timestamp"].to_numpy()
    price = sample["price"].to_numpy()
    sgn   = np.where(sample["side"].to_numpy() == "buy", +1.0, -1.0)
    target = ts + tau_us
    mids = np.full(len(sample), np.nan)
    for start in range(0, len(sample), chunk):
        end = min(start + chunk, len(sample))
        t_lo = int(target[start])   - 5_000_000
        t_hi = int(target[end - 1]) + 5_000_000
        bbo = (
            window_lf(scan("binance", "bbo", sym), t_lo, t_hi)
              .select(["timestamp", "bid_price", "ask_price"])
              .sort("timestamp")
              .collect()
        )
        if bbo.is_empty(): continue
        b_ts  = bbo["timestamp"].to_numpy()
        b_mid = (bbo["bid_price"].to_numpy() + bbo["ask_price"].to_numpy()) * 0.5
        tgt = target[start:end]
        idx = np.searchsorted(b_ts, tgt, side="right") - 1
        valid = (idx >= 0) & (tgt >= b_ts[0]) & (tgt <= b_ts[-1])
        mids[start:end] = np.where(valid, b_mid[np.clip(idx, 0, len(b_mid) - 1)], np.nan)
    return -sgn * (mids - price) / price * 1e4 + 0.5

def attach_pnl(sample: pl.DataFrame, sym: str) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    out_pnl = {}
    for tau_s, tau_us in zip(TAUS_S, TAUS_US):
        out_pnl[tau_s] = markout_pnl_bps(sample, sym, tau_us)
        print(f"  τ={tau_s:>3}s  pnl computed, valid={np.isfinite(out_pnl[tau_s]).mean()*100:.1f}%")
    notional = (sample["price"] * sample["amount"]).to_numpy()
    w        = np.minimum(notional, CLIP)
    sample = sample.with_columns([
        pl.Series("notional", notional),
        pl.Series("weight",   w),
    ])
    return sample, out_pnl

print("computing markout for train")
train, train_pnl = attach_pnl(train, SYM)
print("computing markout for val")
val,   val_pnl   = attach_pnl(val,   SYM)"""))

new_cells.append(code("c-bt-baseline", """\
def metrics(pnl_arr: np.ndarray, w: np.ndarray, f: np.ndarray, n_days: int):
    valid = np.isfinite(pnl_arr)
    pa = pnl_arr[valid]; ww = w[valid]; ff = f[valid]
    pnl_all = (ww * pa).sum() / ww.sum()
    keep_w  = ww * (1 - ff)
    filt_w  = ww * ff
    pnl_kept     = (keep_w * pa).sum() / max(keep_w.sum(), 1e-9)
    pnl_filtered = (filt_w * pa).sum() / max(filt_w.sum(), 1e-9)
    turnover_day = keep_w.sum() / n_days   # NB: this is over the SAMPLED slice; need scale-up
    return dict(pnl_all=pnl_all, pnl_kept=pnl_kept, pnl_filtered=pnl_filtered,
                score=pnl_kept - pnl_all, kept_turnover_per_day=turnover_day,
                kept_frac=(1-ff).mean(), n=len(pa))

# Scale: sample is ~1/k of full table.  Turnover constraint is on FULL data → kept_turnover_per_day * k.
TRAIN_N_TOTAL = n_rows("binance","trades",SYM) * (
    (TRAIN_HI - TRAIN_LO) / (ts_range("binance","trades",SYM)[1] - ts_range("binance","trades",SYM)[0])
)
SCALE_TRAIN = TRAIN_N_TOTAL / len(train)
print(f"sample scale factor on train ≈ {SCALE_TRAIN:.0f}x")

print("\\n=== baseline (no filter) ===")
print(f"{'split':6}  {'τ':>3}s   {'PnL_all':>10}   {'turnover/d':>14}   valid")
for split_name, sample, pnl_dict, days in [
    ("train", train, train_pnl, TRAIN_DAYS),
    ("val",   val,   val_pnl,   VAL_DAYS),
]:
    w = sample["weight"].to_numpy()
    f_keep_all = np.zeros(len(sample))
    scale = (n_rows("binance","trades",SYM) * (TRAIN_HI-TRAIN_LO)/(ts_range("binance","trades",SYM)[1]-ts_range("binance","trades",SYM)[0]) / len(sample)) if split_name=="train" else (n_rows("binance","trades",SYM) * (VAL_HI-VAL_LO)/(ts_range("binance","trades",SYM)[1]-ts_range("binance","trades",SYM)[0]) / len(sample))
    for tau_s in TAUS_S:
        m = metrics(pnl_dict[tau_s], w, f_keep_all, days)
        print(f"{split_name:6}  {tau_s:>3}s   {m['pnl_all']:+9.3f}bps   ${m['kept_turnover_per_day']*scale:>10,.0f}/d  {m['n']:>7,}")"""))

# === Strategy 1 ===
new_cells.append(md("s15-s1", """### 15.1 Strategy 1 — Bybit liquidation momentum

For each trade $t$, compute signed Bybit liquidation pressure over $(t - 30\\text{s},\\ t - 200\\text{ms}]$:
$$\\text{pressure}_i = \\sum_{j:\\ t_i-30s < t_j+200\\text{ms} \\leq t_i} \\sigma_j \\cdot \\text{notional}_j$$
where $\\sigma_j = +1$ for liq buy, $-1$ for liq sell. The +200 ms shift is the mandated information delay.

Filter rule: if taker side matches the sign of pressure (both pushing same way), filter the trade (we'd be on the wrong maker side). Threshold sweep on absolute pressure."""))

new_cells.append(code("c-s1", """\
def liq_pressure(trade_ts: np.ndarray, liq: pl.DataFrame,
                 lookback_us: int, gate_us: int, by_shift_us: int = 0) -> np.ndarray:
    \"\"\"Signed sum of notional for liqs visible to us in (t-lookback, t-gate], with optional ts shift.\"\"\"
    liq = liq.with_columns([
        pl.when(pl.col("side") == "buy").then(pl.col("price") * pl.col("amount"))
                                       .otherwise(-pl.col("price") * pl.col("amount")).alias("sn"),
        (pl.col("timestamp") + by_shift_us).alias("ts_visible"),
    ]).sort("ts_visible")
    ts_v = liq["ts_visible"].to_numpy()
    sn   = liq["sn"].to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(sn)])
    # liqs with ts_visible in (trade_ts - lookback, trade_ts - gate]
    idx_hi = np.searchsorted(ts_v, trade_ts - gate_us,      side="right")
    idx_lo = np.searchsorted(ts_v, trade_ts - lookback_us,  side="right")
    return csum[idx_hi] - csum[idx_lo]

LOOKBACK_S1_US = 30_000_000      # 30 s
GATE_S1_US     = 200_000         # 200 ms (mandated Bybit delay)

by_liq_btc = load_small("bybit", "liquidations", SYM).sort("timestamp")
print(f"bybit liqs {SYM}: {len(by_liq_btc):,}")

def s1_features(sample: pl.DataFrame) -> np.ndarray:
    ts = sample["timestamp"].to_numpy()
    return liq_pressure(ts, by_liq_btc, LOOKBACK_S1_US, GATE_S1_US, by_shift_us=200_000)

train_press = s1_features(train)
val_press   = s1_features(val)
print(f"pressure stats train: nonzero {(train_press!=0).mean()*100:.1f}%, |p90| = ${np.percentile(np.abs(train_press),90):,.0f}")"""))

new_cells.append(code("c-s1-sweep", """\
def s1_filter(sample, pressure, threshold_usd):
    sgn = np.where(sample["side"].to_numpy() == "buy", +1.0, -1.0)
    # filter when taker side = sign of pressure AND |pressure| >= threshold
    same_side = np.sign(pressure) == sgn
    return (same_side & (np.abs(pressure) >= threshold_usd)).astype(float)

THRESHOLDS = [0, 1e3, 1e4, 5e4, 1e5, 5e5, 1e6]
def sweep(sample, pnl_dict, pressure, days, scale):
    rows = []
    for thr in THRESHOLDS:
        f = s1_filter(sample, pressure, thr)
        for tau_s in TAUS_S:
            w = sample["weight"].to_numpy()
            m = metrics(pnl_dict[tau_s], w, f, days)
            rows.append({"thr_usd": thr, "tau_s": tau_s,
                         "score_bps": round(m["score"], 4),
                         "pnl_kept": round(m["pnl_kept"], 4),
                         "pnl_filt": round(m["pnl_filtered"], 4),
                         "kept_frac": round(m["kept_frac"], 3),
                         "kept_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale, 0)})
    return pl.DataFrame(rows)

scale_train = SCALE_TRAIN
scale_val   = (n_rows("binance","trades",SYM)
               * (VAL_HI - VAL_LO) / (ts_range("binance","trades",SYM)[1] - ts_range("binance","trades",SYM)[0])) / len(val)

print("=== Strategy 1 / TRAIN ===")
print(sweep(train, train_pnl, train_press, TRAIN_DAYS, scale_train))
print("\\n=== Strategy 1 / VAL ===")
print(sweep(val, val_pnl, val_press, VAL_DAYS, scale_val))"""))

# === Strategy 2 ===
new_cells.append(md("s15-s2", """### 15.2 Strategy 2 — Binance liquidation reversal

Same pressure idea, but with **Binance** liquidations and **opposite** sign hypothesis (reversal after the liq).

For each trade $t$, compute signed Binance liq pressure over $(t - 60\\text{s},\\ t]$ (no information delay — these are local). Hypothesis: a recent buy-liq cluster means a long-squeeze just exhausted, expect mid to drop, so we filter taker-sells (which would lose) and keep taker-buys (which would win). Effectively the rule mirrors S1 with sign flipped."""))

new_cells.append(code("c-s2", """\
LOOKBACK_S2_US = 60_000_000   # 60 s
bn_liq_btc = load_small("binance", "liquidations", SYM).sort("timestamp")
print(f"binance liqs {SYM}: {len(bn_liq_btc):,}")

def s2_features(sample: pl.DataFrame) -> np.ndarray:
    ts = sample["timestamp"].to_numpy()
    return liq_pressure(ts, bn_liq_btc, LOOKBACK_S2_US, gate_us=0, by_shift_us=0)

train_press2 = s2_features(train)
val_press2   = s2_features(val)

def s2_filter(sample, pressure, threshold_usd):
    # reversal hypothesis: filter when taker side = OPPOSITE of pressure (i.e. taker continuing in liq direction → bad)
    # equivalently: keep taker-buys after buy-liq (mid expected to mean-revert down → maker-seller loses → wait that's bad...)
    # carefully:
    #   pressure>0 (recent buy-liqs) → hypothesis: mid will drop next minute
    #     taker buy at ask → maker sold → mid drops → maker wins → KEEP
    #     taker sell at bid → maker bought → mid drops → maker loses → FILTER
    #   pressure<0 → symmetric: filter taker-buys
    sgn = np.where(sample["side"].to_numpy() == "buy", +1.0, -1.0)
    # FILTER if sign(pressure) == -sgn (taker direction OPPOSITE pressure, i.e. SAME as mean-revert direction)
    # i.e. taker_sells filtered when buy-pressure (pressure>0 and sgn=-1 → -sgn=+1 == sign(pressure))
    filter_when = np.sign(pressure) == -sgn
    return (filter_when & (np.abs(pressure) >= threshold_usd)).astype(float)

# convenience: redefine sweep with arbitrary filter_fn
def sweep_with(filter_fn, sample, pnl_dict, pressure, days, scale):
    rows = []
    w = sample["weight"].to_numpy()
    for thr in THRESHOLDS:
        f = filter_fn(sample, pressure, thr)
        for tau_s in TAUS_S:
            m = metrics(pnl_dict[tau_s], w, f, days)
            rows.append({"thr_usd": thr, "tau_s": tau_s,
                         "score_bps": round(m["score"], 4),
                         "pnl_kept": round(m["pnl_kept"], 4),
                         "pnl_filt": round(m["pnl_filtered"], 4),
                         "kept_frac": round(m["kept_frac"], 3),
                         "kept_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale, 0)})
    return pl.DataFrame(rows)

print("=== Strategy 2 / TRAIN ===")
print(sweep_with(s2_filter, train, train_pnl, train_press2, TRAIN_DAYS, scale_train))
print("\\n=== Strategy 2 / VAL ===")
print(sweep_with(s2_filter, val,   val_pnl,   val_press2,   VAL_DAYS,   scale_val))"""))

# === Strategy 3 ===
new_cells.append(md("s15-s3", """### 15.3 Strategy 3 — Trade size filter

Hypothesis: trades with large notional are informed; mid moves in their direction afterwards → maker on the other side loses. Filter trades with `notional > threshold`."""))

new_cells.append(code("c-s3", """\
def s3_filter(sample, _pressure_unused, threshold_usd):
    return (sample["notional"].to_numpy() > threshold_usd).astype(float)

THRESHOLDS_S3 = [1e3, 1e4, 5e4, 1e5, 2e5, 5e5]

def sweep_s3(sample, pnl_dict, days, scale):
    rows = []
    w = sample["weight"].to_numpy()
    for thr in THRESHOLDS_S3:
        f = s3_filter(sample, None, thr)
        for tau_s in TAUS_S:
            m = metrics(pnl_dict[tau_s], w, f, days)
            rows.append({"thr_usd": thr, "tau_s": tau_s,
                         "score_bps": round(m["score"], 4),
                         "pnl_kept": round(m["pnl_kept"], 4),
                         "pnl_filt": round(m["pnl_filtered"], 4),
                         "kept_frac": round(m["kept_frac"], 3),
                         "kept_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale, 0)})
    return pl.DataFrame(rows)

print("=== Strategy 3 / TRAIN ===")
print(sweep_s3(train, train_pnl, TRAIN_DAYS, scale_train))
print("\\n=== Strategy 3 / VAL ===")
print(sweep_s3(val, val_pnl, VAL_DAYS, scale_val))"""))

# === Summary ===
new_cells.append(md("s15-sum", """### 15.4 Notes / next steps

What to read off the sweep tables above:
- **Score(τ) > 0** = filter beats baseline at that horizon. A consistent column of positive scores across τ is a stronger signal than a one-off.
- **kept_turn_per_day_USD ≥ \\$500{,}000** required — rows below that line are infeasible.
- Mismatch train vs val: if Score collapses on val, the threshold was overfit.

Next-week extensions (out of scope for this notebook):
- Add Strategy 4 (cluster size) once we accept ~80 % of BTC trades share µs — needs streaming aggregation upstream.
- Combine S1+S2+S3 features into a gradient-boosted classifier with weighted samples; predict per τ.
- Add cross-symbol features (BTC liqs → ETH trades and vice versa).
- Tighten the Bybit window: try shorter lookbacks (1s, 5s) — the 30 s window blurs short-term momentum."""))

# Append all new cells to the END (findings log was dropped by Colab on download).
# Also re-add the findings log at the very end.
findings = md("s16", """---
## 16. Findings log

**Date range:** 2025-11-01 00:00 → 2026-04-28 23:59 UTC (179 days, full 6 months).

**Volumes:**
- binance trades BTC = 804 M rows (4.5 M/day) · ETH = 1.37 B rows (7.7 M/day)
- binance bbo    BTC = 203 M · ETH = 220 M (≈1.2 M/day each)
- binance liq    BTC = 236 K (1.3 K/day) · ETH = 271 K (1.5 K/day)
- bybit   liq    BTC = 438 K (2.4 K/day) · ETH = 302 K (1.7 K/day) — **Bybit ≈ 2× Binance on BTC**

**Timestamp unit:** µs UTC ✓ (magnitude 1.76e15, decodes to plausible dates).

**`side` semantics:**
- trades: **taker side ✓** (in a 1-hour BTC window, 99.3 % of `side=buy` trades sit above mid; 99.3 % of `side=sell` below mid)
- liquidations: liquidation-order side; in bear regime, sell-liqs > buy-liqs ✓ (e.g. Binance BTC: 131 k sell vs 105 k buy; Bybit BTC: 280 k sell vs 158 k buy)

**Duplicate timestamps (trades):**
- BTC: 80 % of rows share their µs with another row (uniq ratio 0.21)
- ETH: 84 % (uniq ratio 0.16)
- BBO and Binance liqs: 100 % unique. Bybit liqs: ~98 % unique
- Interpretation: one taker event splits into N maker-fills inside a single µs. For markout PnL, all rows in the same cluster see identical mid(t+τ).

**Trade size distribution:**
- BTC: median \\$232, p99 \\$49 k, p99.9 \\$146 k, p99.99 \\$500 k — clip at \\$100 k catches **0.23 %** of trades
- ETH: median \\$41, p99 \\$27 k, p99.9 \\$86 k, p99.99 \\$300 k — clip catches **0.07 %**
- Side balance ≈ 50/50 (as expected for taker side)
- 6-month gross taker volume: BTC \\$2.4 T, ETH \\$2.17 T

**BBO spread:**
- BTC: median **0.013 bps** (p99 0.016) — essentially always 1 tick
- ETH: median **0.037 bps** (p99 0.055) — wider in bps because tick is bigger fraction of price
- 0 crossed / 0 locked across the full window (no obvious data corruption)
- median bid/ask size: BTC 4 / 4 BTC; ETH 62.9 / 62.3 ETH

**Liquidation magnitudes:**
- Biggest single liq seen: Binance BTC \\$12.6 M, Binance ETH \\$12.1 M; Bybit max ≈ \\$3.5 M (Binance liqs more lumpy)

**BBO around a Binance liquidation (median signed move):**
- t ± 100 ms: 0 bps (effect already in price at the moment of recorded liq)
- t + 1 s: 0 bps
- t + 10 s: ±0.04 bps
- **t + 60 s: −0.84 bps for buy-liqs, +0.46 bps for sell-liqs** — i.e. mean-reversion *opposite* to the liq direction over the minute that follows. p25/p75 widen to ±5 bps (heavy-tailed individual events).

**Bybit → Binance signed response curve:**
- Monotonically rising through +2 s (the edge of our offset window). Peak not yet found at +2 s. The claimed 200 ms is the information-availability delay we must respect, not the impulse-response peak.

**Cross-exchange liquidation alignment:**
- BTC: 43 % of Binance liqs have a same-side Bybit liq within ±2 s. **Median Δt = −56 ms** (Bybit fires first).
- ETH: 38 % matched, median Δt = −96 ms.
- After the +200 ms info delay, Bybit liq becomes visible ≈ 144–104 ms *after* the matching Binance liq. So Bybit cannot anticipate concurrent Binance liqs, but is a useful signal for the seconds-long directional pressure that follows.

**Anything weird:**
- The huge taker-trade duplicate-µs ratio (80 %) — single events become many maker-fill rows.
- Bybit reports more BTC liquidations than Binance over the same period — surprising given Bybit's smaller market share for spot/perp BTC. Possible explanations: different way of recording partial fills, higher Bybit leverage caps, or just more retail concentration on Bybit.
""")

nb["cells"] = nb["cells"] + new_cells + [findings]

json.dump(nb, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"appended {len(new_cells)} cells; total now {len(nb['cells'])}")
