"""Append improved strategies (section 17) to week1_exploration.ipynb."""
import json
from pathlib import Path

PATH = Path("week1_exploration.ipynb")
nb = json.load(open(PATH, encoding="utf-8"))

def md(id_, src):
    return {"cell_type":"markdown","id":id_,"metadata":{},
            "source":src.splitlines(keepends=True)}

def code(id_, src):
    return {"cell_type":"code","id":id_,"execution_count":None,"metadata":{},
            "outputs":[],"source":src.splitlines(keepends=True)}

new_cells = []

new_cells.append(md("s17", """## 17. Improved strategies

Based on the diagnosis of S1–S3 in section 15:

1. **S1 had the wrong sign** — Bybit liqs are a *reversal* signal at τ ≥ 30 s, not momentum. We invert the filter and shorten the lookback to 5 s so we sit close to the impact peak.
2. **S2 overfits the train window** — try multiple lookbacks (10/30/60/120 s) to find a stable one.
3. **S3 filtered the wrong tail** — large trades are *anti-informed* in this data. Invert: filter small (retail emotion) trades instead. Watch turnover constraint.
4. **S4 (new)** — cluster size: if a single µs holds many same-side maker fills, it's an aggressive sweeper; expect adverse selection.
5. **S5 (new)** — simple ensemble: filter when *any* of S1b / S2 / S4 fires."""))

# --- S1b: inverted, short window ---
new_cells.append(md("s17-1", """### 17.1 S1b — Bybit reversal, 5 s lookback

Same signed pressure feature, but lookback = 5 s (vs 30 s) and the filter triggers on the **reversal** direction: filter taker when its side is OPPOSITE to recent Bybit pressure (a buy-squeeze just exhausted → expect down → taker-sell on the way down loses for the maker)."""))

new_cells.append(code("c-s1b", """\
LOOKBACK_S1B_US = 5_000_000   # 5 s
GATE_S1B_US     = 200_000     # 200 ms mandated delay

def s1b_features(sample: pl.DataFrame) -> np.ndarray:
    ts = sample["timestamp"].to_numpy()
    return liq_pressure(ts, by_liq_btc, LOOKBACK_S1B_US, GATE_S1B_US, by_shift_us=200_000)

def s1b_filter(sample, pressure, thr_usd):
    sgn = np.where(sample["side"].to_numpy() == "buy", +1.0, -1.0)
    filter_when = np.sign(pressure) == -sgn   # reversal
    return (filter_when & (np.abs(pressure) >= thr_usd)).astype(float)

train_p1b = s1b_features(train); val_p1b = s1b_features(val)
print(f"S1b pressure 5s: train nonzero {(train_p1b!=0).mean()*100:.1f}%, |p90| ${np.percentile(np.abs(train_p1b),90):,.0f}")

print("\\n=== S1b / TRAIN ===")
print(sweep_with(s1b_filter, train, train_pnl, train_p1b, TRAIN_DAYS, scale_train))
print("\\n=== S1b / VAL ===")
print(sweep_with(s1b_filter, val, val_pnl, val_p1b, VAL_DAYS, scale_val))"""))

# --- S2b: multi-window ---
new_cells.append(md("s17-2", """### 17.2 S2b — Binance reversal, sweep lookback

Same reversal logic as S2 but try lookback ∈ {10, 30, 60, 120} s. Pick whichever is most stable train→val."""))

new_cells.append(code("c-s2b", """\
LOOKBACKS_S2B = [10_000_000, 30_000_000, 60_000_000, 120_000_000]

def s2_filter_generic(sample, pressure, thr_usd):
    sgn = np.where(sample["side"].to_numpy() == "buy", +1.0, -1.0)
    filter_when = np.sign(pressure) == -sgn
    return (filter_when & (np.abs(pressure) >= thr_usd)).astype(float)

for lb in LOOKBACKS_S2B:
    train_p = liq_pressure(train["timestamp"].to_numpy(), bn_liq_btc, lb, gate_us=0)
    val_p   = liq_pressure(val["timestamp"].to_numpy(),   bn_liq_btc, lb, gate_us=0)
    print(f"\\n--- S2b lookback {lb//1_000_000:>3}s ---")
    # short version: best row per (split, τ) by score
    for thr in [0, 1e4, 1e5]:
        tr_f = s2_filter_generic(train, train_p, thr)
        va_f = s2_filter_generic(val,   val_p,   thr)
        for tau_s in TAUS_S:
            w_tr = train["weight"].to_numpy(); w_va = val["weight"].to_numpy()
            m_tr = metrics(train_pnl[tau_s], w_tr, tr_f, TRAIN_DAYS)
            m_va = metrics(val_pnl[tau_s],   w_va, va_f, VAL_DAYS)
            print(f"  thr=${thr:>9,.0f}  τ={tau_s:>3}s  train_score {m_tr['score']:+.3f}  "
                  f"val_score {m_va['score']:+.3f}  kept_frac {m_tr['kept_frac']:.2f}")"""))

# --- S3b: filter SMALL trades ---
new_cells.append(md("s17-3", """### 17.3 S3b — Filter SMALL trades

Inverse of S3. Filter trades with `notional < threshold`. Sweep thresholds; check that kept turnover stays above the \\$500 k/day constraint."""))

new_cells.append(code("c-s3b", """\
def s3b_filter(sample, _ignored, threshold_usd):
    return (sample["notional"].to_numpy() < threshold_usd).astype(float)

THRESHOLDS_S3B = [10, 100, 500, 1_000, 5_000, 10_000, 50_000]

def sweep_s3b(sample, pnl_dict, days, scale):
    rows = []
    w = sample["weight"].to_numpy()
    for thr in THRESHOLDS_S3B:
        f = s3b_filter(sample, None, thr)
        for tau_s in TAUS_S:
            m = metrics(pnl_dict[tau_s], w, f, days)
            rows.append({"thr_usd": thr, "tau_s": tau_s,
                         "score_bps": round(m["score"], 4),
                         "pnl_kept": round(m["pnl_kept"], 4),
                         "pnl_filt": round(m["pnl_filtered"], 4),
                         "kept_frac": round(m["kept_frac"], 3),
                         "kept_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale, 0)})
    return pl.DataFrame(rows)

print("=== S3b / TRAIN ===")
with pl.Config(set_tbl_rows=30):
    print(sweep_s3b(train, train_pnl, TRAIN_DAYS, scale_train))
print("\\n=== S3b / VAL ===")
with pl.Config(set_tbl_rows=30):
    print(sweep_s3b(val, val_pnl, VAL_DAYS, scale_val))"""))

# --- S4: cluster size ---
new_cells.append(md("s17-4", """### 17.4 S4 — Cluster size filter

When many same-side trades stamp the same µs, it's one aggressor crossing multiple maker orders — i.e. a sweeper that has urgency. Hypothesis: sweepers are informed → maker on the other side loses. Filter trades whose `(timestamp, side)` cluster is ≥ K.

We compute cluster size by querying the full Binance trades table in windows covering each chunk of sampled trades (slower than other features but tractable)."""))

new_cells.append(code("c-s4", """\
def cluster_size(sample: pl.DataFrame, sym: str, chunk: int = 200) -> np.ndarray:
    sample = sample.sort("timestamp")
    out = np.zeros(len(sample), dtype=np.int64)
    ts = sample["timestamp"].to_numpy()
    for start in range(0, len(sample), chunk):
        end = min(start + chunk, len(sample))
        t_lo = int(ts[start])
        t_hi = int(ts[end - 1])
        full = (
            window_lf(scan("binance", "trades", sym), t_lo, t_hi)
              .group_by(["timestamp", "side"]).agg(pl.len().alias("cluster_n"))
              .collect()
        )
        sub = sample[start:end].with_row_index("idx_local")
        joined = sub.join(full, on=["timestamp", "side"], how="left")
        joined = joined.sort("idx_local")
        out[start:end] = joined["cluster_n"].fill_null(1).to_numpy()
    return out

print("computing cluster_size for train ...")
train_cl = cluster_size(train, SYM)
print(f"  cluster_size stats train: median {np.median(train_cl)}, p90 {np.percentile(train_cl,90):.0f}, p99 {np.percentile(train_cl,99):.0f}, max {train_cl.max()}")
print("computing cluster_size for val ...")
val_cl = cluster_size(val, SYM)
print(f"  cluster_size stats val:   median {np.median(val_cl)}, p90 {np.percentile(val_cl,90):.0f}, p99 {np.percentile(val_cl,99):.0f}, max {val_cl.max()}")

train = train.with_columns(pl.Series("cluster_n", train_cl))
val   = val.with_columns(pl.Series("cluster_n",   val_cl))"""))

new_cells.append(code("c-s4-sweep", """\
def s4_filter(sample, _ignored, thr):
    return (sample["cluster_n"].to_numpy() >= thr).astype(float)

THRESHOLDS_S4 = [2, 5, 10, 25, 50, 100, 500]

def sweep_s4(sample, pnl_dict, days, scale):
    rows = []
    w = sample["weight"].to_numpy()
    for thr in THRESHOLDS_S4:
        f = s4_filter(sample, None, thr)
        for tau_s in TAUS_S:
            m = metrics(pnl_dict[tau_s], w, f, days)
            rows.append({"thr_cluster": thr, "tau_s": tau_s,
                         "score_bps": round(m["score"], 4),
                         "pnl_kept": round(m["pnl_kept"], 4),
                         "pnl_filt": round(m["pnl_filtered"], 4),
                         "kept_frac": round(m["kept_frac"], 3),
                         "kept_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale, 0)})
    return pl.DataFrame(rows)

print("=== S4 cluster / TRAIN ===")
with pl.Config(set_tbl_rows=30):
    print(sweep_s4(train, train_pnl, TRAIN_DAYS, scale_train))
print("\\n=== S4 cluster / VAL ===")
with pl.Config(set_tbl_rows=30):
    print(sweep_s4(val, val_pnl, VAL_DAYS, scale_val))"""))

# --- S5: ensemble ---
new_cells.append(md("s17-5", """### 17.5 S5 — Ensemble (S1b OR S2 OR S4)

Vote: filter the trade if **any** of the three rules fires at its best threshold from train. Use threshold values chosen from train alone (no val peeking)."""))

new_cells.append(code("c-s5", """\
# Pick best-by-train-score combination
def best_thr_by_train_score(sample_train, pnl_train, feature_train,
                            filter_fn, thr_grid, tau_s):
    best_thr = None; best_score = -np.inf
    w = sample_train["weight"].to_numpy()
    for thr in thr_grid:
        f = filter_fn(sample_train, feature_train, thr)
        m = metrics(pnl_train[tau_s], w, f, TRAIN_DAYS)
        if m["score"] > best_score:
            best_score = m["score"]; best_thr = thr
    return best_thr, best_score

TAU_S_PICK = 120  # tune at τ=120s
THR_GRID_PRESSURE = [0, 1e4, 5e4, 1e5, 5e5]
THR_GRID_CLUSTER  = [5, 10, 25, 50, 100]

thr_1b, _ = best_thr_by_train_score(train, train_pnl, train_p1b, s1b_filter, THR_GRID_PRESSURE, TAU_S_PICK)
thr_2,  _ = best_thr_by_train_score(train, train_pnl, train_press2, s2_filter,  THR_GRID_PRESSURE, TAU_S_PICK)
thr_4,  _ = best_thr_by_train_score(train, train_pnl, None,        s4_filter,   THR_GRID_CLUSTER, TAU_S_PICK)
print(f"chosen thresholds (by train τ={TAU_S_PICK}s): S1b ${thr_1b:,.0f}, S2 ${thr_2:,.0f}, S4 cluster≥{thr_4}")

def s5_filter(sample, features):
    p1b, p2 = features
    f1 = s1b_filter(sample, p1b, thr_1b)
    f2 = s2_filter_generic(sample, p2, thr_2)
    f4 = s4_filter(sample, None, thr_4)
    return np.maximum.reduce([f1, f2, f4])

print("\\n=== S5 ensemble — out-of-sample (val) ===")
for tau_s in TAUS_S:
    f_tr = s5_filter(train, (train_p1b, train_press2))
    f_va = s5_filter(val,   (val_p1b,   val_press2))
    m_tr = metrics(train_pnl[tau_s], train["weight"].to_numpy(), f_tr, TRAIN_DAYS)
    m_va = metrics(val_pnl[tau_s],   val["weight"].to_numpy(),   f_va, VAL_DAYS)
    print(f"τ={tau_s:>3}s  train: score {m_tr['score']:+.3f}  kept_frac {m_tr['kept_frac']:.2f}  "
          f"turn/d ${m_tr['kept_turnover_per_day']*scale_train:>12,.0f}")
    print(f"        val:   score {m_va['score']:+.3f}  kept_frac {m_va['kept_frac']:.2f}  "
          f"turn/d ${m_va['kept_turnover_per_day']*scale_val:>12,.0f}")"""))

# --- 17.6 Summary ---
new_cells.append(md("s17-6", """### 17.6 Summary

Compare best score per strategy (at τ=120s, val) to baseline:
"""))

new_cells.append(code("c-summary", """\
def best_val_score(filter_fn, train_feat, val_feat, thr_grid, tau_s=120):
    # train threshold, then evaluate on val (out-of-sample selection)
    best_thr, _ = best_thr_by_train_score(train, train_pnl, train_feat, filter_fn, thr_grid, tau_s)
    f_va = filter_fn(val, val_feat, best_thr)
    m_va = metrics(val_pnl[tau_s], val["weight"].to_numpy(), f_va, VAL_DAYS)
    return best_thr, m_va

summary = []
for name, fn, tr_feat, va_feat, grid in [
    ("S1  bybit momentum",     s1_filter,        train_press,  val_press,  THR_GRID_PRESSURE),
    ("S1b bybit reversal 5s",  s1b_filter,       train_p1b,    val_p1b,    THR_GRID_PRESSURE),
    ("S2  binance reversal 60s", s2_filter,      train_press2, val_press2, THR_GRID_PRESSURE),
    ("S3  filter LARGE",       s3_filter,        None,         None,       [1e3, 1e4, 1e5, 5e5]),
    ("S3b filter SMALL",       s3b_filter,      None,         None,        THRESHOLDS_S3B),
    ("S4  cluster",            s4_filter,        None,         None,       THR_GRID_CLUSTER),
]:
    thr, m = best_val_score(fn, tr_feat, va_feat, grid)
    summary.append({"strategy": name, "thr_chosen_on_train": thr,
                    "val_score_bps": round(m["score"], 4),
                    "val_pnl_kept": round(m["pnl_kept"], 4),
                    "val_kept_frac": round(m["kept_frac"], 3),
                    "val_turn_per_day_USD": round(m["kept_turnover_per_day"] * scale_val, 0)})
# baseline
m0 = metrics(val_pnl[120], val["weight"].to_numpy(), np.zeros(len(val)), VAL_DAYS)
summary.insert(0, {"strategy": "baseline (no filter)", "thr_chosen_on_train": None,
                   "val_score_bps": 0.0, "val_pnl_kept": round(m0["pnl_all"], 4),
                   "val_kept_frac": 1.0,
                   "val_turn_per_day_USD": round(m0["kept_turnover_per_day"] * scale_val, 0)})

print("=== Strategy comparison @ τ=120s, VAL ===")
with pl.Config(set_tbl_rows=20, set_tbl_cols=10):
    print(pl.DataFrame(summary))"""))

# Insert before findings log (s16)
target = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "## 16. Findings log" in "".join(c["source"]):
        target = i; break
assert target is not None, "findings cell not found"
nb["cells"] = nb["cells"][:target] + new_cells + nb["cells"][target:]
# Bump findings to 18
for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "## 16. Findings log" in "".join(c["source"]):
        c["source"] = "".join(c["source"]).replace("## 16. Findings log", "## 18. Findings log").splitlines(keepends=True)
        break

json.dump(nb, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"added {len(new_cells)} cells; notebook now has {len(nb['cells'])} cells")
