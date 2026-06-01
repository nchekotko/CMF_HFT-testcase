"""
Week-1 ML baseline — weighted gradient boosting that filters Binance maker trades.

Instead of a fixed-sign liquidation rule (which we showed flips direction between Dec and
Feb — see eval_signals.py), we let a gradient-boosted model learn a regime-conditional,
non-linear relationship. Per horizon tau we predict the maker markout PnL of each trade
(sample_weight = w_i = min(notional, 100k)); the filter then drops the trades with the
lowest predicted PnL. The kept fraction per tau is chosen by forward-chaining time-series
CV inside the train window, so it is not tuned on validation.

Features (all computable from the four input frames, no look-ahead):
  side_sign, log_notional, log_cluster (sweeper size),
  spread_bps, hour, vol_60s (realized vol),
  mom_5s / mom_60s aligned to the maker side (recent mid drift = regime proxy),
  Bybit & Binance liq pressure, both aligned-to-side and magnitude, at a few lookbacks.

Artifacts (models + kept fractions) are saved with joblib so classify_trades runs on the
hidden test unchanged.

Usage:
    python ml_baseline.py train     # build train set, CV, fit, save artifacts, eval on val
    python ml_baseline.py eval      # load artifacts, eval on val only
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

import baseline as B

TAUS_S = B.TAUS_S
DAY_US = B.DAY_US
CLIP = B.CLIP
_HERE = Path(__file__).resolve().parent


def _artifact_path(sym: str) -> Path:
    return _HERE / f"ml_artifacts_{sym}.joblib"


def _symbol_of(trades) -> str:
    """Infer BTC/ETH from the trades ticker (e.g. 'perp:ethusdt' -> 'ETH')."""
    tk = trades["ticker"][0] if "ticker" in trades.columns and len(trades) else ""
    return "ETH" if "eth" in tk.lower() else "BTC"

FEATURES = [
    "side_sign", "log_notional", "log_cluster",
    "spread_bps", "hour", "vol_60s",
    "mom_5s_al", "mom_60s_al",
    "pby_al", "pby_abs", "pbn_al", "pbn_abs", "pbn10_al",
]

# Optional trade-volume-imbalance features (experiment). Signed taker flow over a few
# lookbacks, aligned to the maker side + volume-normalised. Causal: only fully-elapsed
# seconds strictly before the trade's second (no same-instant leakage).
VI_LOOKBACKS_US = [5_000_000, 30_000_000, 120_000_000]
VI_FEATURES = [f"{p}{w // 1_000_000}s_al" for w in VI_LOOKBACKS_US for p in ("vi", "vinorm")]

# Per-horizon feature set: VI (order-flow imbalance) is a ~10-30s signal — it lifts Score at
# tau=30 (+0.08 OOS) but adds noise at tau=300 (-0.15). So use VI only at the short horizon.
TAU_VOLIMB = {30: True, 120: False, 300: False}


def features_for(volimb: bool) -> list:
    return FEATURES + VI_FEATURES if volimb else list(FEATURES)


def build_flow_grid(t: np.ndarray, side_sign: np.ndarray, notional: np.ndarray):
    """1-second signed/total taker-flow grid (cumsum) from FULL-day arrays."""
    sec = t // 1_000_000
    edges, inv = np.unique(sec, return_inverse=True)
    signed = np.bincount(inv, weights=side_sign * notional, minlength=len(edges))
    total = np.bincount(inv, weights=notional, minlength=len(edges))
    flow_ts = edges * 1_000_000
    return flow_ts, np.concatenate([[0.0], np.cumsum(signed)]), np.concatenate([[0.0], np.cumsum(total)])
RNG = np.random.default_rng(0)
KEEP_PER_DAY = 60_000          # rows kept per day in the training matrix
LABEL_WINSOR = 50.0            # clip PnL labels to +/- this (bps) for fit stability
# Includes 1.0 ("keep everything"): when a horizon has no edge, CV picks no filtering
# (Score floor = 0) instead of a filter that hurts.
KEEP_FRAC_GRID = np.append(np.round(np.arange(0.40, 0.991, 0.05), 3), 1.0)


# ----------------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------------
def _asof_backward(ref_ts: np.ndarray, ts: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Last `vals` observed at or before each `ref_ts` (NaN if none)."""
    idx = np.searchsorted(ts, ref_ts, side="right") - 1
    out = np.full(len(ref_ts), np.nan)
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def _prep_liq(liq, shift_us: int):
    """Return (ts_visible_sorted, signed_notional_cumsum) for fast windowed pressure queries."""
    if liq.is_empty():
        return np.array([0], dtype=np.int64), np.array([0.0])
    liq = liq.with_columns(
        pl.when(pl.col("side") == "buy")
        .then(pl.col("price") * pl.col("amount"))
        .otherwise(-pl.col("price") * pl.col("amount")).alias("_sn"),
        (pl.col("timestamp") + shift_us).alias("_tv"),
    ).sort("_tv")
    ts_v = liq["_tv"].to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(liq["_sn"].to_numpy())])
    return ts_v, csum


def _pressure(ts_v, csum, t, lookback_us, gate_us) -> np.ndarray:
    hi = np.searchsorted(ts_v, t - gate_us, side="right")
    lo = np.searchsorted(ts_v, t - lookback_us, side="right")
    return csum[hi] - csum[lo]


class FeatureContext:
    """Precomputes BBO/liq/cluster for one day once; builds the feature matrix for any row
    subset. Lets training sample rows cheaply and lets eval/inference chunk huge days."""

    def __init__(self, trades, bbo, liq_bn, liq_by, precomputed_cluster=None,
                 volimb=False, flow_grid=None):
        self.t_all = trades["timestamp"].to_numpy()
        self.side_all = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0)
        self.notional_all = (trades["price"] * trades["amount"]).to_numpy()
        self.weight_all = np.minimum(self.notional_all, CLIP)
        self.cluster_all = (precomputed_cluster.astype(np.float64)
                            if precomputed_cluster is not None
                            else B._cluster_size(trades).astype(np.float64))
        self.n = len(self.t_all)
        self.volimb = volimb
        self._names = features_for(volimb)
        if volimb:
            # In eval the frame is a subsample -> the caller must pass a FULL-day flow grid.
            self.flow_ts, self.flow_signed_cs, self.flow_tot_cs = (
                flow_grid if flow_grid is not None
                else build_flow_grid(self.t_all, self.side_all, self.notional_all))

        bbo = bbo.sort("timestamp")
        self.b_ts = bbo["timestamp"].to_numpy()
        b_mid = (bbo["bid_price"].to_numpy() + bbo["ask_price"].to_numpy()) * 0.5
        self.b_mid = b_mid
        self.b_spread = bbo["ask_price"].to_numpy() - bbo["bid_price"].to_numpy()
        grid = np.arange(self.b_ts[0], self.b_ts[-1] + 1, 1_000_000)
        g_mid = _asof_backward(grid, self.b_ts, b_mid)
        logret = np.diff(np.log(g_mid))
        self.vol_ts = grid[1:]
        self.vol = pd.Series(logret).rolling(60, min_periods=10).std().to_numpy() * 1e4

        liq_bn = B._restrict_to_trade_symbols(trades, liq_bn, strip_perp=True)
        liq_by = B._restrict_to_trade_symbols(trades, liq_by, strip_perp=True)
        self.by_tv, self.by_cs = _prep_liq(liq_by, B.BYBIT_VISIBILITY_US)
        self.bn_tv, self.bn_cs = _prep_liq(liq_bn, 0)

    def build(self, idx) -> np.ndarray:
        """Feature matrix (float32) for rows `idx` (slice or index array)."""
        t = self.t_all[idx]
        ss = self.side_all[idx]
        m = len(t)
        mid_now = _asof_backward(t, self.b_ts, self.b_mid)
        spread_now = _asof_backward(t, self.b_ts, self.b_spread)
        mid_5s = _asof_backward(t - 5_000_000, self.b_ts, self.b_mid)
        mid_60s = _asof_backward(t - 60_000_000, self.b_ts, self.b_mid)
        mom_5s = (mid_now / mid_5s - 1.0) * 1e4
        mom_60s = (mid_now / mid_60s - 1.0) * 1e4

        pby = _pressure(self.by_tv, self.by_cs, t, B.BYBIT_LOOKBACK_US, B.BYBIT_VISIBILITY_US)
        pbn = _pressure(self.bn_tv, self.bn_cs, t, B.BINANCE_LOOKBACK_US, 0)
        pbn10 = _pressure(self.bn_tv, self.bn_cs, t, 10_000_000, 0)

        X = np.empty((m, len(self._names)), dtype=np.float32)
        X[:, 0] = ss
        X[:, 1] = np.log1p(self.notional_all[idx])
        X[:, 2] = np.log1p(self.cluster_all[idx])
        X[:, 3] = spread_now / mid_now * 1e4
        X[:, 4] = (t // 3_600_000_000) % 24
        X[:, 5] = _asof_backward(t, self.vol_ts, self.vol)
        X[:, 6] = ss * mom_5s
        X[:, 7] = ss * mom_60s
        X[:, 8] = ss * np.sign(pby) * np.log1p(np.abs(pby))
        X[:, 9] = np.log1p(np.abs(pby))
        X[:, 10] = ss * np.sign(pbn) * np.log1p(np.abs(pbn))
        X[:, 11] = np.log1p(np.abs(pbn))
        X[:, 12] = ss * np.sign(pbn10) * np.log1p(np.abs(pbn10))
        if self.volimb:
            t_sec = (t // 1_000_000) * 1_000_000   # only fully-elapsed seconds -> causal
            hi = np.searchsorted(self.flow_ts, t_sec, side="left")
            col = 13
            for w in VI_LOOKBACKS_US:
                lo = np.searchsorted(self.flow_ts, t_sec - w, side="left")
                signed = self.flow_signed_cs[hi] - self.flow_signed_cs[lo]
                total = self.flow_tot_cs[hi] - self.flow_tot_cs[lo]
                X[:, col] = ss * np.sign(signed) * np.log1p(np.abs(signed))   # raw, aligned
                X[:, col + 1] = ss * signed / (total + 1.0)                   # normalised, aligned
                col += 2
        return X

    def predict_chunked(self, models: dict, volimb_map=None, chunk: int = 500_000) -> dict:
        """Per-tau predicted PnL for all rows, built in row-chunks to bound memory.

        `volimb_map[tau]` selects the feature set per tau: True -> base+VI (all built columns),
        False -> base only (first len(FEATURES) columns). Build the context with volimb=True so
        the VI columns exist for the tau that needs them.
        """
        nbase = len(FEATURES)
        preds = {tau: np.empty(self.n, dtype=np.float32) for tau in models}
        for s in range(0, self.n, chunk):
            e = min(s + chunk, self.n)
            Xc = self.build(np.arange(s, e))
            for tau, mdl in models.items():
                use = Xc if (volimb_map and volimb_map.get(tau)) else Xc[:, :nbase]
                preds[tau][s:e] = mdl.predict(use)
        return preds


# ----------------------------------------------------------------------------------
# Dataset assembly (subsampled rows from full days)
# ----------------------------------------------------------------------------------
def _traincache_dir(sym: str) -> Path:
    return _HERE / f"_traincache_{sym}"


def build_dataset(lo: int, hi: int, sym: str, stride_days: int, keep_per_day: int,
                  taus=TAUS_S, volimb=False, resume: bool = True) -> dict:
    """Loop full days; compute features+labels; randomly subsample rows to bound memory.

    `taus` is the list of markout horizons (seconds) to label; defaults to the submission set.
    `volimb` appends trade-volume-imbalance features (uses the full-day flow grid).

    Each day's sampled (X, w, t, y) is checkpointed to a small .npz on disk. This box has
    <0.5 GB free RAM, so a transient spike from other apps can abort the process mid-build; on
    re-run (`resume=True`, the default) days already cached are skipped, so repeated runs make
    forward progress and eventually complete. Each restart also frees all prior-day memory.
    """
    pad_us = max(taus) * 1_000_000 + 5_000_000
    ckpt = _traincache_dir(sym)
    ckpt.mkdir(exist_ok=True)

    # --- Phase 1: build any not-yet-cached day (peak RAM = one day at a time). ---
    day, day_idx = lo, 0
    while day < hi:
        d0 = day
        day += stride_days * DAY_US
        fp = ckpt / f"day_{day_idx:03d}.npz"
        day_idx += 1
        if resume and fp.exists():
            print(f"  day {day_idx:>2}: cached ({fp.name})")
            continue
        trades, bbo, liq_bn, liq_by = B.load_day(sym, d0, d0 + DAY_US, pad_us)
        if trades.is_empty() or bbo.is_empty():
            np.savez(fp, empty=np.array(True))
            print(f"  day {day_idx:>2}: empty")
            continue
        ctx = FeatureContext(trades, bbo, liq_bn, liq_by, volimb=volimb)
        n = ctx.n
        if keep_per_day and n > keep_per_day:
            sel = np.sort(RNG.choice(n, size=keep_per_day, replace=False))
        else:
            sel = np.arange(n)
        ys_day = {tau: B.markout_pnl_bps(trades, bbo, tau * 1_000_000)[sel] for tau in taus}
        np.savez(
            fp, X=ctx.build(sel), w=ctx.weight_all[sel], t=ctx.t_all[sel],
            **{f"y_{tau}": ys_day[tau] for tau in taus},
        )
        print(f"  day {day_idx:>2}: {n:,} trades -> kept {len(sel):,}  (cached {fp.name})")
        del trades, bbo, liq_bn, liq_by, ctx, ys_day

    # --- Phase 2: load the small per-day checkpoints and concatenate (cheap). ---
    Xs, ws, ts = [], [], []
    ys = {tau: [] for tau in taus}
    n_days = 0
    for fp in sorted(ckpt.glob("day_*.npz")):
        d = np.load(fp)
        if "empty" in d.files:
            continue
        Xs.append(d["X"]); ws.append(d["w"]); ts.append(d["t"])
        for tau in taus:
            ys[tau].append(d[f"y_{tau}"])
        n_days += 1

    return dict(
        X=np.concatenate(Xs), w=np.concatenate(ws), t=np.concatenate(ts),
        y={tau: np.concatenate(ys[tau]) for tau in taus}, n_days=n_days,
    )


def build_dataset_streaming(lo: int, hi: int, sym: str, stride_days: int, keep_per_day: int,
                            taus=TAUS_S, volimb=False) -> dict:
    """Memory-frugal training-set assembly: stream+`gather_every`-subsample each day so the
    full multi-million-row trade frame is NEVER materialised (mirrors the eval loader). The
    full-day path in build_dataset OOMs on ETH (15-30M rows/day) on the ~7 GB box; this keeps
    peak RAM at the subsample size. The subsample is evenly spaced in time rather than random,
    which is fine for fitting a GBM and gives uniform intraday coverage."""
    pad_us = max(taus) * 1_000_000 + 5_000_000
    Xs, ws, ts = [], [], []
    ys = {tau: [] for tau in taus}
    day = lo
    n_days = 0
    while day < hi:
        d0 = day
        res = _load_eval_day(sym, day, day + DAY_US, pad_us, keep_per_day)
        day += stride_days * DAY_US
        if res is None:
            continue
        sampled, bbo, liq_bn, liq_by, n_full = res
        if bbo.is_empty():
            continue
        n_days += 1
        cluster = sampled["_cln"].fill_null(1).to_numpy().astype(np.float64)
        # VI uses the FULL-day streamed flow grid (not the subsample), like eval.
        grid = flow_grid_for_day(sym, d0, d0 + DAY_US) if volimb else None
        ctx = FeatureContext(sampled, bbo, liq_bn, liq_by, precomputed_cluster=cluster,
                             volimb=volimb, flow_grid=grid)
        Xs.append(ctx.build(np.arange(ctx.n)))
        ws.append(ctx.weight_all)
        ts.append(ctx.t_all)
        for tau in taus:
            ys[tau].append(B.markout_pnl_bps(sampled, bbo, tau * 1_000_000))
        print(f"  day {n_days:>2}: {n_full:,} trades -> sampled {len(sampled):,}")

    return dict(
        X=np.concatenate(Xs), w=np.concatenate(ws), t=np.concatenate(ts),
        y={tau: np.concatenate(ys[tau]) for tau in taus}, n_days=n_days,
    )


# ----------------------------------------------------------------------------------
# Training: per-tau model + kept-fraction chosen by time-series CV
# ----------------------------------------------------------------------------------
def _fit_model(X, y, w) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    )
    yc = np.clip(y, -LABEL_WINSOR, LABEL_WINSOR)
    m.fit(X, yc, sample_weight=w)
    return m


def _score_for_fraction(pred, y, w, keep_frac) -> tuple[float, float]:
    """Score (PnL_kept - PnL_all) and kept turnover-share if we keep the top `keep_frac` by pred."""
    cut = np.quantile(pred, 1.0 - keep_frac)
    keep = pred >= cut
    pnl_all = (w * y).sum() / w.sum()
    kw = w * keep
    if kw.sum() <= 0:
        return -np.inf, 0.0
    pnl_kept = (kw * y).sum() / kw.sum()
    return pnl_kept - pnl_all, kw.sum()


def _cv_pick_fraction(X, y, w, t, n_splits=4) -> tuple[float, list]:
    """Forward-chaining CV: mean fold Score per kept-fraction; return best fraction."""
    order = np.argsort(t)
    X, y, w = X[order], y[order], w[order]
    valid = np.isfinite(y)
    X, y, w = X[valid], y[valid], w[valid]
    n = len(y)
    bounds = np.linspace(0, n, n_splits + 2, dtype=int)  # expanding train, next block = val

    per_frac = {q: [] for q in KEEP_FRAC_GRID}
    for k in range(1, n_splits + 1):
        tr_end = bounds[k]
        va_end = bounds[k + 1]
        if va_end - tr_end < 1000 or tr_end < 1000:
            continue
        m = _fit_model(X[:tr_end], y[:tr_end], w[:tr_end])
        pv = m.predict(X[tr_end:va_end])
        yv, wv = y[tr_end:va_end], w[tr_end:va_end]
        for q in KEEP_FRAC_GRID:
            s, _ = _score_for_fraction(pv, yv, wv, q)
            per_frac[q].append(s)

    mean_score = {q: float(np.mean(v)) if v else -np.inf for q, v in per_frac.items()}
    best_q = max(mean_score, key=mean_score.get)
    table = sorted(((q, mean_score[q]) for q in KEEP_FRAC_GRID), key=lambda x: x[0])
    return best_q, table


def train(sym: str = "BTC", do_eval: bool = True) -> None:
    print(f"[{sym}] Building TRAIN dataset (Dec'25-Jan'26, stride 3 days)...")
    # Build the full feature set (base + VI); each tau then trains on its own slice.
    # Streaming subsample per day: full-day materialisation OOMs on ETH on the ~7 GB box.
    ds = build_dataset_streaming(B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1),
                                 sym, stride_days=3, keep_per_day=KEEP_PER_DAY, volimb=True)
    nbase = len(FEATURES)
    print(f"train matrix: {ds['X'].shape[0]:,} rows x {ds['X'].shape[1]} feats "
          f"(base {nbase} + VI {ds['X'].shape[1]-nbase}) from {ds['n_days']} days")

    artifacts = {"features": FEATURES, "vi_features": VI_FEATURES,
                 "models": {}, "keep_frac": {}, "volimb": {}}
    for tau in TAUS_S:
        use_vi = TAU_VOLIMB.get(tau, False)
        Xt = ds["X"] if use_vi else ds["X"][:, :nbase]
        y = ds["y"][tau]
        print(f"\n--- tau={tau}s  ({'base+VI' if use_vi else 'base'}, {Xt.shape[1]} feats) ---")
        best_q, table = _cv_pick_fraction(Xt, y, ds["w"], ds["t"])
        print("  CV mean Score by kept-fraction:")
        for q, s in table:
            mark = "  <-- best" if q == best_q else ""
            print(f"    keep {q:.2f}:  {s:+.4f}{mark}")
        valid = np.isfinite(y)
        artifacts["models"][tau] = _fit_model(Xt[valid], y[valid], ds["w"][valid])
        artifacts["keep_frac"][tau] = float(best_q)
        artifacts["volimb"][tau] = use_vi

    path = _artifact_path(sym)
    joblib.dump(artifacts, path)
    print(f"\nsaved artifacts -> {path}")
    if do_eval:
        evaluate(sym)
    else:
        print("(skipping in-process eval; run `eval` separately to keep peak RAM low)")


# ----------------------------------------------------------------------------------
# Submission entry point (loads artifacts)
# ----------------------------------------------------------------------------------
def _filter_ticker(df, base: str):
    """Keep rows whose ticker matches `base` (handles 'perp:btcusdt' and 'btcusdt')."""
    if df.is_empty() or "ticker" not in df.columns:
        return df
    return df.filter(pl.col("ticker").str.split(":").list.last() == base)


def _classify_one(trades, bbo, liq_binance, liq_bybit, sym: str) -> dict:
    path = _artifact_path(sym)
    art = joblib.load(path if path.exists() else _artifact_path("BTC"))
    vmap = art.get("volimb")
    # Build VI columns only if some horizon needs them (inference frame is the full day,
    # so FeatureContext computes the flow grid internally).
    ctx = FeatureContext(trades, bbo, liq_binance, liq_bybit, volimb=bool(vmap and any(vmap.values())))
    preds = ctx.predict_chunked(art["models"], volimb_map=vmap)
    out = {}
    for tau in TAUS_S:
        cut = np.quantile(preds[tau], 1.0 - art["keep_frac"][tau])
        out[tau] = (preds[tau] < cut).astype(np.int8)   # filter the bottom (1 - keep_frac)
    return out


def classify_trades(trades, bbo, liq_binance, liq_bybit) -> dict:
    """Return {tau_s: 0/1 filter} (1 = filter out the lowest-predicted-PnL trades).

    Robust to single- or mixed-symbol input: rows are grouped by ticker, each symbol is
    scored with its own model against that symbol's BBO/liquidations, then scattered back
    into full-length arrays preserving the original row order.
    """
    n = len(trades)
    if n == 0:
        return {tau: np.zeros(0, dtype=np.int8) for tau in TAUS_S}

    # Single-symbol fast path.
    if "ticker" not in trades.columns:
        return _classify_one(trades, bbo, liq_binance, liq_bybit, "BTC")
    bases = trades["ticker"].str.split(":").list.last()
    uniq = bases.unique().to_list()
    if len(uniq) == 1:
        return _classify_one(trades, bbo, liq_binance, liq_bybit, _symbol_of(trades))

    out = {tau: np.zeros(n, dtype=np.int8) for tau in TAUS_S}
    base_arr = bases.to_numpy()
    for base in uniq:
        sym = "ETH" if base and "eth" in str(base).lower() else "BTC"
        idx = np.nonzero(base_arr == base)[0]
        sub = _classify_one(trades[idx], _filter_ticker(bbo, base),
                            _filter_ticker(liq_binance, base),
                            _filter_ticker(liq_bybit, base), sym)
        for tau in TAUS_S:
            out[tau][idx] = sub[tau]
    return out


# ----------------------------------------------------------------------------------
# Evaluation on the validation split
# ----------------------------------------------------------------------------------
EVAL_CAP_ROWS = 1_200_000   # per-day row cap for eval (Score is a weighted mean -> unbiased).
# Kept low so a day's FeatureContext fits the ~7 GB box even when only a few hundred MB are free.


def flow_grid_for_day(sym: str, lo: int, hi: int):
    """Stream the FULL-day 1-second signed/total taker-flow grid (memory-safe) for VI features."""
    f = B.SYM_FILE[sym]
    src = B.DATA / "binance_trades" / f"{f}.parquet"
    g = (pl.scan_parquet(src)
         .filter((pl.col("timestamp") >= lo - max(VI_LOOKBACKS_US)) & (pl.col("timestamp") < hi))
         .select([
             (pl.col("timestamp") // 1_000_000).alias("_s"),
             pl.when(pl.col("side") == "buy").then(pl.col("price") * pl.col("amount"))
               .otherwise(-pl.col("price") * pl.col("amount")).alias("_sn"),
             (pl.col("price") * pl.col("amount")).alias("_tn"),
         ])
         .group_by("_s").agg([pl.col("_sn").sum(), pl.col("_tn").sum()])
         .sort("_s").collect(engine="streaming"))
    flow_ts = g["_s"].to_numpy() * 1_000_000
    signed_cs = np.concatenate([[0.0], np.cumsum(g["_sn"].to_numpy())])
    tot_cs = np.concatenate([[0.0], np.cumsum(g["_tn"].to_numpy())])
    return flow_ts, signed_cs, tot_cs


def _load_eval_day(sym: str, lo: int, hi: int, pad_us: int, cap: int):
    """Memory-frugal sampled day for eval: stream the cluster group-by and gather_every-
    subsample so the full multi-million-row frame is never materialised in Python.
    Returns (sampled_trades_with_cluster, bbo, liq_bn, liq_by, n_full) or None if empty."""
    f = B.SYM_FILE[sym]
    src = B.DATA / "binance_trades" / f"{f}.parquet"
    lf = (pl.scan_parquet(src)
          .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
          .select(["timestamp", "side", "price", "amount"]))
    n_full = lf.select(pl.len()).collect(engine="streaming").item()
    if n_full == 0:
        return None
    step = max(1, n_full // cap)
    sizes = (lf.group_by(["timestamp", "side"]).agg(pl.len().alias("_cln"))
             .collect(engine="streaming"))
    sampled = (lf.gather_every(step).collect(engine="streaming")
               .join(sizes, on=["timestamp", "side"], how="left").sort("timestamp"))
    del sizes
    bbo = B._load_window("binance_booktickers", f, lo, hi + pad_us,
                         ["timestamp", "bid_price", "ask_price"])
    liq_bn = B._load_window("binance_liquidations", f, lo - B.BINANCE_LOOKBACK_US, hi,
                            ["timestamp", "ticker", "side", "price", "amount"])
    liq_by = B._load_window("bybit_liquidations", B.BYBIT_FILE[sym],
                            lo - B.BYBIT_LOOKBACK_US, hi,
                            ["timestamp", "ticker", "side", "price", "amount"])
    return sampled, bbo, liq_bn, liq_by, n_full


_ACC_KEYS = ("wp", "w", "kwp", "kw", "kw_turn", "keep_n", "n")


def _evalcache_dir(sym: str) -> Path:
    return _HERE / f"_evalcache_{sym}"


def evaluate(sym: str = "BTC", stride_days: int = 2, resume: bool = True) -> None:
    art = joblib.load(_artifact_path(sym))
    pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    lo, hi = B._epoch_us(2026, 2, 1), B._epoch_us(2026, 3, 1)
    acc = {tau: dict(wp=0.0, w=0.0, kwp=0.0, kw=0.0, kw_turn=0.0, keep_n=0.0, n=0.0)
           for tau in TAUS_S}
    vmap = art.get("volimb")
    use_vi = bool(vmap and any(vmap.values()))
    ckpt = _evalcache_dir(sym)
    ckpt.mkdir(exist_ok=True)
    day, day_idx, n_days = lo, 0, 0

    def _fold_in(day_acc: dict) -> None:
        """Add one day's per-tau scalar stats into the running accumulator."""
        for tau in TAUS_S:
            for k in _ACC_KEYS:
                acc[tau][k] += float(day_acc[tau][k])

    while day < hi:
        d0 = day
        day += stride_days * DAY_US
        fp = ckpt / f"day_{day_idx:03d}.npz"
        day_idx += 1
        if resume and fp.exists():
            d = np.load(fp)
            if int(d["counted"]) == 1:
                n_days += 1
                _fold_in({tau: dict(zip(_ACC_KEYS, d[str(tau)])) for tau in TAUS_S})
                print(f"  day {day_idx:>2}: cached ({fp.name})")
            continue

        res = _load_eval_day(sym, d0, d0 + DAY_US, pad_us, EVAL_CAP_ROWS)
        if res is None or res[1].is_empty():
            np.savez(fp, counted=np.array(0))
            continue
        sampled, bbo, liq_bn, liq_by, n_full = res
        cluster = sampled["_cln"].fill_null(1).to_numpy().astype(np.float64)
        scale = n_full / len(sampled)   # restore full-day turnover from the subsample

        # VI uses the FULL-day flow grid (streamed), not the subsample.
        grid = flow_grid_for_day(sym, d0, d0 + DAY_US) if use_vi else None
        ctx = FeatureContext(sampled, bbo, liq_bn, liq_by, precomputed_cluster=cluster,
                             volimb=use_vi, flow_grid=grid)
        w = ctx.weight_all
        preds = ctx.predict_chunked(art["models"], volimb_map=vmap)
        day_acc = {tau: dict.fromkeys(_ACC_KEYS, 0.0) for tau in TAUS_S}
        for tau in TAUS_S:
            pnl = B.markout_pnl_bps(sampled, bbo, tau * 1_000_000)
            cut = np.quantile(preds[tau], 1.0 - art["keep_frac"][tau])
            f = (preds[tau] < cut).astype(float)
            valid = np.isfinite(pnl)
            p, ww, ff = pnl[valid], w[valid], f[valid]
            kw = ww * (1.0 - ff)
            s = day_acc[tau]
            s["wp"] = float((ww * p).sum());  s["w"] = float(ww.sum())
            s["kwp"] = float((kw * p).sum()); s["kw"] = float(kw.sum())
            s["kw_turn"] = float(kw.sum()) * scale     # scaled back to full-day turnover
            s["keep_n"] = float((1.0 - ff).sum()); s["n"] = float(len(p))
        np.savez(fp, counted=np.array(1),
                 **{str(tau): np.array([day_acc[tau][k] for k in _ACC_KEYS]) for tau in TAUS_S})
        n_days += 1
        _fold_in(day_acc)
        print(f"  day {day_idx:>2}: {n_full:,} trades -> sampled {len(sampled):,} (x{scale:.1f})  (cached {fp.name})")
        del sampled, bbo, liq_bn, liq_by, ctx, preds, grid

    print(f"\n=== VALIDATION (Feb'26, {sym}, {n_days} days) ===")
    print(f"  {'tau':>4}  {'keep':>5}  {'Score':>8}  {'PnL_all':>8}  {'PnL_kept':>9}  "
          f"{'kept%':>6}  {'turn/day(USD)':>15}  {'>=500k?':>8}")
    for tau in TAUS_S:
        s = acc[tau]
        pa = s["wp"] / max(s["w"], 1e-9)
        pk = s["kwp"] / max(s["kw"], 1e-9)
        turn = s["kw_turn"] / n_days
        ok = "OK" if turn >= B.TURNOVER_MIN else "FAIL"
        print(f"  {tau:>4}  {art['keep_frac'][tau]:>5.2f}  {pk - pa:>+8.3f}  {pa:>+8.3f}  "
              f"{pk:>+9.3f}  {s['keep_n']/max(s['n'],1e-9)*100:>5.1f}%  {turn:>15,.0f}  {ok:>8}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    sym = sys.argv[2].upper() if len(sys.argv) > 2 else "BTC"
    if cmd == "eval":
        evaluate(sym)
    else:
        # `train SYM noeval` saves the artifact and skips the in-process eval (lower peak RAM).
        do_eval = not (len(sys.argv) > 3 and sys.argv[3].lower() in ("noeval", "skipeval"))
        train(sym, do_eval=do_eval)
