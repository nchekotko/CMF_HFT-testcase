"""
HW3 — Large-liquidation reaction filter for Binance maker trades.

Builds on the Task-2 baseline (see ``baseline.py`` for the shared data loaders, the
markout-PnL definition and the metric formulas). HW3 isolates a *single, interpretable*
signal instead of the Task-2 ensemble:

    For each "large" liquidation (notional >= threshold) look at the Binance trades that
    arrive in the next ``window_seconds`` and filter the ones whose taker side lines up
    with the side the liquidation pushes the toxic maker into.

EDA (``reaction_curves``) measures, per second-bucket after a large liquidation, the
average maker markout (tau = 30 s) of in-window trades, split into "same direction as the
liquidation" vs "opposite". Those curves fix the two filter parameters:

    liq_threshold_usd  -- minimum liquidation notional to count as "large"
    window_seconds     -- how long after a liquidation the reaction lasts

Submission entry point (matches description.md exactly):

    classify_trades(trades, bbo, liq_binance, liq_bybit) -> {tau_s: 0/1 array}

is a pure function of the four input frames (no look-ahead, no labels), so it runs
unchanged on the hidden BTC+ETH test set.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import baseline as B

# Reuse the Task-constant + helper surface from the baseline so the two stay in lockstep.
TAUS_S = B.TAUS_S                       # (30, 120, 300)
CLIP = B.CLIP                           # 100_000 USD weight clip
TURNOVER_MIN = B.TURNOVER_MIN           # 500_000 USD/day constraint
BYBIT_VISIBILITY_US = B.BYBIT_VISIBILITY_US   # +200 ms cross-exchange visibility

markout_pnl_bps = B.markout_pnl_bps
load_day = B.load_day
_epoch_us = B._epoch_us
DAY_US = B.DAY_US

# ----------------------------------------------------------------------------------
# Chosen filter parameters (see the notebook EDA for how these were picked).
# Defaults are placeholders; the notebook overrides them after the EDA/sweep.
# ----------------------------------------------------------------------------------
DEFAULT_WINDOW_S = 60
# Per-base-symbol USD thresholds = the 99th-percentile liquidation notional on the train
# window (see the notebook). The 99th pctl (not a lower one) is deliberate: on validation the
# edge from lower thresholds / longer windows OVERFITS (it flips negative), and only the very
# largest liquidations carry a continuation that generalises out of sample.
DEFAULT_THRESHOLD_USD = {"btcusdt": 172_000.0, "ethusdt": 158_000.0}
# "same" -> the HW3 wording: filter trades whose taker side matches the liquidation side.
#           The EDA confirms these are the toxic trades: after a large liquidation the mid
#           CONTINUES in the liquidation direction over a few minutes, so the maker who took
#           the same-side flow is run over. "opp" (filter the opposite side) is the inverse
#           and scores negative; kept only as a configurable sanity-check option.
DEFAULT_DIRECTION = "same"


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------
def liq_notional(liq: pl.DataFrame) -> np.ndarray:
    """Per-row liquidation notional in USD (price * amount)."""
    return (liq["price"] * liq["amount"]).to_numpy()


def _base_symbols(trades: pl.DataFrame) -> set[str]:
    """Base symbols traded, e.g. {'btcusdt'} from ticker 'perp:btcusdt'."""
    if "ticker" not in trades.columns:
        return set()
    return {t.split(":")[-1] for t in trades["ticker"].unique().to_list()}


def _threshold_for(base: str, threshold_usd) -> float:
    """Resolve the USD threshold for one base symbol from a scalar or per-symbol dict."""
    if isinstance(threshold_usd, dict):
        return float(threshold_usd.get(base, min(threshold_usd.values())))
    return float(threshold_usd)


def large_liquidations(liq: pl.DataFrame, base: str, threshold_usd) -> pl.DataFrame:
    """Liquidation rows (for ``base``) whose notional >= the resolved threshold,
    with a +200 ms-visible timestamp column ``_ts_vis`` and notional ``_n``.

    ``base`` matches the Bybit ticker; Binance tickers ('perp:btcusdt') are stripped.
    The visibility shift is 0 for Binance and +200 ms for Bybit (passed by the caller via
    ``visible_shift_us``); here we only filter by symbol+notional and add ``_n``.
    """
    if liq.is_empty():
        return liq
    if "ticker" in liq.columns:
        liq = liq.filter(pl.col("ticker").str.split(":").list.last() == base)
    thr = _threshold_for(base, threshold_usd)
    return liq.with_columns((pl.col("price") * pl.col("amount")).alias("_n")).filter(
        pl.col("_n") >= thr
    )


# ----------------------------------------------------------------------------------
# EDA: reaction curves (event study)
# ----------------------------------------------------------------------------------
def reaction_curves(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,
    liq: pl.DataFrame,
    threshold_usd: float,
    window_s: int = 300,
    bucket_s: int = 10,
    tau_us: int = 30 * 1_000_000,
    visible_shift_us: int = 0,
    acc: dict | None = None,
) -> dict:
    """Accumulate, per time-bucket after each large liquidation, the summed maker markout
    (tau = 30 s by default) of in-window trades split by direction.

    Returns/updates ``acc = {'same'|'opp': {'sum','cnt'}, 'edges': bucket_left_edges_s}``.
    Pass the same ``acc`` across days to pool the event study. ``side`` matching:
    a trade is 'same' direction iff its taker side equals the liquidation side.
    ``visible_shift_us`` shifts liquidation timestamps forward (cross-exchange latency).
    """
    n_buckets = window_s // bucket_s
    window_us = window_s * 1_000_000
    bucket_us = bucket_s * 1_000_000
    if acc is None:
        acc = {
            "same": {"sum": np.zeros(n_buckets), "cnt": np.zeros(n_buckets)},
            "opp": {"sum": np.zeros(n_buckets), "cnt": np.zeros(n_buckets)},
            "edges": np.arange(n_buckets) * bucket_s,
        }
    if liq.is_empty() or trades.is_empty():
        return acc

    t_tr = trades["timestamp"].to_numpy()
    buy_tr = trades["side"].to_numpy() == "buy"
    pnl = markout_pnl_bps(trades, bbo, tau_us)

    l_ts = liq["timestamp"].to_numpy() + visible_shift_us
    l_buy = liq["side"].to_numpy() == "buy"
    order = np.argsort(l_ts)
    l_ts, l_buy = l_ts[order], l_buy[order]

    for tliq, lbuy in zip(l_ts, l_buy):
        lo = np.searchsorted(t_tr, tliq, side="left")
        hi = np.searchsorted(t_tr, tliq + window_us, side="right")
        if hi <= lo:
            continue
        b = np.minimum(((t_tr[lo:hi] - tliq) // bucket_us).astype(np.int64), n_buckets - 1)
        same = buy_tr[lo:hi] == lbuy
        pn = pnl[lo:hi]
        ok = np.isfinite(pn)
        for mask, key in ((same, "same"), (~same, "opp")):
            m = mask & ok
            if m.any():
                np.add.at(acc[key]["sum"], b[m], pn[m])
                np.add.at(acc[key]["cnt"], b[m], 1.0)
    return acc


def curve_means(acc: dict, key: str) -> np.ndarray:
    """Bucket means (NaN where empty) from a reaction-curve accumulator."""
    s, c = acc[key]["sum"], acc[key]["cnt"]
    return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


# ----------------------------------------------------------------------------------
# The filter (submission entry point)
# ----------------------------------------------------------------------------------
def _mark_direction(
    f: np.ndarray,
    t_tr: np.ndarray,
    buy_tr: np.ndarray,
    liq_large: pl.DataFrame,
    window_us: int,
    direction: str,
    visible_shift_us: int,
) -> None:
    """Set f[i]=1 for trades within ``window_us`` after a large liquidation whose taker
    side matches (direction='same') / opposes (direction='opp') the liquidation side.

    Vectorised: for each liquidation side, a trade is in-window iff at least one such
    liquidation falls in (t_i - window, t_i]  (using +visible-shifted liq timestamps).
    """
    if liq_large.is_empty():
        return
    for liq_side_buy in (True, False):
        sub = liq_large.filter(pl.col("side") == ("buy" if liq_side_buy else "sell"))
        if sub.is_empty():
            continue
        lt = np.sort(sub["timestamp"].to_numpy() + visible_shift_us)
        n_hi = np.searchsorted(lt, t_tr, side="right")
        n_lo = np.searchsorted(lt, t_tr - window_us, side="right")
        in_window = (n_hi - n_lo) > 0
        # Which taker side do we flag? same -> equals liq side; opp -> opposite.
        flag_buy = liq_side_buy if direction == "same" else (not liq_side_buy)
        f[in_window & (buy_tr == flag_buy)] = 1


def build_filter(
    trades: pl.DataFrame,
    liq_binance: pl.DataFrame,
    liq_bybit: pl.DataFrame,
    threshold_usd=DEFAULT_THRESHOLD_USD,
    window_seconds: int = DEFAULT_WINDOW_S,
    direction: str = DEFAULT_DIRECTION,
    use_bybit: bool = False,
) -> np.ndarray:
    """Return the single 0/1 filter array (1 = filter the trade out).

    The filter is horizon-independent; ``classify_trades`` replicates it across taus.
    """
    n = len(trades)
    f = np.zeros(n, dtype=np.int8)
    if n == 0:
        return f
    window_us = window_seconds * 1_000_000
    t_tr = trades["timestamp"].to_numpy()
    buy_tr = trades["side"].to_numpy() == "buy"

    for base in _base_symbols(trades):
        liq_bn = large_liquidations(liq_binance, base, threshold_usd)
        _mark_direction(f, t_tr, buy_tr, liq_bn, window_us, direction, visible_shift_us=0)
        if use_bybit and liq_bybit is not None and not liq_bybit.is_empty():
            liq_by = large_liquidations(liq_bybit, base, threshold_usd)
            _mark_direction(f, t_tr, buy_tr, liq_by, window_us, direction,
                            visible_shift_us=BYBIT_VISIBILITY_US)
    return f


def classify_trades(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,            # signature-compatible; the filter doesn't need it
    liq_binance: pl.DataFrame,
    liq_bybit: pl.DataFrame,
    *,
    threshold_usd=DEFAULT_THRESHOLD_USD,
    window_seconds: int = DEFAULT_WINDOW_S,
    direction: str = DEFAULT_DIRECTION,
    use_bybit: bool = False,
) -> dict[int, np.ndarray]:
    """Submission entry point: {tau_s: 0/1 filter array} for tau_s in (30, 120, 300).

    The signal is horizon-independent, so the three arrays are identical (the per-tau dict
    just matches the required format)."""
    f = build_filter(trades, liq_binance, liq_bybit, threshold_usd=threshold_usd,
                     window_seconds=window_seconds, direction=direction, use_bybit=use_bybit)
    return {tau: f.copy() for tau in TAUS_S}


# ----------------------------------------------------------------------------------
# Evaluation: one day-by-day pass, several named filters head-to-head (exact sums).
# ----------------------------------------------------------------------------------
def _zero_acc(names):
    keys = ["wp", "w", "kwp", "kw", "fwp", "fw", "keep_n", "n"]
    return {fn: {t: {k: 0.0 for k in keys} for t in TAUS_S} for fn in names}


def evaluate(
    name: str,
    lo: int,
    hi: int,
    filters_fn,
    sym: str = "BTC",
    stride_days: int = 2,
    verbose: bool = True,
) -> dict:
    """Loop [lo, hi) one day at a time; ``filters_fn(trades, bbo, liq_bn, liq_by)`` returns
    {filter_name: 0/1 array}. Accumulate exact weighted sums and report Score/PnL/turnover.

    Returns ``{filter_name: {tau: metrics_dict}}`` for table building in the notebook.
    """
    pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    acc, names, n_days = None, None, 0
    day = lo
    while day < hi:
        trades, bbo, liq_bn, liq_by = load_day(sym, day, day + DAY_US, pad_us)
        day += stride_days * DAY_US
        if trades.is_empty() or bbo.is_empty():
            continue
        n_days += 1
        weight = np.minimum((trades["price"] * trades["amount"]).to_numpy(), CLIP)
        fdict = filters_fn(trades, bbo, liq_bn, liq_by)
        if names is None:
            names = list(fdict)
            acc = _zero_acc(names)
        pnls = {t: markout_pnl_bps(trades, bbo, t * 1_000_000) for t in TAUS_S}
        for t in TAUS_S:
            valid = np.isfinite(pnls[t])
            p, w = pnls[t][valid], weight[valid]
            for fn in names:
                ff = fdict[fn][valid].astype(float)
                kw, fw = w * (1.0 - ff), w * ff
                s = acc[fn][t]
                s["wp"] += float((w * p).sum());   s["w"] += float(w.sum())
                s["kwp"] += float((kw * p).sum()); s["kw"] += float(kw.sum())
                s["fwp"] += float((fw * p).sum()); s["fw"] += float(fw.sum())
                s["keep_n"] += float((1.0 - ff).sum()); s["n"] += float(len(p))

    out = {}
    for fn in names or []:
        out[fn] = {}
        for t in TAUS_S:
            s = acc[fn][t]
            pa = s["wp"] / max(s["w"], 1e-9)
            pk = s["kwp"] / max(s["kw"], 1e-9)
            pf = s["fwp"] / max(s["fw"], 1e-9)
            out[fn][t] = dict(
                score=pk - pa, pnl_all=pa, pnl_kept=pk, pnl_filtered=pf,
                kept_frac=s["keep_n"] / max(s["n"], 1e-9),
                turnover_per_day=s["kw"] / max(n_days, 1),
            )
    out["_n_days"] = n_days
    if verbose:
        _print_eval(name, sym, out)
    return out


def _print_eval(name: str, sym: str, out: dict) -> None:
    n_days = out.get("_n_days", 0)
    print(f"\n=== {name}  ({sym}, {n_days} days) ===")
    for fn, per_tau in out.items():
        if fn == "_n_days":
            continue
        print(f"\n  [{fn}]")
        print(f"    {'tau':>4} {'Score':>8} {'PnL_all':>8} {'PnL_kept':>9} {'PnL_filt':>9} "
              f"{'kept%':>6} {'turn/day(USD)':>15} {'>=500k?':>8}")
        for t in TAUS_S:
            m = per_tau[t]
            ok = "OK" if m["turnover_per_day"] >= TURNOVER_MIN else "FAIL"
            print(f"    {t:>4} {m['score']:>+8.3f} {m['pnl_all']:>+8.3f} "
                  f"{m['pnl_kept']:>+9.3f} {m['pnl_filtered']:>+9.3f} "
                  f"{m['kept_frac']*100:>5.1f}% {m['turnover_per_day']:>15,.0f} {ok:>8}")
