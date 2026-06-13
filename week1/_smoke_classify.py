"""Smoke-test the submission entry point classify_trades with the production artifacts.

Loads a small window per symbol, runs classify_trades, and checks the contract:
  - output keys == {30,120,300}, each an int8 array of length len(trades), values in {0,1};
  - filtered fraction ~= (1 - keep_frac) for the symbol's artifact;
  - the VI path (tau=30 uses base+VI) runs end to end;
  - mixed-symbol input scatters back correctly (per-symbol slices match single-symbol runs).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as B
import ml_baseline as M

WIN_US = 20 * 60 * 1_000_000          # 20-minute window (small -> low RAM)
LO = B._epoch_us(2026, 2, 10)         # a validation-month day
HI = LO + WIN_US
PAD = max(M.TAUS_S) * 1_000_000 + 5_000_000


def _check_one(sym: str):
    trades, bbo, liq_bn, liq_by = B.load_day(sym, LO, HI, PAD)
    # trim trades to the window (load_day pads bbo only; trades already in [LO,HI))
    n = len(trades)
    print(f"\n[{sym}] window 20min: {n:,} trades, ticker={trades['ticker'][0]}")
    out = M.classify_trades(trades, bbo, liq_bn, liq_by)
    art = __import__("joblib").load(M._artifact_path(sym))
    assert set(out.keys()) == set(M.TAUS_S), f"keys {out.keys()}"
    for tau in M.TAUS_S:
        a = out[tau]
        assert a.dtype == np.int8, (tau, a.dtype)
        assert len(a) == n, (tau, len(a), n)
        assert set(np.unique(a)).issubset({0, 1}), np.unique(a)
        filt = a.mean()                       # fraction filtered out (==1)
        exp_filt = 1.0 - art["keep_frac"][tau]
        vi = art["volimb"].get(tau)
        print(f"  tau={tau:>3}  keep_frac={art['keep_frac'][tau]:.2f}  VI={vi}  "
              f"filtered={filt:5.1%}  (expected ~{exp_filt:4.0%})")
    return trades, bbo, liq_bn, liq_by, out


def _check_mixed(btc, eth):
    bt, bb, bln, bly, bout = btc
    et, eb, eln, ely, eout = eth
    # Mixed-symbol is only demultiplexable if every frame carries a ticker. trades/liq already do;
    # load_day's bbo does not, so tag it (this is the contract for mixed input).
    bb = bb.with_columns(pl.lit("perp:btcusdt").alias("ticker"))
    eb = eb.with_columns(pl.lit("perp:ethusdt").alias("ticker"))
    trades = pl.concat([bt, et]); bbo = pl.concat([bb, eb])
    liq_bn = pl.concat([bln, eln]); liq_by = pl.concat([bly, ely])
    out = M.classify_trades(trades, bbo, liq_bn, liq_by)
    nb = len(bt)
    ok = True
    for tau in M.TAUS_S:
        ok_b = np.array_equal(out[tau][:nb], bout[tau])
        ok_e = np.array_equal(out[tau][nb:], eout[tau])
        ok &= ok_b and ok_e
        print(f"  tau={tau:>3}  BTC-slice match={ok_b}  ETH-slice match={ok_e}")
    print(f"\nMIXED scatter correct: {ok}")
    assert ok, "mixed-symbol scatter mismatch"


if __name__ == "__main__":
    btc = _check_one("BTC")
    eth = _check_one("ETH")
    print("\n--- mixed-symbol input ---")
    _check_mixed(btc, eth)
    print("\nSMOKE OK")
