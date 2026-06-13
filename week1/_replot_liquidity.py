"""Cleaner liquidity-recovery plot from the cached hw3_liquidity.npz:
 - spread (bps) vs time, both thresholds (headline inverse-liquidity proxy);
 - top-of-book depth normalised to the pre-event level (%), 95th pctl, total/hit/opposite.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Z = np.load("hw3_liquidity.npz")
offs = Z["offs_s"]; n_days = int(Z["n_days"][0])
pre = offs <= -10                     # clean pre-event window (avoid clustered-liq contamination near 0)


def g(q, k): return Z[f"q{q}_{k}"]


fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# left: spread (bps)
for q, col in ((95, "#DD8452"), (99, "#8172B3")):
    sp = g(q, "spread")
    ax[0].plot(offs, sp, "-o", ms=3, color=col, label=f"{q}th pctl (pre {sp[pre].mean():.3f} bps)")
ax[0].axvline(0, color="k", lw=0.8); ax[0].axhline(g(99, "spread")[pre].mean(), color="grey", ls=":", lw=1)
ax[0].set_xlim(-30, 180); ax[0].set_xlabel("seconds after liquidation"); ax[0].set_ylabel("spread (bps)")
ax[0].set_title("Spread widens ~3x at the liquidation, recovers in ~30-60 s")
ax[0].legend(fontsize=8)

# right: depth normalised to pre-event = 100%, 95th pctl (more events = smoother)
q = 95
for k, lab, col in (("total", "total BBO depth", "#4C72B0"),
                    ("hit", "hit side (consumed)", "#C44E52"),
                    ("opp", "opposite side", "#55A868")):
    v = g(q, k); norm = v / v[pre].mean() * 100
    ax[1].plot(offs, norm, "-o", ms=3, color=col, label=lab)
ax[1].axvline(0, color="k", lw=0.8); ax[1].axhline(100, color="grey", ls=":", lw=1, label="pre-event = 100%")
ax[1].set_xlim(-30, 180); ax[1].set_xlabel("seconds after liquidation")
ax[1].set_ylabel("top-of-book depth (% of pre-event)")
ax[1].set_title(f"Depth recovery — {q}th-pctl liqs (>= ${Z['thr'][0]:,.0f})")
ax[1].legend(fontsize=8)

fig.suptitle(f"BTC liquidity reaction to large liquidations (train, {n_days} days)", y=1.02)
plt.tight_layout()
fig.savefig("hw3_liquidity.png", dpi=120, bbox_inches="tight")
print("re-saved hw3_liquidity.png")
# print recovery half-life-ish summary
for q in (95, 99):
    sp = g(q, "spread"); base = sp[pre].mean(); peak = sp[(offs >= 0) & (offs <= 2)].max()
    rec = offs[(offs > 0) & (sp <= base * 1.3)]
    print(f"{q}th: pre {base:.3f} bps, peak {peak:.3f} bps (~{peak/base:.1f}x), "
          f"back within 30%% of pre by ~{rec.min() if len(rec) else float('nan'):.0f}s")
