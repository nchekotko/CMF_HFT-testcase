# Roadmap

Six concrete extensions, in rough order of expected information gain per
engineering hour. Each item lists the mathematical sketch, citation, and
effort estimate.

## 1. Full Stoikov (2018) micro-price via Markov chain

The micro-price estimator currently shipped is the volume-weighted top-of-book
formula. Stoikov (2018) shows that this is biased and proposes the unbiased
limit

```
G_∞ = R + Q · G_∞   ⇒   G_∞ = (I − Q)⁻¹ · R
```

estimated on a discretised Markov chain over `(imbalance, spread)` states.
The interface in `src/cmf_mm/lob/microprice.py` is already prepared
(`MicropriceEstimator` ABC; `StoikovMarkovChainEstimator` stub).

* Citation: Stoikov, S. (2018). *The Micro-Price: a high-frequency estimator of future prices.* Quantitative Finance, 18(12), 1959–1966.
* Effort: ~2 weeks (1 week implementation + 1 week diagnostics).

## 2. Adverse-selection penalty in the objective

AS 2008 ignores informed flow: the maker is treated as if every fill carried
the same expected drift in the mid. Cartea–Jaimungal–Penalva extend the HJB
to penalise fills against informed counter-parties, which is essentially a
state-dependent γ.

* Citation: Cartea, Á., Jaimungal, S., Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* CUP, ch. 10–11.
* Effort: ~1 week of implementation + 3–5 days for theory and writeup.

## 3. Hawkes-process arrivals instead of Poisson

The constant-A Poisson assumption ignores self-excitation in trade flow,
which is empirically strong. Replacing λ(δ) with a Hawkes intensity

```
λ(t, δ) = μ(δ) + Σ_{tᵢ < t} φ(t − tᵢ; δ)
```

re-prices the optimal-spread term and tightens the model around volatile
regimes. The branching ratio is also a useful diagnostic for adverse-selection
risk.

* Citation: Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). *Hawkes Processes in Finance.* Market Microstructure and Liquidity, 1(01), 1550005.
* Effort: ~2 weeks (calibration + closed-form spread is non-trivial).

## 4. Queue-position modelling

The engine currently assumes that any resting order fills the moment a trade
crosses its level. This is conservative for an MM backtest with no queue
information, but it costs us realism on the spread-capture side. Adding a
queue-position estimator — Cont & Kukanov style — would let us model the
expected time-to-fill and choose levels accordingly.

* Citation: Cont, R., Kukanov, A. (2017). *Optimal order placement in limit order markets.* Quantitative Finance, 17(1), 21–39.
* Effort: ~1.5 weeks.

## 5. Multi-asset MM with cointegration

Single-asset MM ignores cross-asset hedging opportunities. Treating the
inventory penalty as quadratic in a vector q with covariance Σ recovers a
joint-quoting policy whose practical impact is large for cointegrated pairs.

* Citation: Avellaneda, M., Lee, J.-H. (2010). *Statistical Arbitrage in the U.S. Equities Market.* Quantitative Finance, 10(7), 761–782.
* Effort: ~3 weeks (data layer needs to handle multiple symbols cleanly).

## 6. RL parameter adaptation

Static γ is a one-knob fit to an unconditional regime; in practice the
optimal risk-aversion is regime-dependent (volatility, intensity, queue
imbalance). Replace it with a learned policy. This connects to my prior work
on portfolio optimisation with SAC + Transformer state encoders.

* Citation: Spooner, T., Fearnley, J., Savani, R., Koukorinis, A. (2018).
  *Market Making via Reinforcement Learning.* AAMAS-18.
* Effort: ~4 weeks. Requires the simulator to be RL-friendly (gym wrapper).

## Smaller, opportunistic items

* **Latency model** — add a fixed or distributional submit-to-ack delay.
* **Partial fills with explicit queue depth** — drop the no-partial default once item 4 lands.
* **Walk-forward calibration** — recalibrate σ, k, A per rolling window in test.
* **Bootstrap CIs on Sharpe / decomposition** — reportable uncertainty bands.
