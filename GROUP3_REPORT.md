# Group 3 — Multi-Engine LOB Simulation

Multiple independent trading engines trade simultaneously in a simulated limit
order book reconstructed from historical L3 data. Each engine sees a **consistent
private view** = the real historical book **+ its own synthetic orders − the
liquidity it already consumed**, as if it were the only participant.

This is the single reference doc for Group 3: **design**, **what exists**, **how
to run it**, and **how to test it**.

---

## 1. Key design decision: N copies or diffs?

**Diffs.** One shared, read-only `HistoricalLOB` (the "basement") plus one small
per-engine overlay (`EngineView`). The book an engine sees (`SimulatedLOB`) is a
**lazy merge**, never a materialised copy.

| | N full copies | Diffs (chosen) |
|---|---|---|
| Memory | `O(N · book_depth)` | `O(N · live_own_orders)` |
| Replay cost | replay/copy per engine | replay once, shared |
| Cross-engine consistency | copies drift | one source of truth |
| Concurrency | N mutable books to guard | basement immutable → lock-free reads |

The historical book is identical for every engine and is **not** changed by any
engine's actions. Each engine deviates from history only by (a) a few of its own
resting orders and (b) a little liquidity it ate. Storing just that delta is far
smaller than a full copy and keeps a single source of truth.

Demonstrated empirically on real data (§5): 3 engines over 30 000 snapshots hold
**2 live diff entries**, where N full copies would duplicate `3 × 50 = 150`
level-cells *every snapshot*. The equivalence of the two approaches is proven by a
differential test (§6).

---

## 2. Architecture

```
            L3 event stream (Databento MBO, or L2-snapshot shim)
                              │
                              ▼  (writer / replay thread)
                      ┌───────────────┐
                      │ HistoricalLOB │  single-writer, mutable
                      └───────┬───────┘
                              │ publish() → immutable BookSnapshot
              ┌───────────────┼───────────────┐   (atomic pointer)
              ▼               ▼               ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │ EngineView0│   │ EngineView1│   │ EngineViewN│  per-engine diff
     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
           ▼                ▼                ▼
     SimulatedLOB     SimulatedLOB     SimulatedLOB   = snapshot ⊕ view (lazy)
   (engine 0 thread) (engine 1 thread) (engine N thread)
```

### Components — [src/cmf_mm/simlob/](src/cmf_mm/simlob/)

| File | Component | Role |
|---|---|---|
| [events.py](src/cmf_mm/simlob/events.py) | `L3Event`, `L3Action` | Canonical order-by-order event (ADD/MODIFY/CANCEL/TRADE/CLEAR). Everything builds on this so the data source is swappable |
| [historical.py](src/cmf_mm/simlob/historical.py) | `HistoricalLOB`, `BookSnapshot` | Single-writer basement; after applying events `publish()`es an immutable `BookSnapshot` |
| [engine_view.py](src/cmf_mm/simlob/engine_view.py) | `EngineView`, `OwnOrder` | One engine's diff: own resting orders + consumed historical liquidity per `(side, price)` |
| [simulated.py](src/cmf_mm/simlob/simulated.py) | `SimulatedLOB`, `FillSimulator` | Lazy merge `effective(side,px)=max(0, historical−consumed)+own`; fill-at-touch |
| [multi_engine.py](src/cmf_mm/simlob/multi_engine.py) | `MultiEngineSimulator` | Owns the basement + N views; writer/engine entry points |
| [adapters.py](src/cmf_mm/simlob/adapters.py) | `DatabentoMBOAdapter`, `L2ToL3Converter`, `l2_snapshots_to_l3` | Source adapters (prod MBO + L2-snapshot shim) |
| [realdata.py](src/cmf_mm/simlob/realdata.py) | `iter_lob_snapshots` | Reads the project's `data/lob.parquet` into L2 snapshots |
| [scenario.py](src/cmf_mm/simlob/scenario.py) | `build_frames` | Scripted demo scenario shared by the visualisations |
| [cpp/multi_engine_lob.hpp](cpp/multi_engine_lob.hpp) | C++ sketch | Same design in true OS threads |

### Deliverables checklist

| Requirement | Status |
|---|---|
| **HistoricalLOB** reconstructed from L3 replay | ✅ |
| **EngineView** per-engine overlay tracking own orders | ✅ |
| **SimulatedLOB** = HistoricalLOB + EngineView | ✅ |
| Fill simulation "at or through best historical price" (fill-at-touch) | ✅ marketable + resting |
| N concurrent engine views, one shared HistoricalLOB, no data races | ✅ (Python: GIL; true concurrency in the C++ sketch) |
| Engine doesn't see liquidity it already consumed | ✅ |
| Key design question (copies vs diffs) answered | ✅ §1 |

---

## 3. How it works

### Fill model (fill at touch)

* **Marketable on submit** — a buy whose price ≥ best historical ask (or sell ≤
  best bid) executes immediately, walking levels up to its limit price and filling
  **at the resting price of each level** (the touch). Eaten volume is recorded as
  *consumed* in the engine's view, so that engine no longer sees it — other engines
  still do.
* **Passive / resting** — a non-marketable order rests in the `EngineView`. On
  every book advance, `check_resting_fills()` re-checks it: a resting buy fills
  once the best historical ask trades down to its price (symmetric for sells),
  again at the touch price.

Each `OwnOrder` records `volume_ahead` (historical volume resting at its price when
it joined the queue) — the hook for a future FIFO queue-position model with no
schema change.

### Concurrency — N engines, one book, no data races

The only object shared between threads is `HistoricalLOB`, shared **read-only**
through immutable snapshots:

* The **writer thread** mutates the book and `publish()`es a new immutable
  `BookSnapshot`, exposed by swapping a single reference (`latest`).
* **Engine threads** load that reference (atomic under the CPython GIL; an
  `atomic_load` on a `shared_ptr<const>` in C++) and merge it with their own
  `EngineView`, which no other thread touches.

No shared mutable state on the hot path, no locks: readers never see a torn book (a
snapshot is fully built before publication), and a reader holding an older snapshot
stays consistent while the writer races ahead (RCU-style copy-on-publish).
[cpp/multi_engine_lob.hpp](cpp/multi_engine_lob.hpp) maps the identical design to
true OS threads with `std::atomic<std::shared_ptr<const BookSnapshot>>`.

### Production data path

The core consumes only `L3Event`, so the source is an adapter swap:

* **Production** — `DatabentoMBOAdapter`: Databento MBO is already true L3; the
  adapter is a 1:1 field/action rename (typed stub until the files land).
* **Today's data** — `l2_snapshots_to_l3` synthesises pseudo-L3 from the per-level
  deltas of `data/lob.parquet` (one synthetic order per price level:
  ADD/MODIFY/CANCEL). The reconstructed book matches the recorded snapshots exactly
  (§5); only per-order queue identity is approximate — enough for fill-at-touch.

### Known limitations

* **`consumed` is monotonic.** An engine's consumed liquidity is a sticky scalar
  per `(side, price)` that never expires. Over a long replay where prices revisit
  the same levels, an engine's *own past* consumption keeps hiding that level from
  its view, even though later historical orders at that price are different orders
  it should be able to trade. Production fix: reset/decay `consumed` per replay
  window (the basement is re-published per window anyway). The aggregated
  price→volume model also can't distinguish individual orders at a level — intrinsic
  to L2 aggregation; true MBO (L3) keeps per-order identity.
* **Queue position is approximate.** `OwnOrder.volume_ahead` is captured but the
  fill-at-touch model ignores it; FIFO queue-aware fills are the next step.

---

## 4. Setup

```bash
# from CMFtestcase/
uv sync --extra viz --extra dev      # core + viz (streamlit/plotly) + dev tools
```

> On Windows, `uv run mypy ...` / `uv run pytest ...` (console scripts) can hit a
> uv trampoline error — use the module form `uv run python -m mypy ...` /
> `uv run python -m pytest ...` (as below).

---

## 5. How to run

### A. Text trace + static figures (zero-config)
```bash
uv run python scripts/simlob_demo.py
```
Prints a step-by-step decision trace; writes `results/simlob/depth_stages.png`
and `results/simlob/fills_timeline.png`.

### B. Interactive HTML player (no server)
```bash
uv run python scripts/simlob_animate.py
# open results/simlob/simlob_animation.html  → play / pause / step / scrub
```

### C. Streamlit dashboard (richest)
```bash
uv run streamlit run scripts/simlob_app.py
# open http://localhost:8501
```
Dark-themed: animated data-flow diagram (active component highlighted), interactive
depth charts (hover), metric cards, decision log, fills table, and controls (step
slider, prev/next, auto-play, engine multiselect).

### D. Real-data replay (the "real market" path)
```bash
uv run python scripts/simlob_replay_real.py --rows 30000 --engines 3
```
Reconstructs `data/lob.parquet` and runs N engines on it. Example output:
```
snapshots reconstructed : 30,000
top-of-book mismatches  : 0  (PERFECT)
historical depth (levels): 25 bid / 25 ask
synthetic fills (all eng): 6,508
live own orders (diff)  : 2   [N full copies would store 3×50 = 150 cells, every snapshot]
```

---

## 6. How to test

### Run everything
```bash
uv run python -m pytest -q                       # full project suite
uv run python -m pytest tests/test_simlob*.py -q # just Group 3 (53 tests)
```

### Quality gates
```bash
uv run python -m mypy --strict src/cmf_mm/simlob   # type check (clean)
uv run ruff check src/cmf_mm/simlob tests scripts  # lint (clean)
```

### Test layers (53 Group-3 tests)

| File | Tests | Layer — what it proves |
|---|---|---|
| [tests/test_simlob.py](tests/test_simlob.py) | 9 | **Unit** — reconstruction, L2→L3 shim, isolation, fill-at-touch, 8-engine concurrency smoke |
| [tests/test_simlob_invariants.py](tests/test_simlob_invariants.py) | 17 | **Invariants** — basement isolation, determinism, merge identity, non-negativity, snapshot immutability, fill-price sanity (12 randomised seeds) |
| [tests/test_simlob_oracle.py](tests/test_simlob_oracle.py) | 25 | **Differential oracle** — an independent "N full copies" implementation; 25 seeds × 60 ops compare books + fills every step → proves diffs ≡ copies |
| [tests/test_simlob_realdata.py](tests/test_simlob_realdata.py) | 2 | **Real-data integration** — 5 000 real snapshots reconstructed with full-depth match to the recorded book; N engines trade on the real slice. Auto-skips if `data/lob.parquet` is absent |

Run one layer, e.g. the differential oracle:
```bash
uv run python -m pytest tests/test_simlob_oracle.py -q
```

### Coverage map

| Layer | Status |
|---|---|
| 1 · Unit | ✅ 9 |
| 2 · Invariants / property | ✅ 17 |
| 3 · Differential oracle (copies vs diffs) | ✅ 25 |
| 4 · Concurrency | 🟡 8-engine smoke (Python GIL limits true parallelism; real check = C++ + ThreadSanitizer) |
| 5 · Real-data integration | ✅ 2 + 30k-snapshot replay, 0 mismatches |
| 6 · Performance / scale | 🟡 memory (diffs vs copies) shown in replay; formal throughput bench pending |

---

## 7. Open items / next steps

- **Layer 4** — true concurrency can only be validated in the C++ implementation
  (build [cpp/multi_engine_lob.hpp](cpp/multi_engine_lob.hpp) under ThreadSanitizer);
  Python's GIL hides races.
- **Layer 6** — formal throughput/latency benchmarks (events/sec, fill latency).
- **Fill model** — FIFO queue-aware fills using `OwnOrder.volume_ahead`.
- **`consumed` decay** — reset per replay window so a taker's old impact expires.
