// multi_engine_lob.hpp — C++ sketch of the production concurrency design.
//
// The Python package `cmf_mm.simlob` is the executable reference implementation
// of the data structures and the fill model. This header sketches how the SAME
// design maps onto the C++ process from the architecture diagram, where the "no
// data races" requirement is enforced by the type system instead of the GIL.
//
// Design (identical to the Python version):
//   * ONE shared HistoricalLOB (the basement), advanced by a single writer
//     thread (the Backtest-Engine thread).
//   * N EngineView overlays (DIFFS, not copies), each private to one
//     Trading-Engine thread.
//   * SimulatedLOB is a lazy merge view: HistoricalLOB snapshot + EngineView.
//
// Concurrency primitive: the writer never mutates anything a reader can see.
// It builds the next book state and PUBLISHES it as an immutable snapshot via
// an atomic shared_ptr swap. Readers load the current snapshot (a lock-free
// atomic load) and hold it for as long as they need a stable view; the writer
// racing ahead only swaps in a new snapshot and never touches the old one.
// This is RCU-style publication — wait-free on the read path.
//
// Header-only sketch: bodies are illustrative, not a complete build target.

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cmf::simlob {

using OrderId = std::uint64_t;
using Px = std::int64_t;    // price in ticks (integer — exact, no FP drift)
using Qty = std::int64_t;   // size in lots
using Ts = std::int64_t;    // microseconds since epoch

enum class Side { Buy, Sell };
enum class L3Action { Add, Modify, Cancel, Trade, Clear };

struct L3Event {
  Ts ts;
  L3Action action;
  OrderId order_id;
  Side side;
  Px price;
  Qty size;
};

struct Fill {
  OrderId order_id;
  Ts ts;
  Side side;
  Px price;
  Qty size;
  Px mid_at_fill;
};

// ---------------------------------------------------------------------------
// Immutable, shareable snapshot. Once constructed it is never mutated, so any
// number of reader threads may hold `shared_ptr<const BookSnapshot>` safely.
// ---------------------------------------------------------------------------
class BookSnapshot {
 public:
  BookSnapshot(std::uint64_t seq, Ts ts,
               std::map<Px, Qty, std::greater<Px>> bids,
               std::map<Px, Qty> asks)
      : seq_(seq), ts_(ts), bids_(std::move(bids)), asks_(std::move(asks)) {}

  std::uint64_t seq() const noexcept { return seq_; }
  Ts ts() const noexcept { return ts_; }

  std::optional<Px> best_bid() const {
    return bids_.empty() ? std::nullopt : std::optional{bids_.begin()->first};
  }
  std::optional<Px> best_ask() const {
    return asks_.empty() ? std::nullopt : std::optional{asks_.begin()->first};
  }
  Qty volume_at(Side s, Px px) const {
    if (s == Side::Buy) {
      auto it = bids_.find(px);
      return it == bids_.end() ? 0 : it->second;
    }
    auto it = asks_.find(px);
    return it == asks_.end() ? 0 : it->second;
  }

  // bids_ uses greater<> so begin() is the best (highest) bid.
  const std::map<Px, Qty, std::greater<Px>>& bids() const { return bids_; }
  const std::map<Px, Qty>& asks() const { return asks_; }

 private:
  std::uint64_t seq_;
  Ts ts_;
  std::map<Px, Qty, std::greater<Px>> bids_;
  std::map<Px, Qty> asks_;
};

// ---------------------------------------------------------------------------
// HistoricalLOB — single-writer basement. Mutated only by the replay thread;
// readers see it exclusively through published immutable snapshots.
// ---------------------------------------------------------------------------
class HistoricalLOB {
 public:
  HistoricalLOB()
      : latest_(std::make_shared<const BookSnapshot>(
            0, 0, std::map<Px, Qty, std::greater<Px>>{}, std::map<Px, Qty>{})) {}

  // Writer thread only.
  void apply(const L3Event& ev);   // fold one event into bids_/asks_/orders_

  void publish() {                 // freeze + atomically expose a new snapshot
    auto snap = std::make_shared<const BookSnapshot>(++seq_, ts_, bids_, asks_);
    std::atomic_store_explicit(&latest_, snap, std::memory_order_release);
  }

  // Any thread: wait-free atomic load of the current immutable view.
  std::shared_ptr<const BookSnapshot> snapshot() const {
    return std::atomic_load_explicit(&latest_, std::memory_order_acquire);
  }

 private:
  struct Resting { Side side; Px price; Qty size; };
  std::map<Px, Qty, std::greater<Px>> bids_;
  std::map<Px, Qty> asks_;
  std::unordered_map<OrderId, Resting> orders_;
  std::uint64_t seq_ = 0;
  Ts ts_ = 0;
  // C++20: std::atomic<std::shared_ptr<const BookSnapshot>> is cleaner; the
  // free-function atomic_load/store form above works pre-C++20 too.
  std::shared_ptr<const BookSnapshot> latest_;
};

// ---------------------------------------------------------------------------
// EngineView — one engine's private diff. Touched by exactly one engine thread,
// therefore needs no synchronisation of its own.
// ---------------------------------------------------------------------------
class EngineView {
 public:
  explicit EngineView(int engine_id) : engine_id_(engine_id) {}

  struct OwnOrder { OrderId id; Side side; Px price; Qty size; Ts ts; Qty ahead; };

  void add_own(OwnOrder o) { own_.emplace(o.id, o); }
  void reduce_own(OrderId id, Qty q);
  void cancel_own(OrderId id) { own_.erase(id); }
  void consume_historical(Side s, Px px, Qty q) { consumed_[{s, px}] += q; }
  Qty consumed_at(Side s, Px px) const {
    auto it = consumed_.find({s, px});
    return it == consumed_.end() ? 0 : it->second;
  }
  const std::unordered_map<OrderId, OwnOrder>& own() const { return own_; }

 private:
  struct Key { Side s; Px px; bool operator==(const Key&) const = default; };
  struct KeyHash { std::size_t operator()(const Key& k) const noexcept {
    return std::hash<Px>{}(k.px) ^ (static_cast<std::size_t>(k.s) << 1);
  }};
  int engine_id_;
  std::unordered_map<OrderId, OwnOrder> own_;
  std::unordered_map<Key, Qty, KeyHash> consumed_;
};

// ---------------------------------------------------------------------------
// FillSimulator — binds the shared book to one engine's view; fill-at-touch.
// Lives on the engine thread. Reads the shared book via snapshot() (wait-free).
// ---------------------------------------------------------------------------
class FillSimulator {
 public:
  FillSimulator(const HistoricalLOB& hist, EngineView& view)
      : hist_(hist), view_(view) {}

  std::vector<Fill> submit_limit(OrderId id, Side side, Px price, Qty size, Ts ts);
  std::vector<Fill> check_resting_fills();

 private:
  const HistoricalLOB& hist_;  // shared, read-only from here
  EngineView& view_;           // private to this engine thread
};

// ---------------------------------------------------------------------------
// MultiEngineSimulator — owns the basement + N views. The writer thread calls
// advance(); engine threads call their own submit_limit/check_resting_fills.
// No mutex anywhere: the only shared object (HistoricalLOB) is published RCU.
// ---------------------------------------------------------------------------
class MultiEngineSimulator {
 public:
  explicit MultiEngineSimulator(int n) {
    views_.reserve(n);
    sims_.reserve(n);
    for (int i = 0; i < n; ++i) views_.emplace_back(std::make_unique<EngineView>(i));
    for (int i = 0; i < n; ++i)
      sims_.emplace_back(std::make_unique<FillSimulator>(hist_, *views_[i]));
  }

  void advance(const L3Event& ev) { hist_.apply(ev); hist_.publish(); }  // writer
  FillSimulator& engine(int id) { return *sims_[id]; }                   // engine thr

 private:
  HistoricalLOB hist_;
  std::vector<std::unique_ptr<EngineView>> views_;
  std::vector<std::unique_ptr<FillSimulator>> sims_;
};

}  // namespace cmf::simlob
