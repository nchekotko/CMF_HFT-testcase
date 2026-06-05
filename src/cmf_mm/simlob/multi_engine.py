"""MultiEngineSimulator — N engine views over one shared HistoricalLOB.

This is the orchestration layer that answers the assignment's "support N
concurrent engine views with no data races". It owns:

* one :class:`HistoricalLOB` (the basement, single-writer);
* N :class:`EngineView` overlays, one per engine (diffs, not copies);
* N :class:`FillSimulator` instances binding each view to the shared book.

Threading model (mirrors the architecture diagram's Backtest-Engine /
Trading-Engine split):

* The **replay/writer thread** calls :meth:`advance` to fold L3 events into the
  historical book and publish a fresh immutable snapshot.
* Each **engine thread** only touches its own ``FillSimulator``/``EngineView``
  and reads ``historical.latest`` — an atomic reference load under the GIL. No
  engine mutates shared state, so there are no data races and no locks on the
  hot read path.

Per-engine writes never collide because each ``EngineView`` is private to one
engine. The only cross-thread hand-off is the published snapshot pointer.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..types import Fill, OrderId, Side
from .engine_view import EngineView
from .events import L3Event
from .historical import BookSnapshot, HistoricalLOB
from .simulated import FillSimulator, SimulatedLOB


class MultiEngineSimulator:
    def __init__(self, n_engines: int) -> None:
        self._hist = HistoricalLOB()
        self._views: list[EngineView] = [EngineView(i) for i in range(n_engines)]
        self._sims: list[FillSimulator] = [
            FillSimulator(self._hist, v) for v in self._views
        ]

    @property
    def historical(self) -> HistoricalLOB:
        return self._hist

    def advance(self, event: L3Event) -> BookSnapshot:
        """Writer-thread entry point: apply one historical L3 event and publish."""
        return self._hist.apply_and_publish(event)

    def advance_many(self, events: Iterable[L3Event]) -> BookSnapshot:
        """Apply a batch of events, publishing once at the end (cheaper)."""
        for event in events:
            self._hist.apply(event)
        return self._hist.publish()

    def view_for(self, engine_id: int) -> SimulatedLOB:
        """The merged book engine ``engine_id`` currently sees."""
        return self._sims[engine_id].view_now()

    def submit_limit(
        self,
        engine_id: int,
        order_id: OrderId,
        side: Side,
        price: float,
        size: float,
        ts: int,
    ) -> list[Fill]:
        """Engine-thread entry point: submit a limit order for one engine."""
        return self._sims[engine_id].submit_limit(order_id, side, price, size, ts)

    def check_resting_fills(self, engine_id: int) -> list[Fill]:
        """Re-check one engine's resting orders against the latest book."""
        return self._sims[engine_id].check_resting_fills()

    def cancel(self, engine_id: int, order_id: OrderId) -> bool:
        """Cancel a resting own order for one engine; True if it was live."""
        return self._sims[engine_id].cancel(order_id)
