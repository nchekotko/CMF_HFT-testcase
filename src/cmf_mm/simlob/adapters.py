"""Source adapters: turn a vendor-specific feed into the canonical L3 stream.

The core (:class:`~cmf_mm.simlob.historical.HistoricalLOB`) only ever consumes
:class:`~cmf_mm.simlob.events.L3Event`. Swapping data sources is therefore an
adapter change with zero core edits — the production path (Databento MBO) and the
"run on what we have today" path (L2 snapshot diffing) sit behind the same
interface.

Two adapters:

* :class:`DatabentoMBOAdapter` — the *production* path. Databento MBO is already
  true order-by-order L3; the adapter is a 1:1 field rename. (Left as a typed
  stub here because the JSON/Feather files are not in this repo yet.)

* :func:`l2_snapshots_to_l3` — the *fallback* path. Our current ``lob.parquet``
  holds aggregated L2 snapshots, not L3. We synthesise pseudo-L3 events from the
  per-level *deltas* between consecutive snapshots, modelling each price level as
  a single synthetic order (ADD when it appears, MODIFY when its size changes,
  CANCEL when it vanishes). Order identity is per-level rather than per-order, so
  queue position is approximate — but HistoricalLOB reconstructs the exact same
  aggregated book, which is all fill-at-touch needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from ..types import OrderId, Side
from .events import L3Action, L3Event


class L3Source(Protocol):
    """Anything that yields a chronologically ordered L3 event stream."""

    def __iter__(self) -> Iterator[L3Event]: ...


class DatabentoMBOAdapter:
    """Production adapter for Databento MBO (true L3). Stub: wire up on ingest.

    Databento ``action`` codes map directly: A->ADD, C->CANCEL, M->MODIFY,
    T/F->TRADE, R->CLEAR. ``order_id``, ``side`` (B/A), ``price``, ``size`` are
    native fields. Implement ``__iter__`` over the decoded records when the
    JSON/Feather files land.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def __iter__(self) -> Iterator[L3Event]:  # pragma: no cover - stub
        raise NotImplementedError(
            "Wire DatabentoMBOAdapter to the JSON/Feather reader on ingest; "
            "map A/C/M/T/F/R actions to L3Action."
        )


L2Snapshot = tuple[int, list[tuple[float, float]], list[tuple[float, float]]]
"""``(ts, bids, asks)`` with bids/asks as lists of ``(price, size)``."""


class L2ToL3Converter:
    """Stateful L2-snapshot → synthetic-L3 stepper.

    Each price level is modelled as one synthetic order whose id persists while
    the level exists. Between consecutive snapshots, for every level:

    * appeared           -> ADD (size = new)
    * size changed       -> MODIFY (size = new)
    * vanished           -> CANCEL

    The state (previous levels + id map) persists across :meth:`step` calls, so a
    caller can drive a HistoricalLOB one snapshot at a time and check the
    reconstruction against each snapshot — the basis of the real-data test.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._level_id: dict[tuple[Side, float], OrderId] = {}
        self._prev: dict[tuple[Side, float], float] = {}

    def step(self, snapshot: L2Snapshot) -> list[L3Event]:
        """Emit the L3 events that turn the previous book into ``snapshot``."""
        ts, bids, asks = snapshot
        curr: dict[tuple[Side, float], float] = {}
        for side, levels in (("buy", bids), ("sell", asks)):
            for px, sz in levels:
                if sz > 1e-12:
                    curr[(side, px)] = sz  # type: ignore[index]

        events: list[L3Event] = []
        for key in set(self._prev) | set(curr):
            old = self._prev.get(key, 0.0)
            new = curr.get(key, 0.0)
            if abs(new - old) <= 1e-12:
                continue
            side, px = key
            if old <= 1e-12:  # appeared
                self._counter += 1
                oid = OrderId(self._counter)
                self._level_id[key] = oid
                events.append(L3Event(ts, L3Action.ADD, oid, side, px, new))
            elif new <= 1e-12:  # vanished
                oid = self._level_id.pop(key)
                events.append(L3Event(ts, L3Action.CANCEL, oid, side, px, old))
            else:  # resized
                events.append(
                    L3Event(ts, L3Action.MODIFY, self._level_id[key], side, px, new))
        self._prev = curr
        return events


def l2_snapshots_to_l3(snapshots: Iterable[L2Snapshot]) -> Iterator[L3Event]:
    """Flatten a stream of L2 snapshots into a synthetic L3 event stream.

    Thin wrapper over :class:`L2ToL3Converter` for callers that just want the
    flat event stream (e.g. feeding a HistoricalLOB in one pass).
    """
    conv = L2ToL3Converter()
    for snap in snapshots:
        yield from conv.step(snap)
