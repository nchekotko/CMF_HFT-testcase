"""Multi-engine LOB simulation (assignment Group 3).

A shared, historically-reconstructed order book (the "basement") over which N
independent trading engines each see a private view = real book + their own
synthetic orders − liquidity they already consumed.

Design choice (N copies vs diffs): one shared :class:`HistoricalLOB` plus
per-engine *diffs* (:class:`EngineView`); the seen book (:class:`SimulatedLOB`)
is a lazy merge, never a materialised copy. See ``GROUP3_REPORT.md`` at the
project root for the full design, run and test guide.
"""

from __future__ import annotations

from .adapters import (
    DatabentoMBOAdapter,
    L2Snapshot,
    L2ToL3Converter,
    L3Source,
    l2_snapshots_to_l3,
)
from .engine_view import EngineView, OwnOrder
from .events import L3Action, L3Event
from .historical import BookSnapshot, HistoricalLOB
from .multi_engine import MultiEngineSimulator
from .simulated import FillSimulator, SimulatedLOB

__all__ = [
    "BookSnapshot",
    "DatabentoMBOAdapter",
    "EngineView",
    "FillSimulator",
    "HistoricalLOB",
    "L2Snapshot",
    "L2ToL3Converter",
    "L3Action",
    "L3Event",
    "L3Source",
    "MultiEngineSimulator",
    "OwnOrder",
    "SimulatedLOB",
    "l2_snapshots_to_l3",
]
