from __future__ import annotations

from cmf_mm.engine.order_manager import OrderManager
from cmf_mm.types import QuoteAction


def test_place_assigns_unique_ids():
    om = OrderManager()
    a = om.place("buy", 100.0, 1.0, ts=0)
    b = om.place("sell", 101.0, 1.0, ts=0)
    assert a != b
    assert len(om.active_orders) == 2


def test_cancel_removes_order():
    om = OrderManager()
    oid = om.place("buy", 100.0, 1.0, ts=0)
    om.cancel(oid)
    assert oid not in om.active_orders


def test_reconcile_places_when_no_active():
    om = OrderManager()
    om.reconcile([
        QuoteAction(side="buy", price=99.0, size=1.0),
        QuoteAction(side="sell", price=101.0, size=1.0),
    ], ts=0)
    sides = {o.side for o in om.active_orders.values()}
    assert sides == {"buy", "sell"}


def test_reconcile_replaces_when_price_changes():
    om = OrderManager()
    om.reconcile([QuoteAction(side="buy", price=99.0, size=1.0)], ts=0)
    first_id = next(iter(om.active_orders))
    om.reconcile([QuoteAction(side="buy", price=98.0, size=1.0)], ts=1)
    new_id = next(iter(om.active_orders))
    assert first_id != new_id
    assert next(iter(om.active_orders.values())).price == 98.0


def test_reconcile_keeps_when_unchanged():
    om = OrderManager()
    om.reconcile([QuoteAction(side="buy", price=99.0, size=1.0)], ts=0)
    first_id = next(iter(om.active_orders))
    om.reconcile([QuoteAction(side="buy", price=99.0, size=1.0)], ts=1)
    second_id = next(iter(om.active_orders))
    assert first_id == second_id


def test_reconcile_cancels_side_dropped_from_actions():
    om = OrderManager()
    om.reconcile([
        QuoteAction(side="buy", price=99.0, size=1.0),
        QuoteAction(side="sell", price=101.0, size=1.0),
    ], ts=0)
    om.reconcile([QuoteAction(side="buy", price=99.0, size=1.0)], ts=1)
    sides = [o.side for o in om.active_orders.values()]
    assert sides == ["buy"]
