"""Lightweight timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def timed(label: str, sink: Any | None = None) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        msg = f"[timing] {label}: {dt:.3f}s"
        if sink is None:
            print(msg)
        else:
            sink.info(msg)
