"""Inventory statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class InventoryStats:
    mean: float
    std: float
    max_abs: float
    time_weighted_avg: float
    final: float


def inventory_stats(timestamps: np.ndarray, inventory: np.ndarray) -> InventoryStats:
    if inventory.size == 0:
        return InventoryStats(mean=0.0, std=0.0, max_abs=0.0, time_weighted_avg=0.0, final=0.0)

    mean = float(inventory.mean())
    std = float(inventory.std(ddof=1)) if inventory.size > 1 else 0.0
    max_abs = float(np.max(np.abs(inventory)))
    final = float(inventory[-1])

    if inventory.size < 2:
        twa = mean
    else:
        dt = np.diff(timestamps).astype(np.float64)
        seg = 0.5 * (inventory[:-1] + inventory[1:])
        total_t = float(dt.sum())
        twa = float(np.sum(seg * dt) / total_t) if total_t > 0 else mean

    return InventoryStats(mean=mean, std=std, max_abs=max_abs, time_weighted_avg=twa, final=final)
