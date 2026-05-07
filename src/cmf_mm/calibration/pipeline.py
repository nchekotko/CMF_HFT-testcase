"""Combined calibration entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .intensity import IntensityResult, calibrate_intensity
from .volatility import VolatilityResult, estimate_sigma


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    volatility: VolatilityResult
    intensity: IntensityResult

    def to_dict(self) -> dict[str, Any]:
        return {"volatility": asdict(self.volatility), "intensity": asdict(self.intensity)}


def calibrate_all(
    trades_parquet: str | Path,
    lob_parquet: str | Path,
    train_start_ts: int | None,
    train_end_ts: int | None,
    n_intensity_bins: int = 20,
    sigma_sample_stride: int = 1,
) -> CalibrationResult:
    vol = estimate_sigma(
        lob_parquet,
        start_ts=train_start_ts,
        end_ts=train_end_ts,
        sample_stride=sigma_sample_stride,
    )
    intensity = calibrate_intensity(
        trades_parquet,
        lob_parquet,
        start_ts=train_start_ts,
        end_ts=train_end_ts,
        n_bins=n_intensity_bins,
    )
    return CalibrationResult(volatility=vol, intensity=intensity)
