"""V2: Avellaneda–Stoikov with the volume-weighted micro-price as reference.

The only change from V1 is the substitution s → p_micro in the reservation
price. All AS formulas are reused, which keeps the V1↔V2 comparison clean.
"""

from __future__ import annotations

from ..lob.book import OrderBookState
from ..lob.microprice import MicropriceEstimator, WeightedMidEstimator
from .avellaneda_stoikov import ASParams, AvellanedaStoikovStrategy


class ASMicropriceStrategy(AvellanedaStoikovStrategy):
    name = "as_microprice"

    def __init__(
        self,
        params: ASParams,
        microprice_estimator: MicropriceEstimator | None = None,
    ) -> None:
        super().__init__(params)
        self._micro = microprice_estimator or WeightedMidEstimator()

    def _reference_price(self, book: OrderBookState) -> float:
        return self._micro.estimate(book)
