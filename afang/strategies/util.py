from typing import NamedTuple, Optional


class TradeLevels(NamedTuple):
    entry_price: float
    target_price: Optional[float]
    stop_price: Optional[float]
