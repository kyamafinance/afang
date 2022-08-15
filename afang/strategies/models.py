from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeLevels:
    entry_price: float
    target_price: Optional[float]
    stop_price: Optional[float]
