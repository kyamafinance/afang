from dataclasses import dataclass


@dataclass
class Candle:
    open_time: float
    open: float
    high: float
    low: float
    close: float
    volume: float
