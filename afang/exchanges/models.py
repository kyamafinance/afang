from dataclasses import dataclass
from enum import Enum


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Symbol:
    name: str
    base_asset: str
    quote_asset: str
    price_decimals: int
    quantity_decimals: int
    tick_size: float
    step_size: float
