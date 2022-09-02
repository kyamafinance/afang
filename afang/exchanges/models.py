from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    UNKNOWN = "UNKNOWN"


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


@dataclass
class SymbolBalance:
    name: str
    wallet_balance: float


@dataclass
class Order:
    symbol: str
    order_id: str
    side: OrderSide
    original_price: Optional[float]
    average_price: float
    original_quantity: float
    executed_quantity: float
    remaining_quantity: float
    order_type: OrderType
    order_status: str
    time_in_force: str
    commission: Optional[float] = None
