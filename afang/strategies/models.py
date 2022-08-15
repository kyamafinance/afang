from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TradeLevels:
    entry_price: float
    target_price: Optional[float]
    stop_price: Optional[float]


@dataclass
class TradePosition:
    direction: int
    entry_price: float
    entry_time: datetime
    trade_count: int
    open_position: bool = True
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    holding_time: int = 0
    close_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    position_size: Optional[float] = None
    roe: Optional[float] = None
    cost_adjusted_roe: Optional[float] = None
    pnl: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    initial_account_balance: Optional[float] = None
    final_account_balance: Optional[float] = None
