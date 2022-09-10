from enum import Enum


class Mode(Enum):
    data = "data"
    backtest = "backtest"
    optimize = "optimize"
    trade = "trade"


class Exchange(Enum):
    binance = "binance"
    dydx = "dydx"


class Timeframe(Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"
