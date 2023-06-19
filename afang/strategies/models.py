from dataclasses import dataclass
from typing import List, Optional

from afang.database.trades_db.models import TradePosition as DBTradePosition


@dataclass
class TradeLevels:
    entry_price: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    sequence_id: Optional[str] = None


@dataclass
class AnalysisStat:
    name: str
    all_trades: float
    long_trades: float
    short_trades: float
    is_positive_optimization: bool


@dataclass
class MonthlyPnLAnalysis:
    month: str
    all_trades: float
    long_trades: float
    short_trades: float


@dataclass
class SymbolAnalysisResult:
    symbol: str
    trades: List[DBTradePosition]
    sequenced_trades: List[DBTradePosition]
    monthly_pnl: Optional[List[MonthlyPnLAnalysis]] = None
    net_profit: Optional[AnalysisStat] = None
    gross_profit: Optional[AnalysisStat] = None
    gross_loss: Optional[AnalysisStat] = None
    commission: Optional[AnalysisStat] = None
    slippage: Optional[AnalysisStat] = None
    profit_factor: Optional[AnalysisStat] = None
    maximum_drawdown: Optional[AnalysisStat] = None
    total_trades: Optional[AnalysisStat] = None
    winning_trades: Optional[AnalysisStat] = None
    losing_trades: Optional[AnalysisStat] = None
    even_trades: Optional[AnalysisStat] = None
    percent_profitable: Optional[AnalysisStat] = None
    average_roe: Optional[AnalysisStat] = None
    average_trade_pnl: Optional[AnalysisStat] = None
    average_winning_trade: Optional[AnalysisStat] = None
    average_losing_trade: Optional[AnalysisStat] = None
    take_profit_ratio: Optional[AnalysisStat] = None
    trade_expectancy: Optional[AnalysisStat] = None
    max_consecutive_winners: Optional[AnalysisStat] = None
    max_consecutive_losers: Optional[AnalysisStat] = None
    largest_winning_trade: Optional[AnalysisStat] = None
    largest_losing_trade: Optional[AnalysisStat] = None
    average_holding_time: Optional[AnalysisStat] = None
    maximum_holding_time: Optional[AnalysisStat] = None
    average_monthly_pnl: Optional[AnalysisStat] = None
