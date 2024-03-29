import logging
import queue
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import peewee

from afang.database.trades_db.trades_database import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import TradesDatabase
from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Order as ExchangeOrder
from afang.exchanges.models import OrderType
from afang.models import Timeframe

logger = logging.getLogger(__name__)


class Root:
    """Interface used to define class variables for both the Trader and
    Backtester classes."""

    def __init__(self, strategy_name: str):
        """Initialize Root class.

        :param strategy_name: name of the trading strategy.
        """

        # --Shared variables (used by both the Trader and Backtester)
        self.strategy_name: str = strategy_name
        self.allow_long_positions: bool = True
        self.allow_short_positions: bool = True
        self.timeframe: Optional[Timeframe] = None
        self.symbols: Optional[List[str]] = None
        self.exchange: Optional[IsExchange] = None
        self.is_running_backtest: bool = True
        # leverage to use per trade.
        self.leverage: int = 1
        # exchange order fee as a percentage of the trade principal.
        self.commission: float = 0.05
        # number of indicator values to be discarded due to being potentially unstable.
        self.unstable_indicator_values: int = 0
        # percentage of current account balance to risk per trade.
        self.percentage_risk_per_trade: float = 2
        # maximum amount to invest per trade.
        # If `None`, there will be no maximum amount to invest per trade.
        self.max_amount_per_trade: Optional[int] = None
        # Sets how many trade positions can be open at any given time.
        self.max_open_positions: int = 1
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config: Dict = dict()
        # test account initial balance - will be constantly updated to match current account balance.
        self.initial_test_account_balance: float = 10000
        # shared threading lock to prevent race conditions.
        self.shared_lock: threading.Lock = threading.Lock()
        # Order type to be used to open positions.
        self.open_order_type: Literal[
            OrderType.LIMIT, OrderType.MARKET
        ] = OrderType.MARKET
        # trades' database instance.
        self.trades_database: Optional[TradesDatabase] = None
        # Order type to be used to place take profit orders.
        self.take_profit_order_type: Literal[
            OrderType.LIMIT, OrderType.MARKET
        ] = OrderType.MARKET
        # Order type to be used to place stop loss orders.
        self.stop_loss_order_type: Literal[OrderType.MARKET] = OrderType.MARKET

        # --Unique to Backtester (not in Trader)
        self.backtest_to_time: Optional[int] = None
        self.backtest_from_time: Optional[int] = None
        self.open_symbol_positions: Dict[str, List[DBTradePosition]] = defaultdict(list)
        # expected trade slippage as a percentage of the trade principal.
        self.expected_slippage: float = 0.05
        # backtest data that initially contains OHLCV data.
        self.backtest_data: Dict = dict()

        # --Unique to Trader (not in Backtester)
        self.on_demo_mode: Optional[bool] = None
        # exchange orders that are placed while on demo mode.
        self.demo_mode_exchange_orders: Dict[str, List[ExchangeOrder]] = defaultdict(
            list
        )
        # all symbols that have successfully gone through the trading loop within a specified TTL.
        self.actively_trading_symbols: set = set()
        # last time that all actively traded symbols was reported.
        self.last_report_time: datetime = datetime.utcnow()
        # execution queue that will run trader on present symbols FIFO.
        self.trading_execution_queue: queue.Queue = queue.Queue()
        # Orders will only be placed if they enter the order book. May not be supported by all exchanges.
        self.post_only_orders: bool = False
        # Highest accepted fee for a trade on the dYdX exchange. Note that this is specific to dYdX.
        self.dydx_limit_fee: Optional[float] = None

    def fetch_open_trade_positions(
        self, symbols: Optional[List[str]] = None
    ) -> List[DBTradePosition]:
        """Fetch a list of all open trade positions for a list of symbols.

        :param symbols: symbols to fetch open positions for. optional.
        :return: List[DBTradePosition]
        """

        try:
            open_positions: List[DBTradePosition] = DBTradePosition.select().where(
                DBTradePosition.is_open.__eq__(True),
                DBTradePosition.symbol.in_(symbols),
                DBTradePosition.exchange_display_name == self.exchange.display_name,
            )
        except peewee.PeeweeException as db_error:
            logger.error("Could not fetch open trade positions: %s", db_error)
            return []

        return open_positions

    def on_trade_position_opened(self, position: DBTradePosition) -> None:
        """Hook that is called when a trade position is opened.

        :param position: newly opened trade position.
        :return: None
        """

        pass

    def on_trade_position_closed(self, position: DBTradePosition) -> None:
        """Hook that is called when a trade position is closed.

        :param position: closed trade position.
        :return: None
        """

        pass

    def handle_open_trade_positions(
        self,
        symbol: str,
        open_trade_positions: List[DBTradePosition],
        current_trading_candle: Any,
    ) -> None:
        """Hook to be used by user strategies to manipulate open trade
        positions.

        :param symbol: trading symbol.
        :param open_trade_positions: current open trade positions.
        :param current_trading_candle: current trading candle.
        :return: None
        """

        pass
