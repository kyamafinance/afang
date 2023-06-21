import logging
import pathlib
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import peewee
import yaml

from afang.database.trades_db.models import Order as DBOrder
from afang.database.trades_db.models import TradePosition as DBTradePosition
from afang.exchanges.models import OrderType
from afang.strategies.backtester import Backtester
from afang.strategies.models import TradeLevels
from afang.strategies.trader import Trader
from afang.utils.util import generate_uuid

logger = logging.getLogger(__name__)


class IsStrategy(Backtester, Trader):
    """Base interface for a user supported strategy."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize IsStrategy class.

        :param strategy_name: name of the trading strategy.
        """

        Backtester.__init__(self, strategy_name)
        Trader.__init__(self, strategy_name)

        self.strategy_name: str = strategy_name
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config: Dict = self.read_strategy_config()

    def read_strategy_config(self) -> Dict:
        """Read and return the strategy config.

        :return: Dict
        """

        config_file_path = (
            f"{pathlib.Path(__file__).parent}/{self.strategy_name}/config.yaml"
        )
        with open(config_file_path) as config_file:
            config_data = yaml.load(config_file, Loader=yaml.FullLoader)
            return config_data

    def get_watchlist(self, exchange: str) -> List[str]:
        """Get exchange specific strategy watchlist.

        :param exchange: name of exchange to fetch watchlist for.
        :return: List[str]
        """

        watchlist = self.config.get("watchlist", dict())
        if not watchlist:
            return []

        return watchlist.get(exchange, [])

    @abstractmethod
    def generate_features(self, symbol: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the trading strategy.

        - To generate features, add columns to the `data` dataframe that can later
          be used to calculate horizontal trade barriers.
        - Initially, the `data` dataframe will contain OHLCV data.

        :param symbol: trading symbol.
        :param ohlcv_df: OHLCV data for a trading symbol.
        :return: None
        """

        return ohlcv_df

    @abstractmethod
    def is_long_trade_signal_present(
        self, symbol: str, current_trading_candle: Any
    ) -> bool:
        """Check if a long trade signal exists.

        :param symbol: trading symbol.
        :param current_trading_candle: the current trading candle.
        :return: bool
        """

        pass

    @abstractmethod
    def is_short_trade_signal_present(
        self, symbol: str, current_trading_candle: Any
    ) -> bool:
        """Check if a short trade signal exists.

        :param symbol: trading symbol.
        :param current_trading_candle: the current trading candle.
        :return: bool
        """

        pass

    @abstractmethod
    def generate_trade_levels(
        self, symbol: str, current_trading_candle: Any, trade_signal_direction: int
    ) -> TradeLevels:
        """Generate price levels for an individual trade signal.

        :param symbol: trading symbol.
        :param current_trading_candle: the current trading candle.
        :param trade_signal_direction: 1 for a long position. -1 for a
            short position.
        :return: TradeLevels
        """

        return TradeLevels(
            entry_price=current_trading_candle.close,
            target_price=None,
            stop_price=None,
        )

    def define_optimization_param_constraints(self, parameters: Dict) -> Dict:
        """Define constraints that should be applied during backtest parameter
        generation while optimizing the strategy. Should return a dict that
        contains possible mutated parameters.

        :param parameters: parameters generated for strategy
            optimization. These parameters will follow the
            specification provided in `config.yaml`. This dict will
            not contain parameters that are not to be
            optimized.
        :return: Dict
        """

        return parameters

    def open_trade_position(
        self,
        symbol: str,
        direction: int,
        trade_levels: TradeLevels,
        current_trading_candle: Any,
    ) -> Optional[DBTradePosition]:
        """Open a new trade position.

        :param symbol: trading symbol.
        :param direction: trading direction. 1 for LONG. -1 for SHORT.
        :param trade_levels: new trade position price barriers.
        :param current_trading_candle: current price candle.
        :return: Optional[DBTradePosition]
        """

        if self.is_running_backtest:
            return self.open_backtest_position(
                symbol,
                direction,
                trade_levels,
                current_trading_candle.Index.to_pydatetime(),
            )

        return self.trader__open_trade_position(symbol, direction, trade_levels)

    def place_close_trade_position_order(
        self,
        position_id: int,
        close_price: float,
        current_trading_candle: Any,
        close_order_type: OrderType = OrderType.MARKET,
    ) -> None:
        """Attempt to close an existing trade position.

        :param position_id: ID of DB position to place a close trade
            order for.
        :param close_price: price at which the trade should be closed
            at.
        :param current_trading_candle: current price candle.
        :param close_order_type: order type to use for the close
            position order.
        :return: None
        """

        try:
            position = DBTradePosition.get_by_id(position_id)
        except (DBTradePosition.DoesNotExist, peewee.PeeweeException) as db_error:
            logger.error(
                "Close position order cannot be placed. Position %s not found: %s",
                position_id,
                db_error,
            )
            return

        if self.is_running_backtest:
            self.close_backtest_position(
                position,
                close_price,
                current_trading_candle.Index.to_pydatetime(),
                close_order_type,
            )
            return

        self.trader__place_close_trade_position_order(
            position, close_price, close_order_type
        )

    def update_position_trade_levels(
        self, position_id: int, updated_trade_levels: TradeLevels
    ) -> Optional[DBTradePosition]:
        """Update an open position's trade levels.

        NOTE: The entry price of a trade position can not be updated. It will be ignored if updated.

        :param position_id: ID of open trade position to be updated.
        :param updated_trade_levels: updated trade position trade levels.
        :return: Optional[DBTradePosition]
        """

        try:
            position = DBTradePosition.get_by_id(position_id)
        except (DBTradePosition.DoesNotExist, peewee.PeeweeException) as db_error:
            logger.error(
                "Trade position trade levels could not be updated. ID: %s: %s",
                position_id,
                db_error,
            )
            return None

        if not position.is_open:
            logger.error(
                "%s: trade position trade levels could not be updated. "
                "trade position is not open. ID: %s",
                position.symbol,
                position.id,
            )
            return None

        # Cancel all open close position orders.
        if not self.is_running_backtest:
            position_db_orders: List[DBOrder] = position.orders
            for db_order in position_db_orders:
                if (not db_order.is_open) or db_order.is_open_order:
                    continue
                self.cancel_position_order(db_order.symbol, db_order.order_id)

        # Update trade levels.
        position.target_price = updated_trade_levels.target_price
        position.stop_price = updated_trade_levels.stop_price
        position.sequence_id = updated_trade_levels.sequence_id or generate_uuid()
        position.save()

        return position
