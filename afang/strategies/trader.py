import logging
import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from afang.database.trades_db.trades_database import Order as DBOrder
from afang.database.trades_db.trades_database import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import TradesDatabase
from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Order, OrderSide, OrderType, Symbol, SymbolBalance
from afang.models import Timeframe
from afang.strategies.models import TradeLevels
from afang.utils.util import round_float_to_precision

logger = logging.getLogger(__name__)


class Trader(ABC):
    """Base interface for strategy live/demo trading."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize Trader class.

        :param strategy_name: name of the trading strategy.
        """

        self.strategy_name: str = strategy_name
        self.allow_long_positions: bool = True
        self.allow_short_positions: bool = True
        self.timeframe: Optional[Timeframe] = None
        self.symbols: Optional[List[str]] = None
        self.exchange: Optional[IsExchange] = None
        # leverage to use per trade.
        self.leverage: int = 1
        # exchange order fee as a percentage of the trade principal to be used in demo mode.
        self.commission: float = 0.05
        # expected trade slippage as a percentage of the trade principal to be used in demo mode.
        self.expected_slippage: float = 0.05
        # number of indicator values to be discarded due to being potentially unstable.
        self.unstable_indicator_values: int = 0
        # maximum number of candles for a single trade.
        self.max_holding_candles: int = 100
        # percentage of current account balance to risk per trade.
        self.percentage_risk_per_trade: float = 2
        # maximum amount to invest per trade.
        # If `None`, the maximum amount to invest per trade will be the current account balance.
        self.max_amount_per_trade: Optional[int] = None
        # Whether to allow for multiple open positions per symbol at a time.
        self.allow_multiple_open_positions: bool = True
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config: Dict = dict()

        # --Unique to Trader (not in Backtester)
        self.on_demo_mode: Optional[bool] = None
        # execution queue that will run trader on present symbols FIFO.
        self.trading_execution_queue: queue.Queue = queue.Queue()
        # Order type to be used to open positions.
        self.open_order_type: OrderType = OrderType.MARKET
        # Order type to be used to close positions.
        self.close_order_type: OrderType = OrderType.MARKET
        # Orders will only be placed if they enter the order book. May not be supported by all exchanges.
        self.post_only_orders: bool = False
        # Highest accepted fee for a trade on the dYdX exchange. Note that this is specific to dYdX.
        self.dydx_limit_fee: Optional[float] = None

    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the trading strategy.

        - To generate features, add columns to the `data` dataframe that can later
          be used to calculate horizontal trade barriers.
        - Initially, the `data` dataframe will contain OHLCV data.

        :param data: OHLCV data for a trading symbol.
        :return: None
        """

        return data

    @abstractmethod
    def is_long_trade_signal_present(self, data: Any) -> bool:
        """Check if a long trade signal exists.

        :param data: the current trading candle.
        :return: bool
        """

        pass

    @abstractmethod
    def is_short_trade_signal_present(self, data: Any) -> bool:
        """Check if a short trade signal exists.

        :param data: the current trading candle.
        :return: bool
        """

        pass

    @abstractmethod
    def generate_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> TradeLevels:
        """Generate price levels for an individual trade signal.

        :param data: the current trading candle.
        :param trade_signal_direction: 1 for a long position. -1 for a short position.
        :return: TradeLevels
        """

        return TradeLevels(
            entry_price=data.close,
            target_price=None,
            stop_price=None,
        )

    @classmethod
    def fetch_symbol_open_trade_positions(
        cls, symbol: str, trades_database: TradesDatabase
    ) -> List[DBTradePosition]:
        """Fetch a list of all open trade positions for a given symbol.

        :param symbol: symbol to fetch open positions for.
        :param trades_database: TradesDatabase instance.
        :return: List[DBTradePosition]
        """

        filters = (
            DBTradePosition.symbol == symbol,
            DBTradePosition.open_position is True,
        )

        open_positions: List[DBTradePosition] = trades_database.fetch_positions(
            filters=filters
        )

        return open_positions

    def get_next_trading_symbol(self) -> Optional[str]:
        """Get the next trading symbol from the execution queue to trade.

        :return: Optional[str]
        """

        while True:
            if not self.trading_execution_queue.empty():
                return self.trading_execution_queue.get()
            time.sleep(0.5)

    def get_trading_symbol(self, symbol: str) -> Optional[Symbol]:
        """Get exchange trading symbol info.

        :param symbol: name of symbol.
        :return: Optional[Symbol]
        """

        trading_symbol = self.exchange.trading_symbols.get(symbol, None)
        if not trading_symbol:
            logger.error(
                "%s %s: symbol not found in exchange trading symbols",
                self.exchange.display_name,
                symbol,
            )
            return None

        return trading_symbol

    def get_quote_asset_balance(self, symbol: str) -> Optional[SymbolBalance]:
        """Get the quote asset balance of a given symbol.

        :param symbol: trading symbol.
        :return: Optional[SymbolBalance]
        """

        trading_symbol = self.get_trading_symbol(symbol)
        if not trading_symbol:
            return None

        symbol_quote_asset = trading_symbol.quote_asset
        quote_asset_balance = self.exchange.trading_symbol_balance.get(
            symbol_quote_asset, None
        )
        if not quote_asset_balance:
            logger.error(
                "%s %s: quote asset %s not found in exchange trading symbol balances",
                self.exchange.display_name,
                symbol,
                symbol_quote_asset,
            )
            return None

        return quote_asset_balance

    def get_open_order_position_size(self, quote_asset_balance: SymbolBalance) -> float:
        """Get the intended position size for a position open order.

        :param quote_asset_balance: quote asset balance.
        :return: float
        """

        intended_position_size = self.leverage * (
            (self.percentage_risk_per_trade / 100.0)
            * quote_asset_balance.wallet_balance
        )
        if (
            self.max_amount_per_trade
            and intended_position_size > self.max_amount_per_trade
        ):
            intended_position_size = self.max_amount_per_trade

        return intended_position_size

    def get_close_order_qty(self, position: DBTradePosition) -> float:
        """Get the appropriate close order quantity for a position.

        :param position: DB TradePosition.
        :return: float
        """

        position_open_order = self.exchange.active_orders.get(
            position.open_order_id, None
        )
        close_order_quantity = position_open_order.executed_quantity
        position_last_close_order = self.exchange.active_orders.get(
            position.final_close_order_id, None
        )
        if position_last_close_order:
            close_order_quantity = position_last_close_order.remaining_quantity

        return close_order_quantity

    def is_order_qty_valid(self, trading_symbol: Symbol, order_qty: float) -> bool:
        """Validate whether a given order quantity is valid for a given symbol.

        :param trading_symbol: trading symbol.
        :param order_qty: desired order quantity.
        :return: float
        """

        if order_qty <= 0:
            logger.error(
                "%s %s: intended order qty is invalid. intended order qty: %s",
                self.exchange.display_name,
                trading_symbol.name,
                order_qty,
            )
            return False

        return True

    def is_order_price_valid(self, trading_symbol: Symbol, order_price: float) -> bool:
        """Validate whether a given order price is valid for a given symbol.

        :param trading_symbol: trading symbol.
        :param order_price: desired order price.
        :return: float
        """

        if order_price <= 0:
            logger.error(
                "%s %s: intended order price is invalid. intended order price: %s",
                self.exchange.display_name,
                trading_symbol.name,
                order_price,
            )
            return False

        return True

    def create_new_db_position(
        self, db_position: DBTradePosition, trades_database: TradesDatabase
    ) -> None:
        """Add a new database trade position to the DB.

        :param db_position: database trade position to be added to the DB.
        :param trades_database: trades database instance.
        :return: None
        """

        try:
            trades_database.create_new_position(db_position)
            trades_database.session.commit()
        except Exception as e:
            logger.error(
                "%s %s: failed to record new trade position to the DB. position id: %s: %s",
                self.exchange.display_name,
                db_position.symbol,
                db_position.id,
                str(e),
            )

    def create_new_db_order(
        self, db_order: DBOrder, trades_database: TradesDatabase
    ) -> None:
        """Add a new database order to the DB.

        :param db_order: database order to be added to the DB.
        :param trades_database: trades database instance.
        :return: None
        """

        try:
            trades_database.create_new_order(db_order)
            trades_database.session.commit()
        except Exception as e:
            logger.error(
                "%s %s: failed to record new order to the DB. order id: %s: %s",
                self.exchange.display_name,
                db_order.symbol,
                db_order.order_id,
                str(e),
            )

    def update_db_order(
        self,
        symbol: str,
        db_order_id: int,
        updated_order: Dict,
        trades_database: TradesDatabase,
    ) -> None:
        """Update a DB Order.

        :param symbol: DB TradePosition symbol.
        :param db_order_id: ID of the DB Order - this is the DB ID and not the exchange ID.
        :param updated_order: updated order dict.
        :param trades_database: trades database instance.
        :return: None
        """

        try:
            trades_database.update_order(db_order_id, updated_order)
            trades_database.session.commit()
        except Exception as e:
            logger.error(
                "%s %s: failed to update DB order. order. id: %s: %s",
                self.exchange.display_name,
                symbol,
                db_order_id,
                str(e),
            )

    def update_db_trade_position(
        self,
        symbol: str,
        db_position_id: int,
        updated_trade_position: Dict,
        trades_database: TradesDatabase,
    ) -> None:
        """Update a DB TradePosition.

        :param symbol: DB TradePosition symbol.
        :param db_position_id: ID of TradePosition to update.
        :param updated_trade_position: updated trade position dict.
        :param trades_database: trades database instance.
        :return: None
        """

        try:
            trades_database.update_position(db_position_id, updated_trade_position)
            trades_database.session.commit()
        except Exception as e:
            logger.error(
                "%s %s: failed to update DB trade position. pos. id: %s: %s",
                self.exchange.display_name,
                symbol,
                db_position_id,
                str(e),
            )

    def update_closed_order_in_db(
        self, order: Order, db_order: DBOrder, trades_database: TradesDatabase
    ) -> None:
        """Update a closed order's details in the DB.

        :param order: exchange order instance.
        :param db_order: database order instance.
        :param trades_database: TradesDatabase instance.
        :return: None
        """

        updated_order: Dict[str, Any] = dict()
        updated_order["time_in_force"] = order.time_in_force
        updated_order["average_price"] = order.average_price
        updated_order["executed_quantity"] = order.executed_quantity
        updated_order["remaining_quantity"] = order.remaining_quantity
        updated_order["order_status"] = order.order_status
        updated_order["commission"] = order.commission

        self.update_db_order(order.symbol, db_order.id, updated_order, trades_database)

    def cancel_position_open_order(
        self, position: DBTradePosition, trades_database: TradesDatabase
    ) -> None:
        """Cancel a position open order if some/all of its quantity is un-
        executed and update order details in DB.

        :param position: position whose open order is to be canceled.
        :param trades_database: TradesDatabase instance.
        :return: None
        """

        position_open_order = self.exchange.active_orders.get(
            position.open_order_id, None
        )
        if not position_open_order:
            logger.error(
                "%s %s: position open order not canceled because open order was not found on the exchange. ,"
                "position id: %s. open order id: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
                position.open_order_id,
            )
            return None

        if position_open_order.remaining_quantity:
            logger.info(
                "%s %s: attempting to cancel position open order due to partly un-executed qty. "
                "position id: %s. open order id: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
                position.open_order_id,
            )
            self.exchange.cancel_order(position.symbol, order_id=position.open_order_id)

        filters = (
            DBOrder.is_open_order is True,
            DBOrder.order_id == position.open_order_id,
        )
        db_position_open_order = trades_database.fetch_orders(filters, limit=1)
        if not db_position_open_order:
            logger.error(
                "%s %s: position open order not updated because it was not found in the DB. "
                "open order id: %s",
                position.open_order_id,
            )
            return None
        self.update_closed_order_in_db(
            position_open_order, db_position_open_order[0], trades_database
        )

    def cancel_position_last_close_order(
        self, position: DBTradePosition, trades_database: TradesDatabase
    ) -> None:
        """Cancel a position's last close order if one exists and some/all of
        its quantity is un-executed.

        :param position: position whose last close order is to be canceled.
        :param trades_database: TradesDatabase instance.
        :return: None
        """

        position_last_close_order = self.exchange.active_orders.get(
            position.final_close_order_id, None
        )
        if not position_last_close_order:
            return None

        if position_last_close_order.remaining_quantity:
            logger.info(
                "%s %s: attempting to cancel position last close order due to partly un-executed qty. "
                "position id: %s. last close order id: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
                position.final_close_order_id,
            )
            self.exchange.cancel_order(
                position.symbol, order_id=position.final_close_order_id
            )

        filters = (
            DBOrder.is_open_order is False,
            DBOrder.order_id == position.final_close_order_id,
        )
        db_position_last_close_order = trades_database.fetch_orders(filters, limit=1)
        if not db_position_last_close_order:
            logger.error(
                "%s %s: position last close order not updated because it was not found in the DB. "
                "order id: %s",
                position.final_close_order_id,
            )
            return None
        self.update_closed_order_in_db(
            position_last_close_order, db_position_last_close_order[0], trades_database
        )

    def is_order_filled(self, order_id: str) -> bool:
        """
        Returns a bool whether an order has been filled or not.
        NOTE: Even a partial order fill will return true.

        :param order_id: ID of the order to query.
        :return: bool
        """

        order = self.exchange.active_orders.get(order_id, None)
        if order and order.executed_quantity:
            return True

        return False

    def get_symbol_ohlcv_candles_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get a symbol's OHLCV candles as a dataframe.

        :param symbol: symbol whose OHLCV candles are to be fetched.
        :return: Optional[pd.Dataframe]
        """

        if (
            symbol not in self.exchange.trading_price_data
            or not self.exchange.trading_price_data[symbol]
        ):
            logger.error(
                "%s %s: %s %s price data unavailable",
                self.exchange.display_name,
                self.strategy_name,
                symbol,
                self.timeframe.value,
            )
            return None

        ohlcv_candles = self.exchange.trading_price_data[symbol]
        ohlcv_candles_df = pd.DataFrame(ohlcv_candles)
        ohlcv_candles_df.open_time = pd.to_datetime(
            ohlcv_candles_df.open_time.values.astype(np.int64), unit="ms"
        )
        ohlcv_candles_df.set_index("open_time", drop=True, inplace=True)

        return ohlcv_candles_df

    def get_symbol_current_trading_candle(
        self, symbol: str, ohlcv_data: pd.DataFrame
    ) -> Any:
        """Get a symbol's current trading candle from a dataframe that contains
        OHLCV data.

        :param symbol: symbol whose current trading candle is being fetched.
        :param ohlcv_data: dataframe that contains a symbol's OHLCV data.
        :return: Any
        """

        current_candle_data_list = list(ohlcv_data.iloc[-1:].itertuples())
        if not current_candle_data_list:
            logger.error(
                "%s %s: could not fetch current candle data for %s %s",
                self.exchange.display_name,
                self.strategy_name,
                symbol,
                self.timeframe.value,
            )
        current_candle_data = current_candle_data_list[0]

        return current_candle_data

    def open_trade_position(
        self,
        symbol: str,
        direction: int,
        desired_entry_price: float,
        trades_database: TradesDatabase,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[DBTradePosition]:
        """Open a LONG or SHORT trade position for a given symbol.

        :param symbol: symbol to open a trade position for.
        :param direction: whether to open a LONG/SHORT trade position. 1 for LONG. -1 for SHORT.
        :param desired_entry_price: desired price to enter the long trade.
        :param trades_database: TradesDatabase instance.
        :param target_price: price at which the long trade should take profit.
        :param stop_price: price at which the long trade should cut losses.
        :return: None
        """

        logger.info(
            "%s %s: attempting to open a new trade position. entry price: %s. target price: %s. stop price: %s",
            self.exchange.display_name,
            symbol,
            round(desired_entry_price, 4),
            round(target_price, 4),
            round(stop_price, 4),
        )

        trading_symbol = self.get_trading_symbol(symbol)
        if not trading_symbol:
            return None

        quote_asset_balance = self.get_quote_asset_balance(symbol)
        if not quote_asset_balance:
            return None

        intended_position_size = self.get_open_order_position_size(quote_asset_balance)
        intended_open_order_qty = intended_position_size / desired_entry_price
        precise_open_order_qty = round_float_to_precision(
            intended_open_order_qty, trading_symbol.step_size
        )
        if not self.is_order_qty_valid(trading_symbol, precise_open_order_qty):
            return None
        precise_open_order_entry_price = round_float_to_precision(
            desired_entry_price, trading_symbol.tick_size
        )
        if not self.is_order_price_valid(
            trading_symbol, precise_open_order_entry_price
        ):
            return None
        open_order_side = OrderSide.BUY if direction == 1 else OrderSide.SELL

        # Attempt to place an open position order on the exchange.
        open_order_id = self.exchange.place_order(
            symbol_name=symbol,
            side=open_order_side,
            quantity=precise_open_order_qty,
            order_type=self.open_order_type,
            price=precise_open_order_entry_price,
            post_only=self.post_only_orders,
            dydx_limit_fee=self.dydx_limit_fee,
        )
        if not open_order_id:
            logger.error(
                "%s %s: failed to place open order on the exchange. price: %s. qty: %s",
                self.exchange.display_name,
                symbol,
                precise_open_order_entry_price,
                precise_open_order_qty,
            )
            return None

        logger.info(
            "%s %s: open order successfully placed on the exchange. order id: %s",
            self.exchange.display_name,
            symbol,
            open_order_id,
        )

        new_trade_position = DBTradePosition(
            symbol=symbol,
            direction=direction,
            desired_entry_price=desired_entry_price,
            open_order_id=open_order_id,
            position_qty=intended_open_order_qty,
            position_size=intended_position_size,
            target_price=target_price,
            stop_price=stop_price,
            initial_account_balance=quote_asset_balance.wallet_balance,
        )
        self.create_new_db_position(new_trade_position, trades_database)

        position_open_order = DBOrder(
            symbol=symbol,
            is_open_order=True,
            order_id=open_order_id,
            order_side=open_order_side.value,
            original_price=desired_entry_price,
            original_quantity=intended_open_order_qty,
            order_type=self.open_order_type.value,
            position=new_trade_position,
        )
        self.create_new_db_order(position_open_order, trades_database)

        return new_trade_position

    def place_close_trade_position_order(
        self,
        symbol: str,
        position_id: int,
        direction: int,
        close_price: float,
        trades_database: TradesDatabase,
    ) -> None:
        """Place a close trade position order for a given symbol.

        :param symbol: symbol to close a trade position for.
        :param position_id: ID of the position to close.
        :param direction: whether the trade is a LONG/SHORT trade position. 1 for LONG. -1 for SHORT.
        :param close_price: price at which the trade should be closed at.
        :param trades_database: TradesDatabase instance.
        :return: None
        """

        logger.info(
            "%s %s: attempting to place a close trade position order. position id: %s. close price: %s",
            self.exchange.display_name,
            symbol,
            position_id,
            round(close_price, 4),
        )

        # Fetch trading symbol info.
        trading_symbol = self.get_trading_symbol(symbol)
        if not trading_symbol:
            return None

        # Fetch the position to close from the DB.
        position = trades_database.fetch_position_by_id(position_id)
        if not position:
            return None

        # Ensure that a similar close position order is not already present for the position.
        if (
            position.final_close_order_id
            and position.final_desired_close_price == close_price
        ):
            return None

        # Ensure that the trade position is still open.
        if not position.open_position:
            logger.error(
                "%s %s: trade position is already closed. position id: %s",
                self.exchange.display_name,
                symbol,
                position_id,
            )
            return None

        # Check if the trade position open order has been filled - even partially.
        if not self.is_order_filled(position.open_order_id):
            logger.error(
                "%s %s: trade position open order is not yet filled. position id: %s",
                self.exchange.display_name,
                symbol,
                position_id,
            )
            return None

        close_order_quantity = self.get_close_order_qty(position)
        precise_close_order_qty = round_float_to_precision(
            close_order_quantity, trading_symbol.step_size
        )
        if not self.is_order_qty_valid(trading_symbol, precise_close_order_qty):
            return None
        precise_close_order_price = round_float_to_precision(
            close_price, trading_symbol.tick_size
        )
        if not self.is_order_price_valid(trading_symbol, precise_close_order_price):
            return None
        close_order_side = OrderSide.SELL if direction == 1 else OrderSide.BUY

        # Attempt to place a close position order on the exchange.
        close_order_id = self.exchange.place_order(
            symbol_name=symbol,
            side=close_order_side,
            quantity=precise_close_order_qty,
            order_type=self.close_order_type,
            price=precise_close_order_price,
            post_only=self.post_only_orders,
            dydx_limit_fee=self.dydx_limit_fee,
        )
        if not close_order_id:
            logger.error(
                "%s %s: failed to place close order on the exchange. price: %s. qty: %s",
                self.exchange.display_name,
                symbol,
                precise_close_order_price,
                precise_close_order_qty,
            )
            return None

        logger.info(
            "%s %s: close order successfully placed on the exchange. position id: %s. order id: %s",
            self.exchange.display_name,
            symbol,
            position_id,
            close_order_id,
        )

        self.cancel_position_open_order(position, trades_database)
        self.cancel_position_last_close_order(position, trades_database)

        new_close_order = DBOrder(
            symbol=symbol,
            is_open_order=False,
            order_id=close_order_id,
            order_side=close_order_side.value,
            original_price=close_price,
            original_quantity=close_order_quantity,
            order_type=self.close_order_type.value,
            position=position,
        )
        self.create_new_db_order(new_close_order, trades_database)

        updated_trade_position: Dict[str, Any] = dict()
        updated_trade_position["final_close_order_id"] = close_order_id
        updated_trade_position["final_desired_close_price"] = close_price
        self.update_db_trade_position(
            symbol, position_id, updated_trade_position, trades_database
        )

    def handle_open_trade_positions(
        self,
        symbol: str,
        current_candle_data: Any,
        open_symbol_positions: List[DBTradePosition],
        trades_database: TradesDatabase,
    ) -> None:
        """Monitor and handle open positions for a given symbol and place close
        trade position orders if they hit a trade barrier.

        :param symbol: symbol to monitor open positions for.
        :param current_candle_data: the current trading candle data.
        :param open_symbol_positions: open positions for the given symbol.
        :param trades_database: TradesDatabase instance.
        :return: None
        """

        for position in open_symbol_positions:
            should_close_position = False
            desired_close_price = None

            # ensure that the trade position is still open.
            if not position.open_position:
                continue

            # check if the trade position open order has been filled - even partially.
            is_open_order_filled = self.is_order_filled(position.open_order_id)

            # check if upper horizontal barrier has been hit for long positions.
            if (
                is_open_order_filled
                and position.target_price
                and position.direction == 1
                and (
                    (
                        self.close_order_type == OrderType.MARKET
                        and current_candle_data.high >= position.target_price
                    )
                    or (self.close_order_type == OrderType.LIMIT)
                )
            ):
                should_close_position = True
                desired_close_price = position.target_price

            # check if the lower horizontal barrier has been hit for long positions.
            elif (
                is_open_order_filled
                and position.stop_price
                and position.direction == 1
                and (
                    (
                        self.close_order_type == OrderType.MARKET
                        and current_candle_data.low <= position.stop_price
                    )
                    or (self.close_order_type == OrderType.LIMIT)
                )
            ):
                should_close_position = True
                desired_close_price = position.stop_price

            # check if lower horizontal barrier has been hit for short positions.
            elif (
                is_open_order_filled
                and position.target_price
                and position.direction == -1
                and (
                    (
                        self.close_order_type == OrderType.MARKET
                        and current_candle_data.low <= position.target_price
                    )
                    or (self.close_order_type == OrderType.LIMIT)
                )
            ):
                should_close_position = True
                desired_close_price = position.target_price

            # check if upper horizontal barrier has been hit for short positions.
            elif (
                is_open_order_filled
                and position.stop_price
                and position.direction == -1
                and (
                    (
                        self.close_order_type == OrderType.MARKET
                        and current_candle_data.high >= position.stop_price
                    )
                    or (self.close_order_type == OrderType.LIMIT)
                )
            ):
                should_close_position = True
                desired_close_price = position.stop_price

            # check if vertical barrier has been hit.
            # TODO: If the position has not yet been filled - even partially - but
            #  has been open for more than *holding time*, close it.

            if should_close_position:
                # Ensure that a similar close position order is not already present for the position.
                if (
                    position.final_close_order_id
                    and position.final_desired_close_price == desired_close_price
                ):
                    continue

                self.place_close_trade_position_order(
                    symbol,
                    position.id,
                    position.direction,
                    desired_close_price,
                    trades_database,
                )

    def run_symbol_trader(self, symbol: str) -> None:
        """Run trader on a single symbol.

        :param symbol: symbol to trade.
        :return: None
        """

        # ensure symbol is present in the exchange.
        if symbol not in self.exchange.exchange_symbols:
            logger.error(
                "%s %s: provided symbol not present in the exchange",
                self.exchange.display_name,
                symbol,
            )
            return None

        ohlcv_candles_df = self.get_symbol_ohlcv_candles_df(symbol)
        if not ohlcv_candles_df:
            return None

        populated_ohlcv_data = self.generate_features(ohlcv_candles_df)
        current_candle_data = self.get_symbol_current_trading_candle(
            symbol, populated_ohlcv_data
        )

        trades_database = TradesDatabase()
        symbol_open_positions = self.fetch_symbol_open_trade_positions(
            symbol, trades_database
        )

        should_open_new_position = False
        new_position_direction = None

        # open a long position if we get a long trading signal.
        if self.allow_long_positions and self.is_long_trade_signal_present(
            current_candle_data
        ):
            should_open_new_position = True
            new_position_direction = 1

        # open a short position if we get a short trading signal.
        elif self.allow_short_positions and self.is_short_trade_signal_present(
            current_candle_data
        ):
            should_open_new_position = True
            new_position_direction = -1

        if should_open_new_position:
            # only open a position if multiple open positions are allowed or
            # there is no open position.
            if self.allow_multiple_open_positions or not len(symbol_open_positions):
                trade_levels = self.generate_trade_levels(
                    current_candle_data, trade_signal_direction=new_position_direction
                )
                new_trade_position = self.open_trade_position(
                    symbol=symbol,
                    direction=new_position_direction,
                    desired_entry_price=trade_levels.entry_price,
                    trades_database=trades_database,
                    target_price=trade_levels.target_price,
                    stop_price=trade_levels.stop_price,
                )
                if new_trade_position:
                    symbol_open_positions.append(new_trade_position)

        # monitor and handle all open symbol positions.
        self.handle_open_trade_positions(
            symbol, current_candle_data, symbol_open_positions, trades_database
        )

        # close the trades' database connection.
        trades_database.session.close()

        # add the symbol back to the trading execution queue.
        time.sleep(2.0)
        self.trading_execution_queue.put(symbol)

    def run_trader(
        self,
        exchange: IsExchange,
        symbols: Optional[List[str]],
        timeframe: Optional[str],
        demo_mode: Optional[bool] = True,
    ) -> None:
        """Trade multiple symbols either on demo or live mode.

        :param exchange: exchange to use to run the trader.
        :param symbols: exchange symbols to run trader on.
        :param timeframe: timeframe to run the trader on.
        :param demo_mode: whether to run the trader on demo mode.
        :return: None
        """

        # Get symbols to trade.
        self.symbols = symbols
        if not self.symbols:
            self.symbols = self.config.get("watchlist", dict()).get(exchange.name, [])
        if not self.symbols:
            logger.warning(
                "%s: no symbols found to run strategy trader", self.strategy_name
            )
            return None

        # Record exchange to be used for trading.
        self.exchange = exchange
        # Record if the trader is on demo mode.
        self.on_demo_mode = demo_mode

        # Get the trading timeframe.
        if not timeframe:
            timeframe = self.config.get("timeframe", None)
        if not timeframe:
            logger.warning(
                "%s: timeframe not defined for the strategy trader",
                self.strategy_name,
            )
            return None
        # Ensure the trading timeframe is supported.
        try:
            self.timeframe = Timeframe(timeframe)
        except ValueError:
            logger.warning(
                "%s: invalid timeframe %s defined for the strategy trader",
                self.strategy_name,
                timeframe,
            )
            return None

        # Update the strategy config with the working parameters.
        self.config.update(
            {
                "timeframe": self.timeframe.value,
                "exchange": self.exchange,
            }
        )

        # Initialize trading execution queue.
        for symbol in self.symbols:
            logger.info(
                "%s: running %s on %s %s",
                self.exchange.display_name,
                self.strategy_name,
                symbol,
                self.timeframe.value,
            )
            self.trading_execution_queue.put(symbol)

        # Setup exchange for trading.
        self.exchange.setup_exchange_for_trading(self.symbols, self.timeframe)
        self.exchange.change_initial_leverage(self.symbols, self.leverage)

        # Run trading loop.
        max_workers = multiprocessing.cpu_count() - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(
                self.run_symbol_trader, iter(self.get_next_trading_symbol, None)
            )
