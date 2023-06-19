import logging
import multiprocessing
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import peewee

from afang.database.trades_db.trades_database import Order as DBOrder
from afang.database.trades_db.trades_database import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import TradesDatabase
from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Order as ExchangeOrder
from afang.exchanges.models import OrderSide, OrderType, Symbol
from afang.models import Timeframe
from afang.strategies.models import TradeLevels
from afang.strategies.root import Root
from afang.utils.util import (
    generate_uuid,
    milliseconds_to_datetime,
    round_float_to_precision,
)

logger = logging.getLogger(__name__)


class Trader(Root):
    """Base interface for strategy live/demo trading."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize Trader class.

        :param strategy_name: name of the trading strategy.
        """
        Root.__init__(self, strategy_name=strategy_name)

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
        :param trade_signal_direction: 1 for a long position. -1 for a
            short position.
        :return: TradeLevels
        """

        return TradeLevels(
            entry_price=data.close,
            target_price=None,
            stop_price=None,
        )

    def generate_and_verify_trader_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> Optional[TradeLevels]:
        """Generate price levels for an individual trade signal and verify that
        they are valid.

        :param data: the candle where the open trade signal was
            detected.
        :param trade_signal_direction: 1 for a long position. -1 for a
            short position.
        :return: Optional[TradeLevels]
        """

        trade_levels = self.generate_trade_levels(data, trade_signal_direction)

        if (trade_signal_direction == 1 and trade_levels.entry_price < data.close) or (
            trade_signal_direction == -1 and trade_levels.entry_price > data.close
        ):
            logger.warning(
                "Generated trade levels are invalid. direction: %s. candle open time: %s. "
                "desired entry price: %s. current price: %s",
                trade_signal_direction,
                milliseconds_to_datetime(data.open_time),
                trade_levels.entry_price,
                data.close,
            )
            return None

        return trade_levels

    def fetch_order_by_exchange_id(self, order_id: str) -> Optional[DBOrder]:
        """Fetch an order by its exchange order ID.

        :param order_id: exchange order ID of the order to be fetched.
        :return: Optional[DBOrder]
        """

        try:
            db_order: DBOrder = (
                DBOrder.select(DBOrder, DBTradePosition)
                .join(DBTradePosition)
                .where(
                    DBOrder.order_id == order_id,
                    DBOrder.exchange_display_name == self.exchange.display_name,
                )
                .get()
            )
            return db_order
        except (DBOrder.DoesNotExist, peewee.PeeweeException) as error:
            logger.warning(
                "%s: order not found in DB. exchange order id: %s: %s",
                self.exchange.display_name,
                order_id,
                error,
            )
            return None

    def get_next_trading_symbol(self, run_forever: bool = True) -> Optional[str]:
        """Get the next trading symbol from the execution queue to trade.

        :param run_forever: whether to continuously run the function.
            Used for testing purposes.
        :return: Optional[str]
        """

        while True:
            if not self.trading_execution_queue.empty():
                return self.trading_execution_queue.get()
            if not run_forever:
                return None
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

    def get_quote_asset_wallet_balance(self, symbol: str) -> Optional[float]:
        """Get the quote asset wallet balance of a given symbol.

        :param symbol: trading symbol.
        :return: Optional[float]
        """

        if self.on_demo_mode:
            with self.shared_lock:
                return self.initial_test_account_balance

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

        return quote_asset_balance.wallet_balance

    def get_open_order_position_size(self, quote_asset_wallet_balance: float) -> float:
        """Get the intended position size for a position open order.

        :param quote_asset_wallet_balance: quote asset wallet balance.
        :return: float
        """

        intended_position_size = self.leverage * (
            (self.percentage_risk_per_trade / 100.0) * quote_asset_wallet_balance
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

        position_db_orders: List[DBOrder] = position.orders

        close_order_quantity: float = float()
        for db_order in position_db_orders:
            exchange_order = self.get_exchange_order(db_order.symbol, db_order.order_id)
            if not exchange_order:
                continue

            if db_order.is_open_order:
                close_order_quantity += exchange_order.executed_quantity
            else:
                close_order_quantity -= exchange_order.executed_quantity

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

    def update_closed_order_in_db(
        self,
        exchange_order: ExchangeOrder,
        db_order: DBOrder,
    ) -> None:
        """Update a closed order's details in the DB.

        :param exchange_order: exchange order instance.
        :param db_order: database order instance.
        :return: None
        """

        try:
            query = DBOrder.update(
                {
                    DBOrder.is_open: False,
                    DBOrder.time_in_force: exchange_order.time_in_force,
                    DBOrder.average_price: exchange_order.average_price,
                    DBOrder.executed_quantity: exchange_order.executed_quantity,
                    DBOrder.remaining_quantity: exchange_order.remaining_quantity,
                    DBOrder.order_status: exchange_order.order_status,
                    DBOrder.commission: exchange_order.commission,
                }
            ).where(
                DBOrder.id == db_order.id,
                DBOrder.symbol == exchange_order.symbol,
                DBOrder.exchange_display_name == self.exchange.display_name,
            )
            query.execute()
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: closed order %s could not be updated in the DB: %s",
                self.exchange.display_name,
                db_order.symbol,
                db_order.id,
                error,
            )

    def cancel_position_order(self, symbol: str, exchange_order_id: str) -> None:
        """Cancel a position order if some/all of its quantity is un-executed
        and mark the order as completed in the DB.

        :param symbol: name of symbol whose position order is to be
            cancelled.
        :param exchange_order_id: exchange order ID of the order to be
            canceled.
        :return: None
        """

        db_position_order = self.fetch_order_by_exchange_id(exchange_order_id)
        if not db_position_order:
            logger.error(
                "%s %s: position order not cancelled because it was not found in the DB. exchange order id: %s",
                self.exchange.display_name,
                symbol,
                exchange_order_id,
            )
            return None

        exchange_position_order = self.get_exchange_order(
            db_position_order.symbol, exchange_order_id
        )
        if not exchange_position_order:
            logger.error(
                "%s %s: position order not canceled because order was not found on the exchange. "
                "exchange order id: %s",
                self.exchange.display_name,
                db_position_order.symbol,
                exchange_order_id,
            )
            return None

        if exchange_position_order.remaining_quantity:
            logger.info(
                "%s %s: attempting to cancel position order due to un-executed qty. "
                "exchange order id: %s",
                self.exchange.display_name,
                exchange_position_order.symbol,
                exchange_order_id,
            )
            if not self.on_demo_mode:
                self.exchange.cancel_order(
                    exchange_position_order.symbol, order_id=exchange_order_id
                )

        self.update_closed_order_in_db(exchange_position_order, db_position_order)

    def is_order_filled(self, symbol: str, order_exchange_id: str) -> bool:
        """
        Returns a bool on whether an order has been filled or not.
        NOTE: Even a partial order fill will return true.

        :param symbol: order symbol name.
        :param order_exchange_id: exchange ID of the order to query.
        :return: bool
        """

        order = self.get_exchange_order(symbol, order_exchange_id)
        if order and order.executed_quantity:
            return True

        return False

    def get_order_average_price(self, symbol: str, order_exchange_id: str) -> float:
        """Get the average price of a given order.

        :param symbol: order symbol name.
        :param order_exchange_id: order exchange ID.
        :return: float
        """

        order = self.get_exchange_order(symbol, order_exchange_id)
        if not order:
            logger.warning(
                "%s %s: Could not get the order average price for %s order: %s",
                self.exchange.display_name,
                self.strategy_name,
                symbol,
                order_exchange_id,
            )
            return float()

        return order.average_price

    def get_symbol_ohlcv_candles_df(self, symbol: str) -> pd.DataFrame:
        """Get a symbol's OHLCV candles as a dataframe.

        :param symbol: symbol whose OHLCV candles are to be fetched.
        :return: pd.Dataframe
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
            return pd.DataFrame()

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

        :param symbol: symbol whose current trading candle is being
            fetched.
        :param ohlcv_data: dataframe that contains a symbol's OHLCV
            data.
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
            return None

        current_candle_data = current_candle_data_list[0]
        return current_candle_data

    @classmethod
    def get_position_total_commission(cls, position: DBTradePosition) -> float:
        """Get the total commission accrued by a given trade position.

        :param position: database trade position.
        :return: float
        """

        total_commission = float()

        position_order: DBOrder
        for position_order in position.orders:
            total_commission += position_order.commission

        return total_commission

    @classmethod
    def get_position_total_slippage(cls, position: DBTradePosition) -> float:
        """Get the total slippage for a given trade position.

        :param position: database trade position.
        :return: float
        """

        total_slippage = float()

        position_order: DBOrder
        for position_order in position.orders:
            price_execution_difference = (
                position_order.average_price - position_order.original_price
            )
            order_slippage = (
                price_execution_difference
                * position_order.executed_quantity
                * position_order.direction
            )
            total_slippage += order_slippage

        return total_slippage

    @classmethod
    def get_position_executed_qty(cls, position: DBTradePosition) -> float:
        """Get the total executed quantity for a given trade position.

        :param position: database trade position.
        :return: float
        """

        total_executed_qty = float()

        position_order: DBOrder
        for position_order in position.orders:
            if not position_order.is_open_order:
                total_executed_qty += position_order.executed_quantity

        return total_executed_qty

    def get_position_close_price(self, position: DBTradePosition) -> float:
        """Get the average close price for a given trade position.

        :param position: database trade position.
        :return: float
        """

        position_close_price = float()

        position_executed_qty = self.get_position_executed_qty(position)

        position_order: DBOrder
        for position_order in position.orders:
            if not position_order.is_open_order:
                position_close_price += position_order.average_price * (
                    position_order.executed_quantity / position_executed_qty
                )

        return position_close_price

    def get_position_pnl(self, position: DBTradePosition) -> float:
        """Get the PnL of a given trade position.

        :param position: database trade position.
        :return: float
        """

        position_pnl = float()

        position_order: DBOrder
        for position_order in position.orders:
            order_size = position_order.average_price * position_order.executed_quantity
            if position_order.is_open_order:
                order_size *= -1
            position_pnl += order_size

        position_pnl *= position.direction
        position_pnl -= self.get_position_total_commission(position)

        return position_pnl

    def get_position_roe(
        self,
        position: DBTradePosition,
        cost_adjusted: bool = False,
    ) -> float:
        """Get the ROE of a given trade position.

        :param position: database trade position.
        :param cost_adjusted: whether to calculate the cost adjusted ROE
            i.e. inclusive of commission.
        :return: float
        """

        position_pnl = self.get_position_pnl(position)
        if not cost_adjusted:
            position_pnl += self.get_position_total_commission(position)

        open_position_order = self.fetch_order_by_exchange_id(position.open_order_id)
        if not open_position_order:
            logger.error(
                "%s %s: could not calculate position ROE because the open order could not be found. "
                "open order id: %s",
                self.exchange.display_name,
                position.symbol,
                position.open_order_id,
            )
            return float()

        open_order_size = (
            open_position_order.executed_quantity * open_position_order.average_price
        )

        roe = (position_pnl / open_order_size) * 100.0
        return roe

    def close_trade_position(self, position: DBTradePosition) -> None:
        """Mark a trade position as closed in the DB.

        :param position: database trade position.
        :return: None
        """

        quote_asset_wallet_balance = self.get_quote_asset_wallet_balance(
            position.symbol
        )
        if quote_asset_wallet_balance is None:
            return None

        try:
            position_open_order = DBOrder.get(
                DBOrder.order_id == position.open_order_id,
                DBOrder.is_open_order.__eq__(True),
                DBOrder.symbol == position.symbol,
                DBOrder.exchange_display_name == self.exchange.display_name,
            )
        except (DBOrder.DoesNotExist, peewee.PeeweeException) as error:
            logger.error(
                "%s %s: trade position %s not marked as closed. position open order not found: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
                error,
            )
            return None

        # if the position open order is still open, then the open order was filled.
        is_open_order_filled = position_open_order.is_open

        pnl = self.get_position_pnl(position) if is_open_order_filled else float()
        final_account_balance = quote_asset_wallet_balance
        if self.on_demo_mode:
            with self.shared_lock:
                position_margin = position.position_size / self.leverage
                self.initial_test_account_balance += position_margin
                self.initial_test_account_balance += pnl
                final_account_balance = self.initial_test_account_balance

        slippage = (
            self.get_position_total_slippage(position)
            if is_open_order_filled
            else float()
        )
        close_price = (
            self.get_position_close_price(position) if is_open_order_filled else None
        )
        roe = self.get_position_roe(position) if is_open_order_filled else float()
        executed_qty = (
            self.get_position_executed_qty(position)
            if is_open_order_filled
            else float()
        )
        commission = (
            self.get_position_total_commission(position)
            if is_open_order_filled
            else float()
        )
        entry_price = (
            self.get_order_average_price(position.symbol, position.open_order_id)
            if is_open_order_filled
            else None
        )
        cost_adjusted_roe = (
            self.get_position_roe(position, cost_adjusted=True)
            if is_open_order_filled
            else float()
        )

        try:
            query = DBTradePosition.update(
                {
                    DBTradePosition.is_open: False,
                    DBTradePosition.exit_time: datetime.utcnow(),
                    DBTradePosition.pnl: pnl,
                    DBTradePosition.slippage: slippage,
                    DBTradePosition.close_price: close_price,
                    DBTradePosition.roe: roe,
                    DBTradePosition.executed_qty: executed_qty,
                    DBTradePosition.commission: commission,
                    DBTradePosition.final_account_balance: final_account_balance,
                    DBTradePosition.entry_price: entry_price,
                    DBTradePosition.cost_adjusted_roe: cost_adjusted_roe,
                }
            ).where(DBTradePosition.id == position.id)
            query.execute()
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: trade position %s could not be marked as closed in the DB: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
                error,
            )

    def get_exchange_order(self, symbol: str, order_id: str) -> Optional[ExchangeOrder]:
        """Get symbol exchange order.

        :param symbol: symbol whose order is to be fetched.
        :param order_id: exchange order ID.
        :return: Optional[ExchangeOrder]
        """

        if not self.on_demo_mode:
            return self.exchange.get_exchange_order(symbol, order_id)

        for order in self.demo_mode_exchange_orders.get(symbol, list()):
            if order.order_id == order_id:
                return order

        logger.warning(
            "Unable to get %s %s demo order with id: %s",
            self.exchange.display_name,
            symbol,
            order_id,
        )
        return None

    def add_demo_mode_order(
        self,
        symbol_name: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: float = float(),
    ) -> str:
        """Add a new demo mode order.

        :param symbol_name: name of symbol.
        :param side: order side.
        :param quantity: order quantity.
        :param order_type: order type.
        :param price: optional. order price.
        :return: str
        """

        order = ExchangeOrder(
            symbol=symbol_name,
            order_id=generate_uuid(),
            side=side,
            original_price=price,
            average_price=float(),
            original_quantity=quantity,
            executed_quantity=float(),
            remaining_quantity=quantity,
            order_type=order_type,
            order_status="DEMO_ORDER_STATUS",
            time_in_force="DEMO_TIME_IN_FORCE",
            commission=float(),
        )

        with self.shared_lock:
            self.demo_mode_exchange_orders[symbol_name].append(order)

        return order.order_id

    def initialize_demo_mode_orders(self, symbols: List[str]) -> None:
        """Initialize demo mode orders.

        :param symbols: symbols whose demo mode orders should be
            initialized.
        :return: None
        """

        self.trades_database.database.connect(reuse_if_open=True)

        try:
            demo_mode_orders = DBOrder.select().where(
                DBOrder.symbol.in_(symbols),
                DBOrder.exchange_display_name == self.exchange.display_name,
            )
        except peewee.PeeweeException as error:
            logger.error(
                "%s: demo orders not initialized. Couldn't be fetched from the DB: %s",
                self.exchange.display_name,
                error,
            )
            return None

        for order in demo_mode_orders:
            demo_exchange_order = ExchangeOrder(
                symbol=order.symbol,
                order_id=order.order_id,
                side=OrderSide(order.order_side),
                original_price=order.original_price,
                average_price=order.average_price,
                original_quantity=order.original_quantity,
                executed_quantity=order.executed_quantity,
                remaining_quantity=order.remaining_quantity,
                order_type=OrderType(order.order_type),
                order_status="DEMO_ORDER_STATUS",
                time_in_force="DEMO_TIME_IN_FORCE",
                commission=order.commission,
            )

            with self.shared_lock:
                self.demo_mode_exchange_orders[order.symbol].append(demo_exchange_order)

        self.trades_database.database.close()

    def update_executed_demo_order_in_db(self, demo_order: ExchangeOrder) -> None:
        """Update an executed demo exchange order in the DB.

        :param demo_order: demo mode exchange order.
        :return: None
        """

        try:
            query = DBOrder.update(
                {
                    DBOrder.average_price: demo_order.average_price,
                    DBOrder.executed_quantity: demo_order.executed_quantity,
                    DBOrder.remaining_quantity: demo_order.remaining_quantity,
                    DBOrder.commission: demo_order.commission,
                }
            ).where(
                DBOrder.order_id == demo_order.order_id,
                DBOrder.symbol == demo_order.symbol,
                DBOrder.exchange_display_name == self.exchange.display_name,
            )
            query.execute()
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: demo mode order %s not marked as executed in the DB: %s",
                self.exchange.display_name,
                demo_order.symbol,
                demo_order.order_id,
                error,
            )

    def update_symbol_demo_mode_orders(
        self, symbol: str, current_trading_candle: Any
    ) -> None:
        """Update a symbol's demo mode orders depending on the current trading
        candle.

        :param symbol: symbol whose demo mode orders should be updated.
        :param current_trading_candle: current trading candle.
        :return: None
        """

        for order in self.demo_mode_exchange_orders.get(symbol, list()):
            if not order.remaining_quantity:
                continue

            # Fetch DB order.
            try:
                db_order = (
                    DBOrder.select(DBOrder, DBTradePosition)
                    .join(DBTradePosition)
                    .where(
                        DBOrder.order_id == order.order_id,
                        DBOrder.symbol == symbol,
                        DBOrder.exchange_display_name == self.exchange.display_name,
                    )
                    .get()
                )
                db_position = db_order.position
            except (DBOrder.DoesNotExist, peewee.PeeweeException) as error:
                logger.error(
                    "%s %s: demo order %s not updated because it could not be found in the DB: %s",
                    self.exchange.display_name,
                    symbol,
                    order.order_id,
                    error,
                )
                continue

            position_entry_price = self.get_order_average_price(
                db_position.symbol, db_position.open_order_id
            )
            position_size = (
                current_trading_candle.close * order.original_quantity * self.leverage
            )
            commission = (self.commission / 100.0) * position_size

            if (
                # is a MARKET open trade position order.
                (db_order.is_open_order and order.order_type == OrderType.MARKET)
                or (
                    # is a LONG entry order/LONG take profit order/SHORT stop loss order.
                    (
                        (order.side == OrderSide.BUY and db_order.is_open_order)
                        or (
                            position_entry_price
                            and position_entry_price <= order.original_price
                        )
                    )
                    and order.average_price
                    and current_trading_candle.close
                    >= order.original_price
                    >= order.average_price
                )
                or (
                    # is a SHORT entry order/SHORT take profit order/LONG stop loss order.
                    (
                        (order.side == OrderSide.SELL and db_order.is_open_order)
                        or (
                            position_entry_price
                            and position_entry_price >= order.original_price
                        )
                    )
                    and order.average_price
                    and current_trading_candle.close
                    <= order.original_price
                    <= order.average_price
                )
            ):
                order.average_price = current_trading_candle.close
                order.executed_quantity = order.original_quantity
                order.remaining_quantity = float()
                order.commission = commission
                self.update_executed_demo_order_in_db(order)
                continue

            # Temporarily setting the order's average price to
            # help determine whether the order should be executed in future iterations.
            if not order.average_price:
                order.average_price = current_trading_candle.close
            # is a LONG entry order/LONG take profit order/SHORT stop loss order.
            if (order.side == OrderSide.BUY and db_order.is_open_order) or (
                position_entry_price and position_entry_price <= order.original_price
            ):
                order.average_price = min(
                    order.average_price, current_trading_candle.close
                )
            # is a SHORT entry order/SHORT take profit order/LONG stop loss order.
            elif (order.side == OrderSide.SELL and db_order.is_open_order) or (
                position_entry_price and position_entry_price >= order.original_price
            ):
                order.average_price = max(
                    order.average_price, current_trading_candle.close
                )

    def place_order(
        self,
        symbol_name: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float] = None,
    ) -> Optional[str]:
        """Place a new order for a specified symbol on the exchange or
        simulated when on demo mode.

        :param symbol_name: name of symbol.
        :param side: order side.
        :param quantity: order quantity.
        :param order_type: order type.
        :param price: optional. order price.
        :return: Optional[str]
        """

        if self.on_demo_mode:
            order_id = self.add_demo_mode_order(
                symbol_name, side, quantity, order_type, price
            )
            return order_id

        return self.exchange.place_order(
            symbol_name=symbol_name,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            post_only=self.post_only_orders,
            dydx_limit_fee=self.dydx_limit_fee,
        )

    def open_trade_position(
        self,
        symbol: str,
        direction: int,
        trade_levels: TradeLevels,
    ) -> Optional[DBTradePosition]:
        """Open a LONG or SHORT trade position for a given symbol.

        :param symbol: symbol to open a trade position for.
        :param direction: whether to open a LONG/SHORT trade position. 1
            for LONG. -1 for SHORT.
        :param trade_levels: desired trade levels.
        :return: None
        """

        logger.info(
            "%s %s: attempting to open a new trade position. "
            "entry price: %s. target price: %s. stop price: %s",
            self.exchange.display_name,
            symbol,
            round(trade_levels.entry_price, 4),
            round(trade_levels.target_price, 4),
            round(trade_levels.stop_price, 4),
        )

        trading_symbol = self.get_trading_symbol(symbol)
        if not trading_symbol:
            return None

        quote_asset_wallet_balance = self.get_quote_asset_wallet_balance(symbol)
        if quote_asset_wallet_balance is None:
            return None

        intended_position_size = self.get_open_order_position_size(
            quote_asset_wallet_balance
        )
        intended_open_order_qty = intended_position_size / trade_levels.entry_price

        precise_open_order_qty = round_float_to_precision(
            intended_open_order_qty, trading_symbol.step_size
        )
        if not self.is_order_qty_valid(trading_symbol, precise_open_order_qty):
            return None

        precise_open_order_entry_price = round_float_to_precision(
            trade_levels.entry_price, trading_symbol.tick_size
        )
        if not self.is_order_price_valid(
            trading_symbol, precise_open_order_entry_price
        ):
            return None

        precise_position_size = precise_open_order_qty * precise_open_order_entry_price

        open_order_side = OrderSide.BUY if direction == 1 else OrderSide.SELL

        # Attempt to place an open position order on the exchange.
        open_order_id = self.place_order(
            symbol_name=symbol,
            side=open_order_side,
            quantity=precise_open_order_qty,
            order_type=self.open_order_type,
            price=precise_open_order_entry_price,
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

        if self.on_demo_mode:
            with self.shared_lock:
                position_margin = precise_position_size / self.leverage
                self.initial_test_account_balance -= position_margin

        try:
            new_trade_position = DBTradePosition.create(
                symbol=symbol,
                direction=direction,
                entry_time=datetime.utcnow(),
                desired_entry_price=precise_open_order_entry_price,
                open_order_id=open_order_id,
                position_qty=precise_open_order_qty,
                position_size=precise_position_size,
                target_price=trade_levels.target_price,
                stop_price=trade_levels.stop_price,
                initial_account_balance=quote_asset_wallet_balance,
                exchange_display_name=self.exchange.display_name,
                sequence_id=trade_levels.sequence_id or generate_uuid(),
            )

            DBOrder.create(
                symbol=symbol,
                is_open_order=True,
                direction=direction,
                order_id=open_order_id,
                order_side=open_order_side.value,
                raw_price=trade_levels.entry_price,
                original_price=precise_open_order_entry_price,
                original_quantity=precise_open_order_qty,
                remaining_quantity=precise_open_order_qty,
                order_type=self.open_order_type.value,
                exchange_display_name=self.exchange.display_name,
                position=new_trade_position,
            )
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: new trade position could not be persisted to the DB: %s",
                self.exchange.display_name,
                symbol,
                error,
            )
            return None

        return new_trade_position

    def place_close_trade_position_order(
        self,
        position: DBTradePosition,
        close_price: float,
        close_order_type: OrderType,
    ) -> None:
        """Place a close trade position order for a given symbol.

        :param position: DB position to place a close trade order for.
        :param close_price: price at which the trade should be closed
            at.
        :param close_order_type: order type to use for the close
            position order.
        :return: None
        """

        logger.info(
            "%s %s: attempting to place a close trade position order. "
            "position id: %s. close price: %s",
            self.exchange.display_name,
            position.symbol,
            position.id,
            round(close_price, 4),
        )

        # Fetch trading symbol info.
        trading_symbol = self.get_trading_symbol(position.symbol)
        if not trading_symbol:
            return None

        # Ensure that the trade position is still open.
        if not position.is_open:
            logger.error(
                "%s %s: trade position is already closed. position id: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
            )
            return None

        # Check if the trade position open order has been filled - even partially.
        if not self.is_order_filled(position.symbol, position.open_order_id):
            logger.info(
                "%s %s: trade position open order is not filled. position id: %s",
                self.exchange.display_name,
                position.symbol,
                position.id,
            )
            # Mark position as completed by marking the open order as being closed.
            try:
                query = DBOrder.update(is_open=False).where(
                    DBOrder.order_id == position.open_order_id,
                    DBOrder.symbol == position.symbol,
                    DBOrder.exchange_display_name == self.exchange.display_name,
                )
                query.execute()
            except peewee.PeeweeException as error:
                logger.error(
                    "%s %s: trade position %s could not be marked as being temporarily completed: %s",
                    self.exchange.display_name,
                    position.symbol,
                    position.id,
                    error,
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

        close_order_side = OrderSide.SELL if position.direction == 1 else OrderSide.BUY

        # Attempt to place a close position order on the exchange.
        close_order_id = self.place_order(
            symbol_name=position.symbol,
            side=close_order_side,
            quantity=precise_close_order_qty,
            order_type=close_order_type,
            price=precise_close_order_price,
        )
        if not close_order_id:
            logger.error(
                "%s %s: failed to place close order on the exchange. price: %s. qty: %s",
                self.exchange.display_name,
                position.symbol,
                precise_close_order_price,
                precise_close_order_qty,
            )
            return None

        logger.info(
            "%s %s: close order successfully placed on the exchange. position id: %s. order id: %s",
            self.exchange.display_name,
            position.symbol,
            position.id,
            close_order_id,
        )

        try:
            DBOrder.create(
                symbol=position.symbol,
                is_open_order=False,
                direction=position.direction,
                order_id=close_order_id,
                order_side=close_order_side.value,
                raw_price=close_price,
                original_price=precise_close_order_price,
                original_quantity=precise_close_order_qty,
                remaining_quantity=precise_close_order_qty,
                order_type=close_order_type.value,
                exchange_display_name=self.exchange.display_name,
                position=position,
            )
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: close trade position order could not be placed: %s",
                self.exchange.display_name,
                position.symbol,
                error,
            )

    def calibrate_position_order_quantities(self, position: DBTradePosition) -> None:
        """Ensure that the remaining order quantities of all close position
        orders are valid.

        :param position: position whose close position orders are to be
            calibrated.
        :return: None
        """

        remaining_order_quantities = dict()
        total_remaining_qty: float = float()
        position_db_orders: List[DBOrder] = position.orders

        for db_order in position_db_orders:
            exchange_order = self.get_exchange_order(db_order.symbol, db_order.order_id)
            order_executed_qty = exchange_order.executed_quantity or float()
            if db_order.is_open_order:
                total_remaining_qty += order_executed_qty
            else:
                total_remaining_qty -= order_executed_qty
                remaining_order_quantities[
                    db_order.order_id
                ] = exchange_order.remaining_quantity

        for db_order in position_db_orders:
            if (not db_order.is_open) or db_order.is_open_order:
                continue

            remaining_order_qty = remaining_order_quantities[db_order.order_id]
            if remaining_order_qty != total_remaining_qty:
                # cancel position order.
                self.cancel_position_order(db_order.symbol, db_order.order_id)

                # place an updated close position order.
                if total_remaining_qty:
                    self.place_close_trade_position_order(
                        position=position,
                        close_price=db_order.raw_price,
                        close_order_type=OrderType(db_order.order_type),
                    )

    def handle_open_trade_positions(
        self,
        current_candle_data: Any,
        open_symbol_positions: List[DBTradePosition],
    ) -> None:
        """Monitor and handle open positions for a given symbol and place close
        trade position orders if they hit a trade barrier.

        :param current_candle_data: the current trading candle data.
        :param open_symbol_positions: open positions for the given
            symbol.
        :return: None
        """

        for position in open_symbol_positions:
            # ensure that the trade position is still open.
            if not position.is_open:
                continue

            is_open_order_open = False
            open_order_entry_prices: List[float] = list()
            trade: DBOrder
            for trade in position.orders:
                if trade.is_open_order:
                    is_open_order_open = trade.is_open
                    continue

                if trade.is_open:
                    open_order_entry_prices.append(trade.raw_price)

            # check if the trade position open order has been filled - even partially.
            is_open_order_filled = self.is_order_filled(
                position.symbol, position.open_order_id
            )

            # check if upper horizontal barrier has been hit for long positions.
            if (
                is_open_order_filled
                and is_open_order_open
                and position.target_price
                and position.target_price not in open_order_entry_prices
                and position.direction == 1
                and (
                    (
                        self.take_profit_order_type == OrderType.MARKET
                        and current_candle_data.close >= position.target_price
                    )
                    or (
                        self.take_profit_order_type == OrderType.LIMIT
                        and current_candle_data.close < position.target_price
                    )
                )
            ):
                self.place_close_trade_position_order(
                    position=position,
                    close_price=position.target_price,
                    close_order_type=self.take_profit_order_type,
                )

            # check if the lower horizontal barrier has been hit for long positions.
            elif (
                is_open_order_filled
                and is_open_order_open
                and position.stop_price
                and position.stop_price not in open_order_entry_prices
                and position.direction == 1
                and self.stop_loss_order_type == OrderType.MARKET
                and current_candle_data.close <= position.stop_price
            ):
                self.place_close_trade_position_order(
                    position=position,
                    close_price=position.stop_price,
                    close_order_type=self.stop_loss_order_type,
                )

            # check if lower horizontal barrier has been hit for short positions.
            elif (
                is_open_order_filled
                and is_open_order_open
                and position.target_price
                and position.target_price not in open_order_entry_prices
                and position.direction == -1
                and (
                    (
                        self.take_profit_order_type == OrderType.MARKET
                        and current_candle_data.close <= position.target_price
                    )
                    or (
                        self.take_profit_order_type == OrderType.LIMIT
                        and current_candle_data.close > position.target_price
                    )
                )
            ):
                self.place_close_trade_position_order(
                    position=position,
                    close_price=position.target_price,
                    close_order_type=self.take_profit_order_type,
                )

            # check if upper horizontal barrier has been hit for short positions.
            elif (
                is_open_order_filled
                and is_open_order_open
                and position.stop_price
                and position.stop_price not in open_order_entry_prices
                and position.direction == -1
                and self.stop_loss_order_type == OrderType.MARKET
                and current_candle_data.close >= position.stop_price
            ):
                self.place_close_trade_position_order(
                    position=position,
                    close_price=position.stop_price,
                    close_order_type=self.stop_loss_order_type,
                )

            # check if vertical barrier has been hit.
            # TODO: If the position has not yet been filled/has been filled - even partially - but
            #  has been open for more than *holding time*, close it.

            self.calibrate_position_order_quantities(position)

    def handle_finalized_trade_positions(
        self,
        open_symbol_positions: List[DBTradePosition],
    ) -> None:
        """Monitor and handle trade positions that have been finalized on the
        exchange but have not been marked as completed in the trade database.

        :param open_symbol_positions: open positions for the given
            symbol.
        :return: None
        """

        for position in open_symbol_positions:
            # ensure that the trade position is still open.
            if not position.is_open:
                continue

            # check if the trade position open order has been filled - even partially.
            try:
                position_open_order = DBOrder.get(
                    DBOrder.order_id == position.open_order_id,
                    DBOrder.symbol == position.symbol,
                    DBOrder.exchange_display_name == self.exchange.display_name,
                )
            except (DBOrder.DoesNotExist, peewee.PeeweeException) as error:
                logger.error(
                    "%s %s: could not fetch position open order: %s",
                    self.exchange.display_name,
                    position.symbol,
                    error,
                )
                continue

            if (
                not self.is_order_filled(position.symbol, position.open_order_id)
                and position_open_order.is_open
            ):
                continue

            # get total remaining position quantity.
            position_db_orders: List[DBOrder] = position.orders
            total_remaining_position_qty: float = float()
            for db_order in position_db_orders:
                exchange_order = self.get_exchange_order(
                    db_order.symbol, db_order.order_id
                )
                order_executed_qty = exchange_order.executed_quantity or float()
                if db_order.is_open_order:
                    total_remaining_position_qty += order_executed_qty
                else:
                    total_remaining_position_qty -= order_executed_qty

            # ensure that the position is ready to be closed.
            if total_remaining_position_qty > 0 and position_open_order.is_open:
                continue

            # close the trade position.
            self.close_trade_position(position)
            for db_order in position_db_orders:
                self.cancel_position_order(position.symbol, db_order.order_id)

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
        if ohlcv_candles_df.empty:
            return None

        populated_ohlcv_data = self.generate_features(ohlcv_candles_df)
        current_candle_data = self.get_symbol_current_trading_candle(
            symbol, populated_ohlcv_data
        )
        if current_candle_data is None:
            return None

        # open the trades' database connection.
        self.trades_database.database.connect(reuse_if_open=True)

        if self.on_demo_mode:
            self.update_symbol_demo_mode_orders(symbol, current_candle_data)

        symbol_open_positions = self.fetch_open_trade_positions([symbol])

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
                trade_levels = self.generate_and_verify_trader_trade_levels(
                    current_candle_data, trade_signal_direction=new_position_direction
                )
                if trade_levels:
                    self.open_trade_position(
                        symbol=symbol,
                        direction=new_position_direction,
                        trade_levels=trade_levels,
                    )

        # monitor and handle all open symbol positions.
        self.handle_open_trade_positions(current_candle_data, symbol_open_positions)
        self.handle_finalized_trade_positions(symbol_open_positions)

        # close the trades' database connection.
        self.trades_database.database.close()

        # add the symbol back to the trading execution queue.
        time.sleep(2.0)
        self.trading_execution_queue.put(symbol)

    def get_db_name(self) -> str:
        """Get the appropriate database name.

        :return: str
        """

        db_name = "trades.sqlite3"
        if self.on_demo_mode:
            db_name = "trades_on-demo-mode.sqlite3"
        elif self.exchange.testnet:
            db_name = "trades_on-testnet.sqlite3"

        return db_name

    def run_trader(
        self,
        exchange: IsExchange,
        symbols: Optional[List[str]],
        timeframe: Optional[str],
        demo_mode: Optional[bool] = True,
        db_path: Optional[str] = None,
    ) -> None:
        """Trade multiple symbols either on demo or live mode.

        :param exchange: exchange to use to run the trader.
        :param symbols: exchange symbols to run trader on.
        :param timeframe: timeframe to run the trader on.
        :param demo_mode: whether to run the trader on demo mode.
        :param db_path: database path. optional.
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

        logger.info(
            "%s: running %s on %s mode",
            self.exchange.display_name,
            self.strategy_name,
            "demo" if self.on_demo_mode else "live",
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

        # Initialize the trades' database with the appropriate name.
        db_name = self.get_db_name()
        self.trades_database = TradesDatabase(db_name=db_path if db_path else db_name)

        # Initialize demo exchange orders.
        if self.on_demo_mode:
            self.initialize_demo_mode_orders(self.symbols)

        # Run trading loop.
        max_workers = multiprocessing.cpu_count() - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(
                self.run_symbol_trader, iter(self.get_next_trading_symbol, None)
            )
