import logging
import multiprocessing
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import peewee

from afang.database.ohlcv_db.ohlcv_database import OHLCVDatabase
from afang.database.trades_db.trades_database import Order as DBOrder
from afang.database.trades_db.trades_database import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import TradesDatabase
from afang.exchanges import IsExchange
from afang.exchanges.models import OrderSide, OrderType
from afang.models import Timeframe
from afang.strategies.analyzer import StrategyAnalyzer
from afang.strategies.models import SymbolAnalysisResult, TradeLevels
from afang.strategies.root import Root
from afang.utils.util import generate_uuid, resample_timeframe, time_str_to_milliseconds

logger = logging.getLogger(__name__)


class Backtester(Root):
    """Base interface for strategy backtests."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize Backtester class.

        :param strategy_name: name of the trading strategy.
        """

        Root.__init__(self, strategy_name=strategy_name)

    @abstractmethod
    def plot_backtest_indicators(self) -> Dict:
        """Get the indicators to plot on the backtest analysis dashboard.

        :return: Dict
        """

        return dict()

    def open_backtest_position(
        self,
        symbol: str,
        direction: int,
        trade_levels: TradeLevels,
        entry_time: datetime,
    ) -> Optional[DBTradePosition]:
        """Open a trade position for a given symbol.

        :param symbol: symbol to open a long trade position for.
        :param direction: new position trade direction. 1 for LONG. -1
            for SHORT.
        :param trade_levels: desired trade levels.
        :param entry_time: time at which the long trade was entered.
        :return: Optional[DBTradePosition]
        """

        open_order_id = generate_uuid()
        open_order_side = OrderSide.BUY if direction == 1 else OrderSide.SELL

        with self.shared_lock:
            initial_test_account_balance = self.initial_test_account_balance

            position_size = (
                self.leverage
                * (self.percentage_risk_per_trade / 100.0)
                * initial_test_account_balance
            )
            # Ensure that there is enough capital to open the trade.
            if position_size <= 0:
                logger.error(
                    "%s %s: inadequate capital to open new trade position",
                    self.exchange.display_name,
                    symbol,
                )
                return None
            if self.max_amount_per_trade and position_size > self.max_amount_per_trade:
                position_size = self.max_amount_per_trade

            position_margin = position_size / self.leverage
            self.initial_test_account_balance -= position_margin

        position_qty = position_size / trade_levels.entry_price

        try:
            new_trade_position = DBTradePosition.create(
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                desired_entry_price=trade_levels.entry_price,
                open_order_id=open_order_id,
                position_qty=position_qty,
                position_size=position_size,
                target_price=trade_levels.target_price,
                stop_price=trade_levels.stop_price,
                initial_account_balance=initial_test_account_balance,
                exchange_display_name=self.exchange.display_name,
            )

            DBOrder.create(
                symbol=symbol,
                is_open_order=True,
                direction=direction,
                order_id=open_order_id,
                order_side=open_order_side.value,
                raw_price=trade_levels.entry_price,
                original_price=trade_levels.entry_price,
                original_quantity=position_qty,
                remaining_quantity=position_qty,
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

    def update_closed_position_orders(
        self, position: DBTradePosition, close_order_type: OrderType
    ) -> None:
        """Update database orders for a closed backtest trade position.

        :param position: DB trade position whose close position orders
            need to be updated.
        :param close_order_type: order type to use for the close
            position order.
        :return: None
        """

        # create position close order.
        close_order_id = generate_uuid()
        close_order_side = OrderSide.SELL if position.direction == 1 else OrderSide.BUY
        try:
            DBOrder.create(
                symbol=position.symbol,
                is_open_order=False,
                direction=position.direction,
                is_open=False,
                order_id=close_order_id,
                order_side=close_order_side.value,
                raw_price=position.close_price,
                original_price=position.close_price,
                average_price=position.close_price,
                original_quantity=position.position_qty,
                executed_quantity=position.position_qty,
                remaining_quantity=0,
                order_type=close_order_type,
                exchange_display_name=self.exchange.display_name,
                commission=position.commission / 2,
                position=position,
            )
        except peewee.PeeweeException as error:
            logger.error(
                "%s %s: closed position close order could not be created: %s",
                self.exchange.display_name,
                position.symbol,
                error,
            )

    def close_backtest_position(
        self,
        position: DBTradePosition,
        close_price: float,
        exit_time: datetime,
        close_order_type: OrderType = OrderType.MARKET,
    ) -> Optional[DBTradePosition]:
        """Close an open trade position for a given symbol.

        :param position: DB trade position to close.
        :param close_price: price at which the trade exited.
        :param exit_time: time at which the trade exited.
        :param close_order_type: order type to use for the close
            position order.
        :return: Optional[DBTradePosition]
        """

        if not position.is_open:
            logger.error(
                "%s %s: attempting to close a closed position %s",
                self.exchange.display_name,
                position.symbol,
                position.position_id,
            )
            return None

        is_entry_order_filled = True if position.entry_price else False

        roe = float()
        position.executed_qty = float()
        if is_entry_order_filled:
            position.close_price = close_price
            position.executed_qty = position.position_qty
            roe = (
                (close_price / position.entry_price - 1) * position.direction
            ) * 100.0

        cost_adjusted_roe = roe - (2 * self.commission) - self.expected_slippage
        pnl = position.position_size * (cost_adjusted_roe / 100.0)
        commission = position.position_size * ((2 * self.commission) / 100.0)
        slippage = position.position_size * (self.expected_slippage / 100.0)

        position.roe = round(roe, 4) if is_entry_order_filled else float()
        position.pnl = round(pnl, 4) if is_entry_order_filled else float()
        position.commission = round(commission, 4) if is_entry_order_filled else float()
        position.slippage = round(slippage, 4) if is_entry_order_filled else float()
        position.cost_adjusted_roe = (
            round(cost_adjusted_roe, 4) if is_entry_order_filled else float()
        )

        position_margin = position.position_size / self.leverage
        with self.shared_lock:
            self.initial_test_account_balance += position_margin
            self.initial_test_account_balance += position.pnl
            position.final_account_balance = self.initial_test_account_balance

        position.is_open = False
        position.exit_time = exit_time

        # Persist finalized trade position.
        position.save()

        # Update position orders.
        if is_entry_order_filled:
            self.update_closed_position_orders(position, close_order_type)

        return position

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

        :param data: the historical price dataframe row at the current
            time in backtest.
        :return: bool
        """

        pass

    @abstractmethod
    def is_short_trade_signal_present(self, data: Any) -> bool:
        """Check if a short trade signal exists.

        :param data: the historical price dataframe row at the current
            time in backtest.
        :return: bool
        """

        pass

    @abstractmethod
    def generate_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> TradeLevels:
        """Generate price levels for an individual trade signal.

        :param data: the historical price dataframe row where the open
            trade signal was detected.
        :param trade_signal_direction: 1 for a long position. -1 for a
            short position.
        :return: TradeLevels
        """

        return TradeLevels(
            entry_price=data.close,
            target_price=None,
            stop_price=None,
        )

    def generate_and_verify_backtest_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> Optional[TradeLevels]:
        """Generate price levels for an individual trade signal and verify that
        they are valid.

        :param data: the historical price dataframe row where the open
            trade signal was detected.
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
                data.Index,
                trade_levels.entry_price,
                data.close,
            )
            return None

        return trade_levels

    def handle_position_open_orders(
        self, data: Any, open_symbol_positions: List[DBTradePosition]
    ) -> None:
        """Monitor and fill position open orders if they meet the necessary
        requirements.

        :param data: the historical price dataframe row at the current
            time in backtest.
        :param open_symbol_positions: open symbol trade positions.
        :return: None
        """

        for position in open_symbol_positions:
            # ensure that the trade position was not opened during the current candle.
            if position.entry_time == data.Index.to_pydatetime():
                continue

            # ensure that the trade position is still open.
            if not position.is_open:
                continue

            if data.high >= position.desired_entry_price >= data.low:
                # fill position open order
                position.entry_price = position.desired_entry_price
                position.save()

                try:
                    query = DBOrder.update(
                        {
                            DBOrder.is_open: False,
                            DBOrder.average_price: position.desired_entry_price,
                            DBOrder.executed_quantity: position.position_qty,
                            DBOrder.remaining_quantity: float(),
                            DBOrder.commission: position.position_size
                            * (self.commission / 100.0),
                        }
                    ).where(
                        DBOrder.order_id == position.open_order_id,
                        DBOrder.symbol == position.symbol,
                        DBOrder.exchange_display_name == self.exchange.display_name,
                    )
                    query.execute()
                except peewee.PeeweeException as error:
                    logger.error(
                        "%s %s: open order could not be marked as closed in the DB: %s",
                        self.exchange.display_name,
                        position.symbol,
                        error,
                    )

    def handle_open_backtest_positions(
        self, data: Any, open_symbol_positions: List[DBTradePosition]
    ) -> bool:
        """Monitor and handle open positions for a given symbol and close them
        if they hit a trade barrier. Returns true if a trade position is
        closed.

        :param data: the historical price dataframe row at the current
            time in backtest.
        :param open_symbol_positions: open symbol trade positions.
        :return: None
        """

        has_trade_position_closed = False
        for position in open_symbol_positions:
            # ensure that the trade position was not opened during the current candle.
            if position.entry_time == data.Index.to_pydatetime():
                continue

            # ensure that the trade position is still open.
            if not position.is_open:
                continue

            # increment trade holding time by 1.
            position.holding_time += 1
            position.save()

            close_position = None
            is_entry_order_filled = True if position.entry_price else False

            # check if the lower horizontal barrier has been hit for long positions.
            if (
                is_entry_order_filled
                and position.stop_price
                and data.low <= position.stop_price
                and data.high >= position.stop_price >= data.low
                and position.direction == 1
            ):
                close_position = self.close_backtest_position(
                    position,
                    position.stop_price,
                    data.Index.to_pydatetime(),
                    self.stop_loss_order_type,
                )

            # check if upper horizontal barrier has been hit for long positions.
            elif (
                is_entry_order_filled
                and position.target_price
                and data.high >= position.target_price
                and data.high >= position.target_price >= data.low
                and position.direction == 1
            ):
                close_position = self.close_backtest_position(
                    position,
                    position.target_price,
                    data.Index.to_pydatetime(),
                    self.take_profit_order_type,
                )

            # check if upper horizontal barrier has been hit for short positions.
            elif (
                is_entry_order_filled
                and position.stop_price
                and data.high >= position.stop_price
                and data.high >= position.stop_price >= data.low
                and position.direction == -1
            ):
                close_position = self.close_backtest_position(
                    position,
                    position.stop_price,
                    data.Index.to_pydatetime(),
                    self.stop_loss_order_type,
                )

            # check if lower horizontal barrier has been hit for short positions.
            elif (
                is_entry_order_filled
                and position.target_price
                and data.low <= position.target_price
                and data.high >= position.target_price >= data.low
                and position.direction == -1
            ):
                close_position = self.close_backtest_position(
                    position,
                    position.target_price,
                    data.Index.to_pydatetime(),
                    self.take_profit_order_type,
                )

            # check if vertical barrier has been hit.
            elif position.holding_time >= self.max_holding_candles:
                close_position = self.close_backtest_position(
                    position, data.close, data.Index.to_pydatetime()
                )

            # check if current candle is the last candle in the provided historical price data.
            elif data.Index == self.backtest_data[position.symbol].index.values[-1]:
                close_position = self.close_backtest_position(
                    position, data.close, data.Index.to_pydatetime()
                )

            if close_position:
                has_trade_position_closed = True

        return has_trade_position_closed

    def run_symbol_backtest(self, symbol: str) -> None:
        """Run trading backtest for a single symbol.

        :param symbol: symbol to run backtest for.
        :return: None
        """

        if symbol not in self.exchange.exchange_symbols:
            logger.error(
                "%s %s: provided symbol not present in the exchange",
                self.exchange.display_name,
                symbol,
            )
            return None

        logger.info(
            "%s %s %s: started backtest on the %s strategy",
            symbol,
            self.exchange.display_name,
            self.timeframe.value,
            self.strategy_name,
        )

        with self.shared_lock:
            ohlcv_db = OHLCVDatabase(self.exchange, symbol)
            ohlcv_data = ohlcv_db.get_data(
                symbol, self.backtest_from_time, self.backtest_to_time
            )
        if ohlcv_data is None:
            logger.warning(
                "%s %s %s: unable to get price data for the %s strategy",
                symbol,
                self.exchange.display_name,
                self.timeframe.value,
                self.strategy_name,
            )
            return None

        resampled_ohlcv_data = resample_timeframe(ohlcv_data, self.timeframe.value)

        # generate trading features.
        populated_ohlcv_data = self.generate_features(resampled_ohlcv_data)

        # remove unstable indicator values.
        idx = self.unstable_indicator_values
        self.backtest_data[symbol] = populated_ohlcv_data.iloc[idx:]

        # open the trades' database connection.
        self.trades_database.database.connect(reuse_if_open=True)

        open_symbol_positions: List[DBTradePosition] = []
        for row in self.backtest_data[symbol].itertuples():
            should_open_new_position = False
            new_position_direction = None

            # open a long position if we get a long trading signal.
            if (
                self.allow_long_positions
                and self.is_long_trade_signal_present(row)
                and row.Index != self.backtest_data[symbol].index.values[-1]
            ):
                should_open_new_position = True
                new_position_direction = 1

            # open a short position if we get a short trading signal.
            elif (
                self.allow_short_positions
                and self.is_short_trade_signal_present(row)
                and row.Index != self.backtest_data[symbol].index.values[-1]
            ):
                should_open_new_position = True
                new_position_direction = -1

            if should_open_new_position:
                # only open a position if multiple open positions are allowed or
                # there is no open position.
                if self.allow_multiple_open_positions or not len(open_symbol_positions):
                    trade_levels = self.generate_and_verify_backtest_trade_levels(
                        row, trade_signal_direction=new_position_direction
                    )
                    if trade_levels:
                        new_trade_position = self.open_backtest_position(
                            symbol=symbol,
                            direction=new_position_direction,
                            trade_levels=trade_levels,
                            entry_time=row.Index.to_pydatetime(),
                        )
                        if new_trade_position:
                            open_symbol_positions = self.fetch_open_trade_positions(
                                [symbol]
                            )

            # monitor and handle all open positions.
            self.handle_position_open_orders(row, open_symbol_positions)
            has_trade_position_closed = self.handle_open_backtest_positions(
                row, open_symbol_positions
            )
            if has_trade_position_closed:
                open_symbol_positions = self.fetch_open_trade_positions([symbol])

        # close the trades' database connection.
        self.trades_database.database.close()

        logger.info(
            "%s %s %s: completed backtest on the %s strategy",
            symbol,
            self.exchange.display_name,
            self.timeframe.value,
            self.strategy_name,
        )

    def run_backtest(
        self,
        exchange: IsExchange,
        symbols: Optional[List[str]],
        timeframe: Optional[str],
        from_time: Optional[str],
        to_time: Optional[str],
    ) -> Optional[List[SymbolAnalysisResult]]:
        """Run trading backtest for multiple symbols at once and return
        analysis results on the backtest.

        :param exchange: exchange to use to run the backtest.
        :param symbols: exchange symbols to run backtest on.
        :param timeframe: timeframe to run the backtest on.
        :param from_time: desired begin time of the backtest.
        :param to_time: desired end time of the backtest.
        :return: Optional[List[SymbolAnalysisResult]]
        """

        # Get symbols to backtest.
        self.symbols = symbols
        if not self.symbols:
            self.symbols = self.config.get("watchlist", dict()).get(exchange.name, [])
        if not self.symbols:
            logger.warning(
                "%s: no symbols found to run strategy backtest", self.strategy_name
            )
            return None

        # Record exchange to be used for backtest.
        self.exchange = exchange

        # Get the backtesting timeframe.
        if not timeframe:
            timeframe = self.config.get("timeframe", None)
        if not timeframe:
            logger.warning(
                "%s: timeframe not defined for the strategy backtest",
                self.strategy_name,
            )
            return None
        # Ensure the trading timeframe is supported.
        try:
            self.timeframe = Timeframe(timeframe)
        except ValueError:
            logger.warning(
                "%s: invalid timeframe %s defined for the strategy backtest",
                self.strategy_name,
                timeframe,
            )
            return None

        # Get the backtesting duration.
        self.backtest_from_time = 0
        if from_time:
            self.backtest_from_time = time_str_to_milliseconds(from_time)
        self.backtest_to_time = int(time.time()) * 1000
        if to_time:
            self.backtest_to_time = time_str_to_milliseconds(to_time)

        # Update the strategy config with the working parameters.
        self.config.update(
            {
                "timeframe": self.timeframe.value,
                "exchange": self.exchange,
                "backtest_from_time": self.backtest_from_time,
                "backtest_to_time": self.backtest_to_time,
            }
        )

        # Initialize backtest trades database.
        self.trades_database = TradesDatabase(db_name="trades_backtest.sqlite3")
        with self.trades_database.database:
            self.trades_database.database.drop_tables(
                self.trades_database.models, safe=True
            )
            self.trades_database.database.create_tables(
                self.trades_database.models, safe=True
            )

        max_workers = multiprocessing.cpu_count() - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.run_symbol_backtest, self.symbols)

        # Analyze the trading strategy.
        strategy_analyzer = StrategyAnalyzer(strategy=self)
        strategy_analysis = strategy_analyzer.run_analysis()

        return strategy_analysis
