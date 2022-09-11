import logging
import multiprocessing
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from afang.database.ohlcv_database import OHLCVDatabase
from afang.exchanges import IsExchange
from afang.strategies.analyzer import StrategyAnalyzer
from afang.strategies.models import TradeLevels, TradePosition
from afang.utils.util import resample_timeframe, time_str_to_milliseconds

logger = logging.getLogger(__name__)


class Backtester(ABC):
    """Base interface for strategy backtests."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize Backtester class.

        :param strategy_name: name of the trading strategy.
        """

        self.strategy_name: str = strategy_name
        self.allow_long_positions: bool = True
        self.allow_short_positions: bool = True
        self.timeframe: Optional[str] = None
        self.symbols: Optional[List[str]] = None
        self.exchange: Optional[IsExchange] = None
        self.backtest_to_time: Optional[int] = None
        self.backtest_from_time: Optional[int] = None
        # leverage to use per trade.
        self.leverage: int = 1
        # exchange order fee as a percentage of the trade principal.
        self.commission: float = 0.05
        # expected trade slippage as a percentage of the trade principal.
        self.expected_slippage: float = 0.05
        # number of indicator values to be discarded due to being potentially unstable.
        self.unstable_indicator_values: int = 0
        # maximum number of candles for a single trade.
        self.max_holding_candles: int = 100
        # account initial balance - will be constantly updated to match current account balance.
        self.current_backtest_balance: float = 10000
        # percentage of current account balance to risk per trade.
        self.percentage_risk_per_trade: float = 2
        # maximum amount to invest per trade.
        # If `None`, the maximum amount to invest per trade will be the current account balance.
        self.max_amount_per_trade: Optional[int] = None
        # Whether to allow for multiple open positions per symbol at a time.
        self.allow_multiple_open_positions: bool = True
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config: Dict = dict()
        # backtest data that initially contains OHLCV data.
        self.backtest_data: Dict = dict()
        # backtest trade positions.
        self.trade_positions: Dict = dict()

    @staticmethod
    def generate_uuid() -> str:
        """Generate a random UUID.

        :return: str
        """

        return str(uuid.uuid4())

    @abstractmethod
    def plot_backtest_indicators(self) -> Dict:
        """Get the indicators to plot on the backtest analysis dashboard.

        :return: Dict
        """

        return dict()

    def open_long_backtest_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> None:
        """Open a long trade position for a given symbol.

        :param symbol: symbol to open a long trade position for.
        :param entry_price: price to enter the long trade.
        :param entry_time: time at which the long trade was entered.
        :param target_price: price at which the long trade should take profit.
        :param stop_price: price at which the long trade should cut losses.
        :return: None
        """

        new_position = TradePosition(
            direction=1,
            entry_price=entry_price,
            entry_time=entry_time,
            trade_count=len(self.trade_positions.get(symbol, {})) + 1,
            target_price=target_price,
            stop_price=stop_price,
        )

        if not self.trade_positions.get(symbol, dict()):
            self.trade_positions[symbol] = dict()
        self.trade_positions[symbol][Backtester.generate_uuid()] = new_position

    def open_short_backtest_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> None:
        """Open a short trade position for a given symbol.

        :param symbol: symbol to open a short trade position for.
        :param entry_price: price to enter the short trade.
        :param entry_time: time at which the short trade was entered.
        :param target_price: price at which the short trade should take profit.
        :param stop_price: price at which the short trade should cut losses.
        :return: None
        """

        new_position = TradePosition(
            direction=-1,
            entry_price=entry_price,
            entry_time=entry_time,
            trade_count=len(self.trade_positions.get(symbol, {})) + 1,
            target_price=target_price,
            stop_price=stop_price,
        )

        if not self.trade_positions.get(symbol, dict()):
            self.trade_positions[symbol] = dict()
        self.trade_positions[symbol][Backtester.generate_uuid()] = new_position

    def fetch_open_backtest_positions(self, symbol: str) -> List[TradePosition]:
        """Fetch a list of all open backtest positions for a given symbol.

        :param symbol: symbol to fetch open positions for.
        :return: List[TradePosition]
        """

        open_positions = list()
        position: TradePosition
        for (_, position) in self.trade_positions.get(symbol, dict()).items():
            if position.open_position:
                open_positions.append(position)

        return open_positions

    def close_backtest_position(
        self, symbol: str, position_id: str, close_price: float, exit_time: datetime
    ) -> TradePosition:
        """Close an open trade position for a given symbol.

        :param symbol: symbol whose position should be closed.
        :param position_id: ID of the position to close.
        :param close_price: price at which the trade exited.
        :param exit_time: time at which the trade exited.
        :return: TradePosition
        """

        position: Optional[TradePosition] = self.trade_positions[symbol].get(
            position_id, None
        )
        if not position:
            raise LookupError(
                f"Position ID {position_id} does not exist for symbol {symbol}"
            )
        if not position.open_position:
            raise ValueError(
                f"Attempting to close closed position {position_id} of symbol {symbol}"
            )

        position.exit_time = exit_time
        position.close_price = close_price
        position.initial_account_balance = self.current_backtest_balance

        roe = ((close_price / position.entry_price - 1) * position.direction) * 100.0
        position.roe = round(roe, 4)

        position.position_size = self.leverage * (
            (self.percentage_risk_per_trade / 100.0) * self.current_backtest_balance
        )
        if (
            self.max_amount_per_trade
            and position.position_size > self.max_amount_per_trade
        ):
            position.position_size = self.max_amount_per_trade

        cost_adjusted_roe = (
            position.roe - (2 * self.commission) - self.expected_slippage
        )
        position.cost_adjusted_roe = round(cost_adjusted_roe, 4)

        if self.current_backtest_balance <= 0:
            position.roe = 0
            position.position_size = 0
            position.cost_adjusted_roe = 0

        pnl = position.position_size * (cost_adjusted_roe / 100.0)
        position.pnl = round(pnl, 4)
        commission = position.position_size * ((2 * self.commission) / 100.0)
        position.commission = round(commission, 4)
        slippage = position.position_size * (self.expected_slippage / 100.0)
        position.slippage = round(slippage, 4)

        self.current_backtest_balance += position.pnl

        position.open_position = False
        position.final_account_balance = self.current_backtest_balance

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

        :param data: the historical price dataframe row at the current time in backtest.
        :return: bool
        """

        pass

    @abstractmethod
    def is_short_trade_signal_present(self, data: Any) -> bool:
        """Check if a short trade signal exists.

        :param data: the historical price dataframe row at the current time in backtest.
        :return: bool
        """

        pass

    @abstractmethod
    def generate_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> TradeLevels:
        """Generate price levels for an individual trade signal.

        :param data: the historical price dataframe row where the open trade signal was detected.
        :param trade_signal_direction: 1 for a long position. -1 for a short position.
        :return: TradeLevels
        """

        return TradeLevels(
            entry_price=data.close,
            target_price=None,
            stop_price=None,
        )

    def handle_open_backtest_positions(self, symbol: str, data: Any) -> None:
        """Monitor and handle open positions for a given symbol and close them
        if they hit a trade barrier.

        :param symbol: symbol to monitor open positions for.
        :param data: the historical price dataframe row at the current time in backtest.
        :return: None
        """

        # handle each open trade position.
        position: TradePosition
        for (position_id, position) in self.trade_positions.get(symbol, dict()).items():

            # ensure that the trade position was not opened during the current candle.
            if position.entry_time == data.Index:
                continue

            # ensure that the trade position is still open.
            if not position.open_position:
                continue

            # increment trade holding time by 1.
            position.holding_time += 1

            # check if the lower horizontal barrier has been hit for long positions.
            if (
                position.stop_price
                and data.low <= position.stop_price
                and position.direction == 1
            ):
                self.close_backtest_position(
                    symbol, position_id, position.stop_price, data.Index
                )

            # check if upper horizontal barrier has been hit for long positions.
            elif (
                position.target_price
                and data.high >= position.target_price
                and position.direction == 1
            ):
                self.close_backtest_position(
                    symbol, position_id, position.target_price, data.Index
                )

            # check if upper horizontal barrier has been hit for short positions.
            elif (
                position.stop_price
                and data.high >= position.stop_price
                and position.direction == -1
            ):
                self.close_backtest_position(
                    symbol, position_id, position.stop_price, data.Index
                )

            # check if lower horizontal barrier has been hit for short positions.
            elif (
                position.target_price
                and data.low <= position.target_price
                and position.direction == -1
            ):
                self.close_backtest_position(
                    symbol, position_id, position.target_price, data.Index
                )

            # check if vertical barrier has been hit.
            elif position.holding_time >= self.max_holding_candles:
                self.close_backtest_position(
                    symbol, position_id, data.close, data.Index
                )

            # check if current candle is the last candle in the provided historical price data.
            elif data.Index == self.backtest_data[symbol].index.values[-1]:
                self.close_backtest_position(
                    symbol, position_id, data.close, data.Index
                )

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
            self.timeframe,
            self.strategy_name,
        )

        ohlcv_db = OHLCVDatabase(self.exchange, symbol)
        ohlcv_data = ohlcv_db.get_data(
            symbol, self.backtest_from_time, self.backtest_to_time
        )
        if ohlcv_data is None:
            logger.warning(
                "%s %s %s: unable to get price data for the %s strategy",
                symbol,
                self.exchange.display_name,
                self.timeframe,
                self.strategy_name,
            )
            return None

        resampled_ohlcv_data = resample_timeframe(ohlcv_data, self.timeframe)

        # generate trading features.
        populated_ohlcv_data = self.generate_features(resampled_ohlcv_data)

        # remove unstable indicator values.
        idx = self.unstable_indicator_values
        self.backtest_data[symbol] = populated_ohlcv_data.iloc[idx:]

        for row in self.backtest_data[symbol].itertuples():
            # open a long position if we get a long trading signal.
            if (
                self.allow_long_positions
                and self.is_long_trade_signal_present(row)
                and row.Index != self.backtest_data[symbol].index.values[-1]
            ):
                # only open a position if multiple open positions are allowed or
                # there is no open position.
                if not self.allow_multiple_open_positions and len(
                    self.fetch_open_backtest_positions(symbol)
                ):
                    continue

                trade_levels = self.generate_trade_levels(row, trade_signal_direction=1)
                self.open_long_backtest_position(
                    symbol=symbol,
                    entry_price=trade_levels.entry_price,
                    entry_time=row.Index,
                    target_price=trade_levels.target_price,
                    stop_price=trade_levels.stop_price,
                )

            # open a short position if we get a short trading signal.
            if (
                self.allow_short_positions
                and self.is_short_trade_signal_present(row)
                and row.Index != self.backtest_data[symbol].index.values[-1]
            ):
                # only open a position if multiple open positions are allowed or
                # there is no open position.
                if not self.allow_multiple_open_positions and len(
                    self.fetch_open_backtest_positions(symbol)
                ):
                    continue

                trade_levels = self.generate_trade_levels(
                    row, trade_signal_direction=-1
                )
                self.open_short_backtest_position(
                    symbol=symbol,
                    entry_price=trade_levels.entry_price,
                    entry_time=row.Index,
                    target_price=trade_levels.target_price,
                    stop_price=trade_levels.stop_price,
                )

            # monitor and handle all open positions.
            self.handle_open_backtest_positions(symbol, row)

        logger.info(
            "%s %s %s: completed backtest on the %s strategy",
            symbol,
            self.exchange.display_name,
            self.timeframe,
            self.strategy_name,
        )

    def run_backtest(
        self,
        exchange: IsExchange,
        symbols: Optional[List[str]],
        timeframe: Optional[str],
        from_time: Optional[str],
        to_time: Optional[str],
    ) -> Optional[List[Dict]]:
        """Run trading backtest for multiple symbols at once and return
        analysis results on the backtest.

        :param exchange: exchange to use to run the backtest.
        :param symbols: exchange symbols to run backtest on.
        :param timeframe: timeframe to run the backtest on.
        :param from_time: desired begin time of the backtest.
        :param to_time: desired end time of the backtest.
        :return: Optional[List[Dict]]
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
        self.timeframe = timeframe
        if not self.timeframe:
            self.timeframe = self.config.get("timeframe", None)
        if not self.timeframe:
            logger.warning(
                "%s: timeframe not defined for the strategy backtest",
                self.strategy_name,
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
                "timeframe": self.timeframe,
                "exchange": self.exchange,
                "backtest_from_time": self.backtest_from_time,
                "backtest_to_time": self.backtest_to_time,
            }
        )

        max_workers = multiprocessing.cpu_count() - 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.run_symbol_backtest, self.symbols)

        # Analyze the trading strategy.
        strategy_analyzer = StrategyAnalyzer(strategy=self)
        strategy_analysis = strategy_analyzer.run_analysis()

        return strategy_analysis
