import argparse
import logging
import multiprocessing
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from afang.database.ohlcv_database import OHLCVDatabase
from afang.exchanges import IsExchange
from afang.strategies.util import TradeLevels
from afang.utils.util import resample_timeframe, time_str_to_milliseconds

logger = logging.getLogger(__name__)


class Backtester(ABC):
    """Base interface for strategy backtests."""

    @abstractmethod
    def __init__(self, strategy_name: str) -> None:
        """Initialize Backtester class.

        :param strategy_name: name of the trading strategy.
        """

        self.strategy_name = strategy_name
        self.allow_long_positions = True
        self.allow_short_positions = True
        # leverage to use per trade.
        self.leverage = 1
        # exchange order fee as a percentage of the trade principal.
        self.commission = 0.05
        # expected trade slippage as a percentage of the trade principal.
        self.expected_slippage = 0.05
        # number of indicator values to be discarded due to being potentially unstable.
        self.unstable_indicator_values = 0
        # maximum number of candles for a single trade.
        self.max_holding_candles = 100
        # account initial balance - will be constantly updated to match current account balance.
        self.current_backtest_balance = 10000
        # percentage of current account balance to risk per trade.
        self.percentage_risk_per_trade = 2
        # maximum amount to invest per trade.
        # If `None`, the maximum amount to invest per trade will be the current account balance.
        self.max_amount_per_trade = None
        # Whether to allow for multiple open positions per symbol at a time.
        self.allow_multiple_open_positions = True
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config: Dict = dict()
        # backtest data that initially contains OHLCV data.
        self.backtest_data: Dict = dict()
        # backtest positions.
        self.backtest_positions: Dict = dict()

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

        new_position = {
            "open_position": True,
            "direction": 1,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "target_price": target_price,
            "stop_price": stop_price,
            "holding_time": 0,
            "trade_count": len(self.backtest_positions.get(symbol, {})) + 1,
        }

        if symbol in self.backtest_positions:
            self.backtest_positions[symbol][Backtester.generate_uuid()] = new_position
        else:
            self.backtest_positions[symbol] = {Backtester.generate_uuid(): new_position}

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

        new_position = {
            "open_position": True,
            "direction": -1,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "target_price": target_price,
            "stop_price": stop_price,
            "holding_time": 0,
            "trade_count": len(self.backtest_positions.get(symbol, {})) + 1,
        }

        if symbol in self.backtest_positions:
            self.backtest_positions[symbol][Backtester.generate_uuid()] = new_position
        else:
            self.backtest_positions[symbol] = {Backtester.generate_uuid(): new_position}

    def fetch_open_backtest_positions(self, symbol: str) -> List[Dict]:
        """Fetch a list of all open backtest positions for a given symbol.

        :param symbol: symbol to fetch open positions for.
        :return: List[Dict]
        """

        open_positions = list()
        for (_, position) in self.backtest_positions.get(symbol, dict()).items():
            if position.get("open_position"):
                open_positions.append(position)

        return open_positions

    def close_backtest_position(
        self, symbol: str, position_id: str, close_price: float, exit_time: datetime
    ) -> None:
        """Close an open trade position for a given symbol.

        :param symbol: symbol whose position should be closed.
        :param position_id: ID of the position to close.
        :param close_price: price at which the trade exited.
        :param exit_time: time at which the trade exited.
        :return: None
        """

        position = self.backtest_positions[symbol].get(position_id)

        position["exit_time"] = exit_time
        position["close_price"] = close_price
        position["initial_account_balance"] = self.current_backtest_balance

        roe = (
            (close_price / position.get("entry_price") - 1) * position.get("direction")
        ) * 100.0
        position["roe"] = round(roe, 4)

        position["position_size"] = self.leverage * (
            (self.percentage_risk_per_trade / 100.0) * self.current_backtest_balance
        )
        if (
            self.max_amount_per_trade
            and position["position_size"] > self.max_amount_per_trade
        ):
            position["position_size"] = self.max_amount_per_trade

        cost_adjusted_roe = (
            position["roe"] - (2 * self.commission) - self.expected_slippage
        )
        position["cost_adjusted_roe"] = round(cost_adjusted_roe, 4)

        if self.current_backtest_balance <= 0:
            position["roe"] = 0
            position["position_size"] = 0
            position["cost_adjusted_roe"] = 0

        pnl = position["position_size"] * (cost_adjusted_roe / 100.0)
        position["pnl"] = round(pnl, 4)
        commission = position["position_size"] * ((2 * self.commission) / 100.0)
        position["commission"] = round(commission, 4)
        slippage = position["position_size"] * (self.expected_slippage / 100.0)
        position["slippage"] = round(slippage, 4)

        self.current_backtest_balance += position["pnl"]

        position["open_position"] = False
        position["final_account_balance"] = self.current_backtest_balance

    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> None:
        """Generate features for the trading strategy.

        - To generate features, add columns to the `data` dataframe that can later
          be used to calculate horizontal trade barriers.
        - Initially, the `data` dataframe will contain OHLCV data.

        :param data: OHLCV data for a trading symbol.
        :return: None
        """

        pass

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

        # Get the IDs of all the open trade positions.
        open_position_ids = list()
        for (position_id, position) in self.backtest_positions.get(
            symbol, dict()
        ).items():
            if position.get("open_position"):
                open_position_ids.append(position_id)

        # handle each open trade position.
        for open_position_id in open_position_ids:
            open_position = self.backtest_positions[symbol].get(open_position_id)

            # ensure that the trade position is still open.
            if not open_position.get("open_position"):
                continue

            # increment trade holding time by 1.
            open_position["holding_time"] += 1

            # check if the lower horizontal barrier has been hit for long positions.
            if (
                open_position["stop_price"]
                and data.low <= open_position["stop_price"]
                and open_position["direction"] == 1
            ):
                self.close_backtest_position(
                    symbol, open_position_id, open_position["stop_price"], data.Index
                )

            # check if upper horizontal barrier has been hit for long positions.
            elif (
                open_position["target_price"]
                and data.high >= open_position["target_price"]
                and open_position["direction"] == 1
            ):
                self.close_backtest_position(
                    symbol, open_position_id, open_position["target_price"], data.Index
                )

            # check if upper horizontal barrier has been hit for short positions.
            elif (
                open_position["stop_price"]
                and data.high >= open_position["stop_price"]
                and open_position["direction"] == -1
            ):
                self.close_backtest_position(
                    symbol, open_position_id, open_position["stop_price"], data.Index
                )

            # check if lower horizontal barrier has been hit for short positions.
            elif (
                open_position["target_price"]
                and data.low <= open_position["target_price"]
                and open_position["direction"] == -1
            ):
                self.close_backtest_position(
                    symbol, open_position_id, open_position["target_price"], data.Index
                )

            # check if vertical barrier has been hit.
            elif open_position["holding_time"] >= self.max_holding_candles:
                self.close_backtest_position(
                    symbol, open_position_id, data.close, data.Index
                )

            # check if current candle is the last candle in the provided historical price data.
            elif data.Index == self.backtest_data[symbol].index.values[-1]:
                self.close_backtest_position(
                    symbol, open_position_id, data.close, data.Index
                )

    def run_symbol_backtest(
        self,
        symbol: str,
        exchange: IsExchange,
        timeframe: str,
        backtest_from_time: int,
        backtest_to_time: int,
    ) -> None:
        """Run trading backtest for a single symbol.

        :param symbol: symbol to run backtest for.
        :param exchange: exchange being used.
        :param timeframe: backtesting timeframe.
        :param backtest_from_time: timestamp in ms of backtest begin date.
        :param backtest_to_time: timestamp in ms of backtest end date.
        :return: None
        """

        logger.info(
            "%s %s %s: started backtest on the %s strategy",
            symbol,
            exchange.name,
            timeframe,
            self.strategy_name,
        )

        ohlcv_db = OHLCVDatabase(None, exchange.name, symbol)
        ohlcv_data = ohlcv_db.get_data(symbol, backtest_from_time, backtest_to_time)
        resampled_ohlcv_data = resample_timeframe(ohlcv_data, timeframe)
        self.backtest_data[symbol] = resampled_ohlcv_data

        # generate trading features.
        self.generate_features(self.backtest_data[symbol])

        # remove unstable indicator values.
        idx = self.unstable_indicator_values
        self.backtest_data[symbol] = self.backtest_data[symbol].iloc[idx:]

        for row in self.backtest_data[symbol].itertuples():
            # open a long position if we get a long trading signal.
            if self.allow_long_positions and self.is_long_trade_signal_present(row):
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
            elif self.allow_short_positions and self.is_short_trade_signal_present(row):
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
            else:
                self.handle_open_backtest_positions(symbol, row)

        logger.info(
            "%s %s %s: completed backtest on the %s strategy",
            symbol,
            exchange.name,
            timeframe,
            self.strategy_name,
        )

    def run_backtest(self, exchange: IsExchange, cli_args: argparse.Namespace) -> None:
        """Run trading backtest for multiple symbols at once.

        :param exchange: exchange to use to run the backtest.
        :param cli_args: command line arguments.
        :return: None
        """

        # Get symbols to backtest.
        symbols = cli_args.symbols
        if not symbols:
            symbols = self.config.get("watchlist", dict()).get(exchange.name, [])
        if not symbols:
            logger.warning(
                "%s: no symbols found to run strategy backtest", self.strategy_name
            )
            return None

        # Get the backtesting timeframe.
        timeframe = cli_args.timeframe
        if not timeframe:
            timeframe = self.config.get("timeframe", None)
        if not timeframe:
            logger.warning(
                "%s: timeframe not defined for the strategy backtest",
                self.strategy_name,
            )
            return None

        # Get the backtesting duration.
        backtest_from_time = 0
        if cli_args.from_time:
            backtest_from_time = time_str_to_milliseconds(cli_args.from_time)
        backtest_to_time = int(time.time()) * 1000
        if cli_args.to_time:
            backtest_to_time = time_str_to_milliseconds(cli_args.to_time)

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for symbol in symbols:
            if symbol not in exchange.symbols:
                logger.warning(
                    "%s %s: provided symbol not present in the exchange",
                    exchange.name,
                    symbol,
                )
                continue

            pool.apply_async(
                self.run_symbol_backtest,
                (symbol, exchange, timeframe, backtest_from_time, backtest_to_time),
            )

        pool.close()
        pool.join()
