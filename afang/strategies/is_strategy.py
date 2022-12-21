import logging
import pathlib
from abc import abstractmethod
from typing import Any, Dict, List

import pandas as pd
import yaml

from afang.strategies.backtester import Backtester
from afang.strategies.models import TradeLevels
from afang.strategies.trader import Trader

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

        self.strategy_name = strategy_name
        self.allow_long_positions = True
        self.allow_short_positions = True
        # leverage to use per trade.
        self.leverage = 1
        # maximum number of candles for a single trade.
        self.max_holding_candles = 100
        # percentage of current account balance to risk per trade.
        self.percentage_risk_per_trade = 2
        # maximum amount to invest per trade.
        # If `None`, the maximum amount to invest per trade will be the current account balance.
        self.max_amount_per_trade = None
        # Whether to allow for multiple open positions per symbol at a time.
        self.allow_multiple_open_positions = True
        # strategy configuration parameters i.e. contents of strategy `config.yaml`.
        self.config = self.read_strategy_config()

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

    def define_optimization_param_constraints(self, parameters: Dict) -> Dict:
        """Define constraints that should be applied during backtest parameter
        generation while optimizing the strategy. Should return a dict that
        contains possible mutated parameters.

        :param parameters: parameters generated for strategy optimization. These parameters
        will follow the specification provided in `config.yaml`. This dict will not contain parameters
        that are not to be optimized.

        :return: Dict
        """

        pass
