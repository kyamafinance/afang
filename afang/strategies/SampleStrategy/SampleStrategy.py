from typing import Any, Dict

import pandas as pd
import talib

from afang.strategies.is_strategy import IsStrategy, TradeLevels


class SampleStrategy(IsStrategy):
    """Sample user defined strategy."""

    def __init__(self) -> None:
        """Initialize SampleStrategy class."""

        IsStrategy.__init__(self, strategy_name="SampleStrategy")

        self.leverage = 5
        # hold an open position for a max of 48 hrs.
        self.max_holding_candles = 192
        self.max_amount_per_trade = 1000
        self.unstable_indicator_values = 300
        self.allow_multiple_open_positions = False

    def plot_backtest_indicators(self) -> Dict:
        """Get the indicators to plot on the backtest analysis dashboard.

        :return: Dict
        """

        return dict()

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the trading strategy.

        - To generate features, add columns to the `data` dataframe that can later
          be used to calculate horizontal trade barriers.
        - Initially, the `data` dataframe will contain OHLCV data.

        :param data: OHLCV data for a trading symbol.
        :return: None
        """

        params = self.config["parameters"]

        # EMA.
        data["ema"] = talib.EMA(data.close, timeperiod=params["ema_period"])

        # MACD.
        data["macd"], data["macd_signal"], _ = talib.MACD(
            data.close,
            fastperiod=params["macd_period_fast"],
            slowperiod=params["macd_period_slow"],
            signalperiod=params["macd_signal"],
        )
        data["prev_macd"] = data["macd"].shift(1)
        data["prev_macd_signal"] = data["macd_signal"].shift(1)

        # PSAR.
        data["psar"] = talib.SAR(
            data.high,
            data.low,
            acceleration=params["psar_acceleration"],
            maximum=params["psar_max_val"],
        )

        return data

    def is_long_trade_signal_present(self, data: Any) -> bool:
        """Check if a long trade signal exists.

        :param data: the historical price dataframe row at the current time in backtest.
        :return: bool
        """

        # Ensure that the candle is above the EMA.
        if not data.low > data.ema:
            return False

        # Ensure that the MACD line is crossing above the signal line.
        if not data.macd > data.macd_signal:
            return False
        if not data.prev_macd < data.prev_macd_signal:
            return False

        # Ensure that the PSAR is below the candle low.
        if not data.psar < data.low:
            return False

        return True

    def is_short_trade_signal_present(self, data: Any) -> bool:
        """Check if a short trade signal exists.

        :param data: the historical price dataframe row at the current time in backtest.
        :return: bool
        """

        # Ensure that the candle is below the EMA.
        if not data.high < data.ema:
            return False

        # Ensure that the MACD line is crossing below the signal line.
        if not data.macd < data.macd_signal:
            return False
        if not data.prev_macd > data.prev_macd_signal:
            return False

        # Ensure that the PSAR is above the candle high.
        if not data.psar > data.high:
            return False

        return True

    def generate_trade_levels(
        self, data: Any, trade_signal_direction: int
    ) -> TradeLevels:
        """Generate price levels for an individual trade signal.

        :param data: the historical price dataframe row where the open trade signal was detected.
        :param trade_signal_direction: 1 for a long position. -1 for a short position.
        :return: TradeLevels
        """

        params = self.config["parameters"]

        entry_price = data.close
        stop_price = data.psar

        if trade_signal_direction == 1:
            # long trade.
            risk = entry_price - stop_price
            target_price = entry_price + (params["RR"] * risk)
        else:
            # short trade.
            risk = stop_price - entry_price
            target_price = entry_price - (params["RR"] * risk)

        return TradeLevels(
            entry_price=entry_price,
            target_price=target_price if target_price >= 0 else 0,
            stop_price=stop_price,
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

        # ensure that psar acceleration is less than psar max value.
        # the psar acceleration and the psar max value could end up being the same value.
        # however, if this were to happen, the optimizer would discard the backtest due to
        # extremely low backtest performance results.
        psar_acceleration = min(
            parameters["psar_acceleration"], parameters["psar_max_val"]
        )
        psar_max_val = max(parameters["psar_acceleration"], parameters["psar_max_val"])

        parameters["psar_acceleration"] = psar_acceleration
        parameters["psar_max_val"] = psar_max_val

        return parameters
