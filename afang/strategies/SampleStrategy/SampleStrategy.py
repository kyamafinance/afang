from typing import Any, Dict

import pandas as pd
from finta import TA

from afang.strategies.is_strategy import IsStrategy, TradeLevels


class SampleStrategy(IsStrategy):
    """Sample user defined strategy."""

    def __init__(self) -> None:
        """Initialize SampleStrategy class."""

        super().__init__(strategy_name="SampleStrategy")

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
        data["ema"] = TA.EMA(data, params["ema_period"], "close", True)

        # MACD.
        macd = TA.MACD(
            data,
            params["macd_period_fast"],
            params["macd_period_slow"],
            params["macd_signal"],
            "close",
            True,
        )
        data["macd"] = macd["MACD"]
        data["prev_macd"] = data["macd"].shift(1)
        data["macd_signal"] = macd["SIGNAL"]
        data["prev_macd_signal"] = data["macd_signal"].shift(1)

        # PSAR.
        working_data = data.copy()
        psar = TA.PSAR(working_data, params["psar_increment"], params["psar_max_val"])
        data["psar"] = psar["psar"]

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
