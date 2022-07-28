import argparse
import datetime
import uuid
from collections import namedtuple
from typing import Any, Optional

import pandas as pd
import pytest

from afang.database.ohlcv_database import OHLCVDatabase


@pytest.fixture
def ohlcv_row() -> Any:
    trade_exit_time = pd.to_datetime("2022-01-01 01:00:05")
    OHLCVRow = namedtuple(
        "OHLCVRow", ["Index", "open", "high", "low", "close", "volume"]
    )
    ohlcv_row = OHLCVRow(trade_exit_time, 50, 200, 10, 100, 550)

    return ohlcv_row


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [7, 8, 9, 10, 11, 12],
            "low": [13, 14, 15, 16, 17, 18],
            "close": [19, 20, 21, 22, 23, 24],
            "volume": [25, 26, 27, 28, 29, 30],
        },
        index=[
            pd.to_datetime("2022-01-01 01:00:00"),
            pd.to_datetime("2022-01-01 01:00:01"),
            pd.to_datetime("2022-01-01 01:00:02"),
            pd.to_datetime("2022-01-01 01:00:03"),
            pd.to_datetime("2022-01-01 01:00:04"),
            pd.to_datetime("2022-01-01 01:00:05"),
        ],
    )


@pytest.fixture
def ohlcv_db(ohlcv_root_db_dir, ohlcv_df) -> OHLCVDatabase:
    class DummyOHLCVDatabase(OHLCVDatabase):
        def get_data(
            self, symbol: str, from_time: int, to_time: int
        ) -> Optional[pd.DataFrame]:
            return pd.DataFrame(
                {
                    "open": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
                    "high": [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12],
                    "low": [13, 14, 15, 16, 17, 18, 13, 14, 15, 16, 17, 18],
                    "close": [19, 20, 21, 22, 23, 24, 19, 20, 21, 22, 23, 24],
                    "volume": [25, 26, 27, 28, 29, 30, 25, 26, 27, 28, 29, 30],
                },
                index=[
                    pd.to_datetime("2022-01-01 01:00:00"),
                    pd.to_datetime("2022-01-01 01:01:00"),
                    pd.to_datetime("2022-01-01 01:02:00"),
                    pd.to_datetime("2022-01-01 01:03:00"),
                    pd.to_datetime("2022-01-01 01:04:00"),
                    pd.to_datetime("2022-01-01 01:05:00"),
                    pd.to_datetime("2022-01-01 01:06:00"),
                    pd.to_datetime("2022-01-01 01:07:00"),
                    pd.to_datetime("2022-01-01 01:08:00"),
                    pd.to_datetime("2022-01-01 01:09:00"),
                    pd.to_datetime("2022-01-01 01:10:00"),
                    pd.to_datetime("2022-01-01 01:11:00"),
                ],
            )

    return DummyOHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)


def test_generate_uuid(dummy_is_strategy) -> None:
    generated_uuid = dummy_is_strategy.generate_uuid()
    uuid.UUID(str(generated_uuid))


def test_open_long_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.generate_uuid", side_effect=[1, 2, 3]
    )

    trade_entry_time = datetime.datetime(2022, 1, 1, 1, 0)
    dummy_is_strategy.open_long_backtest_position(
        "test_symbol", 10, trade_entry_time, 100, 5
    )
    dummy_is_strategy.open_long_backtest_position(
        "test_symbol", 10, trade_entry_time, None, None
    )
    dummy_is_strategy.open_long_backtest_position(
        "test_symbol_2", 87, trade_entry_time, 890, None
    )

    open_trades = dummy_is_strategy.backtest_positions

    assert open_trades["test_symbol"] == {
        1: {
            "open_position": True,
            "direction": 1,
            "entry_price": 10,
            "entry_time": trade_entry_time,
            "target_price": 100,
            "stop_price": 5,
            "holding_time": 0,
            "trade_count": 1,
        },
        2: {
            "open_position": True,
            "direction": 1,
            "entry_price": 10,
            "entry_time": trade_entry_time,
            "target_price": None,
            "stop_price": None,
            "holding_time": 0,
            "trade_count": 2,
        },
    }
    assert open_trades["test_symbol_2"] == {
        3: {
            "open_position": True,
            "direction": 1,
            "entry_price": 87,
            "entry_time": trade_entry_time,
            "target_price": 890,
            "stop_price": None,
            "holding_time": 0,
            "trade_count": 1,
        }
    }


def test_open_short_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.generate_uuid", side_effect=[1, 2, 3]
    )

    trade_entry_time = datetime.datetime(2022, 1, 1, 1, 0)
    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 100, trade_entry_time, 10, 1000
    )
    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 100, trade_entry_time, None, None
    )
    dummy_is_strategy.open_short_backtest_position(
        "test_symbol_2", 890, trade_entry_time, 87, None
    )

    open_trades = dummy_is_strategy.backtest_positions

    assert open_trades["test_symbol"] == {
        1: {
            "open_position": True,
            "direction": -1,
            "entry_price": 100,
            "entry_time": trade_entry_time,
            "target_price": 10,
            "stop_price": 1000,
            "holding_time": 0,
            "trade_count": 1,
        },
        2: {
            "open_position": True,
            "direction": -1,
            "entry_price": 100,
            "entry_time": trade_entry_time,
            "target_price": None,
            "stop_price": None,
            "holding_time": 0,
            "trade_count": 2,
        },
    }
    assert open_trades["test_symbol_2"] == {
        3: {
            "open_position": True,
            "direction": -1,
            "entry_price": 890,
            "entry_time": trade_entry_time,
            "target_price": 87,
            "stop_price": None,
            "holding_time": 0,
            "trade_count": 1,
        }
    }


def test_fetch_open_backtest_positions(mocker, dummy_is_strategy) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.generate_uuid", side_effect=[1, 2, 3]
    )

    trade_entry_time = datetime.datetime(2022, 1, 1, 1, 0)
    dummy_is_strategy.open_long_backtest_position(
        "test_symbol", 10, trade_entry_time, 100, 5
    )
    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 100, trade_entry_time, 10, 1000
    )
    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 890, trade_entry_time, 87, None
    )

    dummy_is_strategy.backtest_positions["test_symbol"][3]["open_position"] = False
    open_positions = dummy_is_strategy.fetch_open_backtest_positions("test_symbol")

    assert len(open_positions) == 2
    assert open_positions[0]["entry_price"] == 10
    assert open_positions[1]["entry_price"] == 100


def test_close_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.generate_uuid",
        side_effect=["1", "2", "3"],
    )

    trade_entry_time = datetime.datetime(2022, 1, 1, 1, 0)
    trade_exit_time = datetime.datetime(2022, 1, 1, 2, 0)

    dummy_is_strategy.open_long_backtest_position(
        "test_symbol", 100, trade_entry_time, 150, 50
    )
    dummy_is_strategy.close_backtest_position("test_symbol", "1", 150, trade_exit_time)

    # set a max amount per trade.
    dummy_is_strategy.max_amount_per_trade = 100

    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 100, trade_entry_time, 50, 150
    )
    dummy_is_strategy.close_backtest_position("test_symbol", "2", 150, trade_exit_time)

    # set current account balance to <=0.
    dummy_is_strategy.current_backtest_balance = -1

    dummy_is_strategy.open_short_backtest_position(
        "test_symbol", 100, trade_entry_time, 50, 150
    )
    dummy_is_strategy.close_backtest_position("test_symbol", "3", 150, trade_exit_time)

    assert dummy_is_strategy.backtest_positions["test_symbol"]["1"] == {
        "open_position": False,
        "direction": 1,
        "entry_price": 100,
        "entry_time": trade_entry_time,
        "target_price": 150,
        "stop_price": 50,
        "holding_time": 0,
        "trade_count": 1,
        "exit_time": trade_exit_time,
        "close_price": 150,
        "initial_account_balance": 10000,
        "roe": 50.0,
        "position_size": 200.0,
        "cost_adjusted_roe": 49.85,
        "pnl": 99.7,
        "commission": 0.2,
        "slippage": 0.1,
        "final_account_balance": 10099.7,
    }

    assert dummy_is_strategy.backtest_positions["test_symbol"]["2"] == {
        "open_position": False,
        "direction": -1,
        "entry_price": 100,
        "entry_time": trade_entry_time,
        "target_price": 50,
        "stop_price": 150,
        "holding_time": 0,
        "trade_count": 2,
        "exit_time": trade_exit_time,
        "close_price": 150,
        "initial_account_balance": 10099.7,
        "roe": -50.0,
        "position_size": 100,
        "cost_adjusted_roe": -50.15,
        "pnl": -50.15,
        "commission": 0.1,
        "slippage": 0.05,
        "final_account_balance": 10049.550000000001,
    }

    assert dummy_is_strategy.backtest_positions["test_symbol"]["3"] == {
        "open_position": False,
        "direction": -1,
        "entry_price": 100,
        "entry_time": trade_entry_time,
        "target_price": 50,
        "stop_price": 150,
        "holding_time": 0,
        "trade_count": 3,
        "exit_time": trade_exit_time,
        "close_price": 150,
        "initial_account_balance": -1,
        "roe": 0,
        "position_size": 0,
        "cost_adjusted_roe": 0,
        "pnl": -0.0,
        "commission": 0.0,
        "slippage": 0.0,
        "final_account_balance": -1.0,
    }


@pytest.mark.parametrize(
    "entry_price, target_price, stop_price, direction, max_holding_candles, expected_close_price",
    [
        (80, 100, 15, 1, 1, 15),
        (80, 150, 8, 1, 1, 150),
        (80, 50, 200, -1, 1, 200),
        (80, 11, 210, -1, 1, 11),
        (80, 8, 210, -1, 1, 100),
        (80, 8, 210, -1, 2, 100),
    ],
)
def test_handle_open_backtest_positions(
    mocker,
    dummy_is_strategy,
    ohlcv_row,
    ohlcv_df,
    entry_price,
    target_price,
    stop_price,
    direction,
    max_holding_candles,
    expected_close_price,
) -> None:
    dummy_is_strategy.backtest_data["test_symbol"] = ohlcv_df
    dummy_is_strategy.max_holding_candles = max_holding_candles

    mocker.patch(
        "afang.strategies.backtester.Backtester.generate_uuid",
        side_effect=["1"],
    )
    mocked_close_backtest_position = mocker.patch(
        "afang.strategies.backtester.Backtester.close_backtest_position",
        return_value=None,
    )

    trade_entry_time = datetime.datetime(2022, 1, 1, 1, 0)
    if direction == 1:
        dummy_is_strategy.open_long_backtest_position(
            "test_symbol", entry_price, trade_entry_time, target_price, stop_price
        )
    else:
        dummy_is_strategy.open_short_backtest_position(
            "test_symbol", entry_price, trade_entry_time, target_price, stop_price
        )

    dummy_is_strategy.handle_open_backtest_positions("test_symbol", ohlcv_row)

    assert mocked_close_backtest_position.assert_called
    assert mocked_close_backtest_position.call_args.args[0] == "test_symbol"
    assert mocked_close_backtest_position.call_args.args[1] == "1"
    assert mocked_close_backtest_position.call_args.args[2] == expected_close_price


def test_run_symbol_backtest(
    mocker, dummy_is_exchange, dummy_is_strategy, ohlcv_db
) -> None:
    mocker.patch("afang.strategies.backtester.OHLCVDatabase", return_value=ohlcv_db)

    mocked_generate_features = mocker.patch(
        "afang.strategies.backtester.Backtester.generate_features"
    )
    mocked_is_long_trade_signal_present = mocker.patch(
        "afang.strategies.backtester.Backtester.is_long_trade_signal_present"
    )
    mocked_generate_trade_levels = mocker.patch(
        "afang.strategies.backtester.Backtester.generate_trade_levels"
    )
    mocked_open_long_backtest_position = mocker.patch(
        "afang.strategies.backtester.Backtester.open_long_backtest_position"
    )
    mocked_is_short_trade_signal_present = mocker.patch(
        "afang.strategies.backtester.Backtester.is_short_trade_signal_present"
    )
    mocked_fetch_open_backtest_positions = mocker.patch(
        "afang.strategies.backtester.Backtester.fetch_open_backtest_positions"
    )
    mocked_open_short_backtest_position = mocker.patch(
        "afang.strategies.backtester.Backtester.open_short_backtest_position"
    )
    mocked_handle_open_backtest_positions = mocker.patch(
        "afang.strategies.backtester.Backtester.handle_open_backtest_positions"
    )

    dummy_is_strategy.unstable_indicator_values = 1
    dummy_is_strategy.run_symbol_backtest(
        "test_symbol", dummy_is_exchange, "5m", 0, 1000
    )

    assert mocked_generate_features.assert_called
    assert mocked_is_long_trade_signal_present.assert_called
    assert mocked_fetch_open_backtest_positions.assert_called
    assert mocked_generate_trade_levels.assert_called
    assert mocked_open_long_backtest_position.assert_called
    assert mocked_is_short_trade_signal_present.assert_called
    assert mocked_open_short_backtest_position.assert_called
    assert mocked_handle_open_backtest_positions.assert_called
    assert len(dummy_is_strategy.backtest_data["test_symbol"].index) == 2


def test_run_backtest(mocker, dummy_is_exchange, dummy_is_strategy) -> None:
    mock_run_symbol_backtest = mocker.patch(
        "afang.strategies.backtester.Backtester.run_symbol_backtest", return_value=None
    )

    cli_args = argparse.Namespace(symbols=[], timeframe="", from_time=0, to_time=0)
    dummy_is_strategy.run_backtest(dummy_is_exchange, cli_args)

    assert mock_run_symbol_backtest.assert_called
