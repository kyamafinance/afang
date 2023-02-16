import datetime
from collections import namedtuple
from typing import Any, Optional

import pandas as pd
import pytest

import afang.strategies.backtester as backtester
from afang.database.ohlcv_db.ohlcv_database import OHLCVDatabase
from afang.exchanges.models import Symbol
from afang.models import Timeframe
from afang.strategies.models import TradePosition


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
def ohlcv_db_no_data(ohlcv_root_db_dir, dummy_is_exchange, ohlcv_df) -> OHLCVDatabase:
    class DummyOHLCVDatabase(OHLCVDatabase):
        def get_data(
            self, symbol: str, from_time: int, to_time: int
        ) -> Optional[pd.DataFrame]:
            return None

    return DummyOHLCVDatabase(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)


@pytest.fixture
def ohlcv_db(ohlcv_root_db_dir, dummy_is_exchange, ohlcv_df) -> OHLCVDatabase:
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

    return DummyOHLCVDatabase(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)


def test_open_long_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch.object(backtester, "generate_uuid", side_effect=[1, 2, 3])

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

    open_trades = dummy_is_strategy.trade_positions

    assert open_trades["test_symbol"] == {
        1: TradePosition(
            open_position=True,
            direction=1,
            entry_price=10,
            entry_time=trade_entry_time,
            target_price=100,
            stop_price=5,
            holding_time=0,
            trade_count=1,
        ),
        2: TradePosition(
            open_position=True,
            direction=1,
            entry_price=10,
            entry_time=trade_entry_time,
            target_price=None,
            stop_price=None,
            holding_time=0,
            trade_count=2,
        ),
    }

    assert open_trades["test_symbol_2"] == {
        3: TradePosition(
            open_position=True,
            direction=1,
            entry_price=87,
            entry_time=trade_entry_time,
            target_price=890,
            stop_price=None,
            holding_time=0,
            trade_count=1,
        )
    }


def test_open_short_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch.object(backtester, "generate_uuid", side_effect=[1, 2, 3])

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

    open_trades = dummy_is_strategy.trade_positions

    assert open_trades["test_symbol"] == {
        1: TradePosition(
            open_position=True,
            direction=-1,
            entry_price=100,
            entry_time=trade_entry_time,
            target_price=10,
            stop_price=1000,
            holding_time=0,
            trade_count=1,
        ),
        2: TradePosition(
            open_position=True,
            direction=-1,
            entry_price=100,
            entry_time=trade_entry_time,
            target_price=None,
            stop_price=None,
            holding_time=0,
            trade_count=2,
        ),
    }

    assert open_trades["test_symbol_2"] == {
        3: TradePosition(
            open_position=True,
            direction=-1,
            entry_price=890,
            entry_time=trade_entry_time,
            target_price=87,
            stop_price=None,
            holding_time=0,
            trade_count=1,
        )
    }


def test_fetch_open_backtest_positions(mocker, dummy_is_strategy) -> None:
    mocker.patch.object(backtester, "generate_uuid", side_effect=[1, 2, "3"])

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

    dummy_is_strategy.close_backtest_position("test_symbol", "3", 50, trade_entry_time)

    open_positions = dummy_is_strategy.fetch_open_symbol_backtest_positions(
        "test_symbol"
    )

    assert len(open_positions) == 2
    assert open_positions[0].entry_price == 10
    assert open_positions[1].entry_price == 100


def test_close_backtest_position(mocker, dummy_is_strategy) -> None:
    mocker.patch.object(backtester, "generate_uuid", side_effect=["1", "2", "3"])

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

    assert dummy_is_strategy.trade_positions["test_symbol"]["1"] == TradePosition(
        open_position=False,
        direction=1,
        entry_price=100,
        entry_time=trade_entry_time,
        target_price=150,
        stop_price=50,
        holding_time=0,
        trade_count=1,
        exit_time=trade_exit_time,
        close_price=150,
        initial_account_balance=10000,
        roe=50.0,
        position_size=200.0,
        cost_adjusted_roe=49.85,
        pnl=99.7,
        commission=0.2,
        slippage=0.1,
        final_account_balance=10099.7,
    )

    assert dummy_is_strategy.trade_positions["test_symbol"]["2"] == TradePosition(
        open_position=False,
        direction=-1,
        entry_price=100,
        entry_time=trade_entry_time,
        target_price=50,
        stop_price=150,
        holding_time=0,
        trade_count=2,
        exit_time=trade_exit_time,
        close_price=150,
        initial_account_balance=10099.7,
        roe=-50.0,
        position_size=100,
        cost_adjusted_roe=-50.15,
        pnl=-50.15,
        commission=0.1,
        slippage=0.05,
        final_account_balance=10049.550000000001,
    )

    assert dummy_is_strategy.trade_positions["test_symbol"]["3"] == TradePosition(
        open_position=False,
        direction=-1,
        entry_price=100,
        entry_time=trade_entry_time,
        target_price=50,
        stop_price=150,
        holding_time=0,
        trade_count=3,
        exit_time=trade_exit_time,
        close_price=150,
        initial_account_balance=-1,
        roe=0,
        position_size=0,
        cost_adjusted_roe=0,
        pnl=-0.0,
        commission=0.0,
        slippage=0.0,
        final_account_balance=-1.0,
    )


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

    mocker.patch.object(backtester, "generate_uuid", side_effect=["1"])
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

    test_symbol_position: TradePosition = dummy_is_strategy.trade_positions[
        "test_symbol"
    ]["1"]
    assert test_symbol_position.holding_time == 1


def test_handle_open_backtest_positions_same_candle(
    dummy_is_strategy, ohlcv_row
) -> None:
    dummy_is_strategy.backtest_data["test_symbol"] = ohlcv_df
    dummy_is_strategy.max_holding_candles = 2
    dummy_is_strategy.open_long_backtest_position(
        "test_symbol", 1, ohlcv_row.Index, 1, 1
    )
    dummy_is_strategy.handle_open_backtest_positions("test_symbol", ohlcv_row)

    assert (
        len(dummy_is_strategy.fetch_open_symbol_backtest_positions("test_symbol")) == 1
    )


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
        "afang.strategies.backtester.Backtester.fetch_open_symbol_backtest_positions"
    )
    mocked_open_short_backtest_position = mocker.patch(
        "afang.strategies.backtester.Backtester.open_short_backtest_position"
    )
    mocked_handle_open_backtest_positions = mocker.patch(
        "afang.strategies.backtester.Backtester.handle_open_backtest_positions"
    )

    dummy_is_exchange.exchange_symbols["test_symbol"] = Symbol(
        name="test_symbol",
        base_asset="test",
        quote_asset="symbol",
        price_decimals=2,
        quantity_decimals=2,
        tick_size=2,
        step_size=2,
    )
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.timeframe = Timeframe.M5
    dummy_is_strategy.backtest_from_time = 0
    dummy_is_strategy.backtest_to_time = 1000
    dummy_is_strategy.unstable_indicator_values = 1
    dummy_is_strategy.run_symbol_backtest("test_symbol")

    assert mocked_generate_features.assert_called
    assert mocked_is_long_trade_signal_present.assert_called
    assert mocked_fetch_open_backtest_positions.assert_called
    assert mocked_generate_trade_levels.assert_called
    assert mocked_open_long_backtest_position.assert_called
    assert mocked_is_short_trade_signal_present.assert_called
    assert mocked_open_short_backtest_position.assert_called
    assert mocked_handle_open_backtest_positions.assert_called
    assert len(dummy_is_strategy.backtest_data["test_symbol"].index) == 2


def test_run_symbol_backtest_symbol_not_in_exchange(
    caplog, dummy_is_exchange, dummy_is_strategy
) -> None:
    dummy_is_strategy.timeframe = Timeframe.M5
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.run_symbol_backtest("test_symbol")

    assert caplog.records[-1].levelname == "ERROR"
    assert "provided symbol not present in the exchange" in caplog.text


def test_run_symbol_backtest_no_ohlcv_data(
    mocker, caplog, dummy_is_strategy, dummy_is_exchange, ohlcv_db_no_data
) -> None:
    mocker.patch(
        "afang.strategies.backtester.OHLCVDatabase", return_value=ohlcv_db_no_data
    )

    dummy_is_exchange.exchange_symbols["test_symbol"] = Symbol(
        name="test_symbol",
        base_asset="test",
        quote_asset="symbol",
        price_decimals=2,
        quantity_decimals=2,
        tick_size=2,
        step_size=2,
    )
    dummy_is_strategy.timeframe = Timeframe.M5
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.run_symbol_backtest("test_symbol")

    assert caplog.records[-1].levelname == "WARNING"
    assert (
        "test_symbol test_exchange 5m: unable to get price data for the test_strategy strategy"
        in caplog.text
    )


@pytest.mark.parametrize(
    "timeframe, expected_timeframe", [(None, "1h"), ("5m", "5m"), ("invalid", None)]
)
def test_run_backtest(
    mocker, dummy_is_exchange, dummy_is_strategy, timeframe, expected_timeframe, caplog
) -> None:
    mock_run_symbol_backtest = mocker.patch(
        "afang.strategies.backtester.Backtester.run_symbol_backtest", return_value=None
    )
    mock_run_analysis = mocker.patch(
        "afang.strategies.backtester.StrategyAnalyzer.run_analysis", return_value=None
    )

    dummy_is_strategy.run_backtest(
        dummy_is_exchange, [], timeframe, "2021-01-01", "2022-02-02"
    )

    assert mock_run_symbol_backtest.assert_called
    assert mock_run_analysis.assert_called

    if not expected_timeframe:
        assert caplog.records[0].levelname == "WARNING"
        assert (
            "test_strategy: invalid timeframe invalid defined for the strategy backtest"
            in caplog.text
        )
        return

    assert dummy_is_strategy.config["timeframe"] == expected_timeframe
    assert list(dummy_is_strategy.config.keys()) == [
        "name",
        "timeframe",
        "watchlist",
        "parameters",
        "optimizer",
        "exchange",
        "backtest_from_time",
        "backtest_to_time",
    ]
