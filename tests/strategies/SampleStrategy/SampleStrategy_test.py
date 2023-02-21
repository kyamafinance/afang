from collections import namedtuple

import pandas as pd
import pytest

from afang.strategies.models import TradeLevels
from afang.strategies.SampleStrategy.SampleStrategy import SampleStrategy


@pytest.fixture
def sample_strategy(mocker) -> SampleStrategy:
    mocker.patch(
        "afang.strategies.SampleStrategy.SampleStrategy.SampleStrategy.read_strategy_config",
        return_value={
            "name": "test_strategy",
            "timeframe": "1h",
            "watchlist": {"test_exchange": ["test_symbol"]},
            "parameters": {
                "RR": 1.5,
                "ema_period": 200,
                "macd_signal": 9,
                "macd_period_fast": 12,
                "macd_period_slow": 26,
                "psar_max_val": 0.2,
                "psar_acceleration": 0.02,
            },
        },
    )
    return SampleStrategy()


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "high": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "low": [
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
            ],
            "close": [
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ],
            "volume": [
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
            ],
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


def test_sample_strategy_params(sample_strategy) -> None:
    assert sample_strategy.leverage == 5
    assert sample_strategy.max_holding_candles == 192
    assert sample_strategy.max_amount_per_trade == 1000
    assert sample_strategy.unstable_indicator_values == 300
    assert not sample_strategy.allow_multiple_open_positions


def test_generate_features(sample_strategy, ohlcv_df) -> None:
    expected_df = sample_strategy.generate_features(ohlcv_df)
    expected_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema",
        "macd",
        "prev_macd",
        "macd_signal",
        "prev_macd_signal",
        "psar",
    ]
    assert all(x in list(expected_df.columns) for x in expected_columns)


@pytest.mark.parametrize(
    "low, ema, macd, macd_signal, prev_macd, prev_macd_signal, psar, expected_signal",
    [
        (25, 50, 200, 10, 100, 550, 90, False),
        (50, 25, 10, 20, 100, 550, 90, False),
        (50, 25, 20, 10, 550, 110, 90, False),
        (50, 25, 20, 10, 110, 550, 90, False),
        (50, 25, 20, 10, 110, 550, 10, True),
    ],
)
def test_is_long_trade_signal_present(
    sample_strategy,
    low,
    ema,
    macd,
    macd_signal,
    prev_macd,
    prev_macd_signal,
    psar,
    expected_signal,
) -> None:
    OHLCVRow = namedtuple(
        "OHLCVRow",
        ["low", "ema", "macd", "macd_signal", "prev_macd", "prev_macd_signal", "psar"],
    )
    ohlcv_row = OHLCVRow(low, ema, macd, macd_signal, prev_macd, prev_macd_signal, psar)

    is_signal_expected = sample_strategy.is_long_trade_signal_present(ohlcv_row)
    assert is_signal_expected == expected_signal


@pytest.mark.parametrize(
    "high, ema, macd, macd_signal, prev_macd, prev_macd_signal, psar, expected_signal",
    [
        (50, 25, 200, 10, 100, 550, 90, False),
        (25, 50, 20, 10, 100, 550, 90, False),
        (25, 50, 10, 20, 100, 550, 90, False),
        (25, 50, 10, 20, 550, 100, 10, False),
        (25, 50, 10, 20, 550, 100, 90, True),
    ],
)
def test_is_short_trade_signal_present(
    sample_strategy,
    high,
    ema,
    macd,
    macd_signal,
    prev_macd,
    prev_macd_signal,
    psar,
    expected_signal,
) -> None:
    OHLCVRow = namedtuple(
        "OHLCVRow",
        ["high", "ema", "macd", "macd_signal", "prev_macd", "prev_macd_signal", "psar"],
    )
    ohlcv_row = OHLCVRow(
        high, ema, macd, macd_signal, prev_macd, prev_macd_signal, psar
    )

    is_signal_expected = sample_strategy.is_short_trade_signal_present(ohlcv_row)
    assert is_signal_expected == expected_signal


@pytest.mark.parametrize(
    "close, psar, direction, expected_entry_price, expected_target_price, expected_stop_price",
    [
        (100, 50, 1, 100, 175.0, 50),
        (50, 60, -1, 50, 35.0, 60),
        (50, 100, -1, 50, 0, 100),
    ],
)
def test_generate_trade_levels(
    sample_strategy,
    close,
    psar,
    direction,
    expected_entry_price,
    expected_target_price,
    expected_stop_price,
) -> None:
    OHLCVRow = namedtuple(
        "OHLCVRow",
        ["close", "psar"],
    )
    ohlcv_row = OHLCVRow(close, psar)

    expected_trade_levels = sample_strategy.generate_trade_levels(
        ohlcv_row, trade_signal_direction=direction
    )

    assert expected_trade_levels == TradeLevels(
        entry_price=expected_entry_price,
        target_price=expected_target_price,
        stop_price=expected_stop_price,
    )


@pytest.mark.parametrize(
    "input_params, expected_params",
    [
        (
            {"psar_acceleration": 1, "psar_max_val": 2},
            {"psar_acceleration": 1, "psar_max_val": 2},
        ),
        (
            {"psar_acceleration": 2, "psar_max_val": 1},
            {"psar_acceleration": 1, "psar_max_val": 2},
        ),
    ],
)
def test_define_optimization_param_constraints(
    sample_strategy, input_params, expected_params
) -> None:
    result_params = sample_strategy.define_optimization_param_constraints(input_params)
    assert expected_params == result_params
