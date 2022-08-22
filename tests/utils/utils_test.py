from datetime import datetime

import pandas as pd
import pytest

from afang.utils.util import (
    get_float_precision,
    milliseconds_to_datetime,
    resample_timeframe,
    time_str_to_milliseconds,
)


def test_milliseconds_to_datetime() -> None:
    expected_dt = datetime(year=2022, month=2, day=14, hour=5, minute=25, second=20)
    assert milliseconds_to_datetime(1644816320000) == expected_dt


def test_time_str_to_milliseconds() -> None:
    assert time_str_to_milliseconds("2022-01-01") == 1640995200000


def test_time_str_to_milliseconds_exception() -> None:
    time_str = "2022/01/01"
    value_err_msg = f"Provided time string:{time_str} not in the format '%Y-%m-%d'"
    with pytest.raises(ValueError, match=value_err_msg):
        time_str_to_milliseconds(time_str)


def test_resample_timeframe() -> None:
    test_df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [7, 8, 9, 10, 11, 12],
            "low": [13, 14, 15, 16, 17, 18],
            "close": [19, 20, 21, 22, 23, 24],
            "volume": [25, 26, 27, 28, 29, 30],
        },
        index=[
            pd.to_datetime("2022-01-01 01:00:00"),
            pd.to_datetime("2022-01-01 01:01:00"),
            pd.to_datetime("2022-01-01 01:02:00"),
            pd.to_datetime("2022-01-01 01:03:00"),
            pd.to_datetime("2022-01-01 01:04:00"),
            pd.to_datetime("2022-01-01 01:05:00"),
        ],
    )

    expected_df = pd.DataFrame(
        {
            "open": [1, 6],
            "high": [11, 12],
            "low": [13, 18],
            "close": [23, 24],
            "volume": [135, 30],
        },
        index=[
            pd.to_datetime("2022-01-01 01:00:00"),
            pd.to_datetime("2022-01-01 01:05:00"),
        ],
    )

    assert resample_timeframe(test_df, "5m").equals(expected_df)


@pytest.mark.parametrize(
    "input_float, expected_precision", [(20, 0), (0, 0), (20.32, 2), (43.53255, 5)]
)
def test_get_float_precision(input_float, expected_precision) -> None:
    precision = get_float_precision(input_float)
    assert precision == expected_precision
