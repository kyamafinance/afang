import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def milliseconds_to_datetime(milliseconds: int) -> datetime:
    """Convert a UNIX timestamp in milliseconds to a datetime object.

    :param milliseconds: UNIX timestamp in milliseconds.
    :return: datetime
    """

    return datetime.utcfromtimestamp(milliseconds / 1000)


def time_str_to_milliseconds(time_str: str) -> int:
    """Convert a UTC time string in the format '%Y-%m-%d' to a timestamp in
    milliseconds.

    :param time_str: time string in the format '%Y-%m-%d'.
    :return: int
    """

    try:
        date = datetime.strptime(time_str, "%Y-%m-%d")
        date = date.replace(tzinfo=timezone.utc)
        milliseconds = int(date.timestamp()) * 1000

        return milliseconds

    except ValueError:
        raise ValueError(
            f"Provided time string:{time_str} not in the format '%Y-%m-%d'"
        )


def resample_timeframe(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a 1 minute price data OHLCV dataframe to a different timeframe.

    :param data: 1 minute price data OHLCV dataframe.
    :param timeframe: desired timeframe to resample the price data into.
    :return: pd.DataFrame
    """

    tf_mapping = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1H",
        "4h": "4H",
        "12h": "12H",
        "1d": "D",
    }
    return data.resample(tf_mapping[timeframe]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def get_float_precision(input_float: float) -> int:
    """Get the precision of a floating point number.

    :param input_float: float.
    :return: int
    """

    input_float_str = f"{input_float:.8f}"
    while input_float_str[-1] == "0":
        input_float_str = input_float_str[:-1]

    split_input_float = input_float_str.split(".")

    if len(split_input_float) > 1:
        return len(split_input_float[1])

    return 0
