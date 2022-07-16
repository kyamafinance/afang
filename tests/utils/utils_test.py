from datetime import datetime

from afang.utils.util import milliseconds_to_datetime


def test_milliseconds_to_datetime() -> None:
    expected_dt = datetime(year=2022, month=2, day=14, hour=5, minute=25, second=20)
    assert milliseconds_to_datetime(1644816320000) == expected_dt
