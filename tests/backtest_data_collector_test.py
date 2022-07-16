import pytest

from afang.backtest_data_collector import (
    collect_all,
    fetch_initial_data,
    fetch_most_recent_data,
    fetch_older_data,
)
from afang.database.ohlcv_database import OHLCVDatabase


@pytest.mark.parametrize(
    "historical_data, expected_return_value",
    [
        (None, (None, None)),
        ([], (None, None)),
        ([(1, 2, 3, 4, 5, 6)], (None, None)),
        ([(1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)], (1, 1)),
        (
            [
                (2, 2, 3, 4, 5, 6),
                (1, 2, 3, 4, 5, 6),
                (4, 2, 3, 4, 5, 6),
                (3, 2, 3, 4, 5, 6),
            ],
            (1, 3),
        ),
    ],
)
def test_fetch_initial_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_data function
    mocked_get_historical_data = mocker.patch(
        "afang.backtest_data_collector.IsExchange.get_historical_data"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.backtest_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(ohlcv_root_db_dir, "test_exchange", "test_symbol")
    test_ohlcv_db.create_dataset("test_symbol")
    return_val = fetch_initial_data(dummy_is_exchange, "test_symbol", test_ohlcv_db)

    assert mocked_get_historical_data.assert_called
    assert mocked_write_data.assert_called
    assert return_val == expected_return_value


@pytest.mark.parametrize(
    "historical_data, expected_return_value",
    [
        ([(1, 2, 3, 4, 5, 6)], 0),
        ([(-2, 2, 3, 4, 5, 6), (-1, 2, 3, 4, 5, 6)], None),
        ([(1, 2, 3, 4, 5, 6), (2, 2, 3, 4, 5, 6), (3, 2, 3, 4, 5, 6)], None),
    ],
)
def test_fetch_most_recent_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_data function
    mocked_get_historical_data = mocker.patch(
        "afang.backtest_data_collector.IsExchange.get_historical_data"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.backtest_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(ohlcv_root_db_dir, "test_exchange", "test_symbol")
    test_ohlcv_db.create_dataset("test_symbol")
    return_val = fetch_most_recent_data(
        dummy_is_exchange, "test_symbol", test_ohlcv_db, 0, 1, 1
    )

    assert mocked_get_historical_data.assert_called
    assert mocked_write_data.assert_called
    assert return_val == expected_return_value


@pytest.mark.parametrize(
    "historical_data, expected_return_value",
    [
        ([], 5),
        ([(10, 2, 3, 4, 5, 6), (11, 2, 3, 4, 5, 6)], None),
        ([(1, 2, 3, 4, 5, 6), (2, 2, 3, 4, 5, 6), (3, 2, 3, 4, 5, 6)], None),
    ],
)
def test_fetch_older_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_data function
    mocked_get_historical_data = mocker.patch(
        "afang.backtest_data_collector.IsExchange.get_historical_data"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.backtest_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(ohlcv_root_db_dir, "test_exchange", "test_symbol")
    test_ohlcv_db.create_dataset("test_symbol")
    return_val = fetch_older_data(
        dummy_is_exchange, "test_symbol", test_ohlcv_db, 5, 1, 1
    )

    assert mocked_get_historical_data.assert_called
    assert mocked_write_data.assert_called
    assert return_val == expected_return_value


def test_collect_all(mocker, dummy_is_exchange, ohlcv_root_db_dir):
    dummy_is_exchange.symbols = ["test_symbol"]

    # mock the get_min_max_timestamp function
    mocked_get_min_max_timestamp = mocker.patch(
        "afang.backtest_data_collector.OHLCVDatabase.get_min_max_timestamp"
    )
    mocked_get_min_max_timestamp.return_value = (None, None)

    # mock the fetch_initial_data function
    mocked_fetch_initial_data = mocker.patch(
        "afang.backtest_data_collector.fetch_initial_data"
    )
    mocked_fetch_initial_data.return_value = (50, 60)

    # mock the fetch_most_recent_data function
    mocked_fetch_most_recent_data = mocker.patch(
        "afang.backtest_data_collector.fetch_most_recent_data"
    )
    mocked_fetch_most_recent_data.return_value = 70

    # mock the fetch_older_data function
    mocked_fetch_older_data = mocker.patch(
        "afang.backtest_data_collector.fetch_older_data"
    )
    mocked_fetch_older_data.return_value = 40

    # mock the is_dataset_valid function
    mocked_is_dataset_valid = mocker.patch(
        "afang.backtest_data_collector.OHLCVDatabase.is_dataset_valid"
    )
    mocked_is_dataset_valid.return_value = True

    collect_all(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir, 1, 1)

    assert mocked_get_min_max_timestamp.assert_called
    assert mocked_fetch_initial_data.assert_called
    assert mocked_fetch_most_recent_data.assert_called
    assert mocked_fetch_older_data.assert_called
    assert mocked_is_dataset_valid.assert_called
