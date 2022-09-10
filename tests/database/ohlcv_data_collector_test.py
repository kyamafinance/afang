import pytest

from afang.database.ohlcv_data_collector import (
    fetch_historical_price_data,
    fetch_initial_data,
    fetch_most_recent_data,
    fetch_older_data,
    fetch_symbol_data,
)
from afang.database.ohlcv_database import OHLCVDatabase
from afang.exchanges.models import Candle


@pytest.mark.parametrize(
    "historical_data, expected_return_value",
    [
        (None, (None, None)),
        ([], (None, None)),
        ([Candle(1, 2, 3, 4, 5, 6)], (None, None)),
        ([Candle(1, 2, 3, 4, 5, 6), Candle(1, 2, 3, 4, 5, 6)], (1, 1)),
        (
            [
                Candle(2, 2, 3, 4, 5, 6),
                Candle(1, 2, 3, 4, 5, 6),
                Candle(4, 2, 3, 4, 5, 6),
                Candle(3, 2, 3, 4, 5, 6),
            ],
            (1, 3),
        ),
    ],
)
def test_fetch_initial_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_candles function
    mocked_get_historical_data = mocker.patch(
        "afang.database.ohlcv_data_collector.IsExchange.get_historical_candles"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.database.ohlcv_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)
    test_ohlcv_db.create_dataset("test_symbol")
    return_val = fetch_initial_data(dummy_is_exchange, "test_symbol", test_ohlcv_db)

    assert mocked_get_historical_data.assert_called
    assert mocked_write_data.assert_called
    assert return_val == expected_return_value


@pytest.mark.parametrize(
    "historical_data, expected_return_value",
    [
        ([Candle(1, 2, 3, 4, 5, 6)], 0),
        ([Candle(-2, 2, 3, 4, 5, 6), Candle(-1, 2, 3, 4, 5, 6)], None),
        (
            [
                Candle(1, 2, 3, 4, 5, 6),
                Candle(2, 2, 3, 4, 5, 6),
                Candle(3, 2, 3, 4, 5, 6),
            ],
            None,
        ),
    ],
)
def test_fetch_most_recent_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_candles function
    mocked_get_historical_data = mocker.patch(
        "afang.database.ohlcv_data_collector.IsExchange.get_historical_candles"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.database.ohlcv_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)
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
        ([Candle(10, 2, 3, 4, 5, 6), Candle(11, 2, 3, 4, 5, 6)], None),
        (
            [
                Candle(1, 2, 3, 4, 5, 6),
                Candle(2, 2, 3, 4, 5, 6),
                Candle(3, 2, 3, 4, 5, 6),
            ],
            None,
        ),
    ],
)
def test_fetch_older_data(
    mocker, dummy_is_exchange, ohlcv_root_db_dir, historical_data, expected_return_value
) -> None:
    # mock the get_historical_candles function
    mocked_get_historical_data = mocker.patch(
        "afang.database.ohlcv_data_collector.IsExchange.get_historical_candles"
    )
    mocked_get_historical_data.return_value = historical_data

    # mock the write_data function
    mocked_write_data = mocker.patch(
        "afang.database.ohlcv_data_collector.OHLCVDatabase.write_data"
    )

    test_ohlcv_db = OHLCVDatabase(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)
    test_ohlcv_db.create_dataset("test_symbol")
    return_val = fetch_older_data(
        dummy_is_exchange, "test_symbol", test_ohlcv_db, 5, 1, 1
    )

    assert mocked_get_historical_data.assert_called
    assert mocked_write_data.assert_called
    assert return_val == expected_return_value


def test_fetch_symbol_data(mocker, dummy_is_exchange, ohlcv_root_db_dir):
    dummy_is_exchange.exchange_symbols = ["test_symbol"]

    # mock the get_min_max_timestamp function
    mocked_get_min_max_timestamp = mocker.patch(
        "afang.database.ohlcv_data_collector.OHLCVDatabase.get_min_max_timestamp"
    )
    mocked_get_min_max_timestamp.return_value = (None, None)

    # mock the fetch_initial_data function
    mocked_fetch_initial_data = mocker.patch(
        "afang.database.ohlcv_data_collector.fetch_initial_data"
    )
    mocked_fetch_initial_data.return_value = (50, 60)

    # mock the fetch_most_recent_data function
    mocked_fetch_most_recent_data = mocker.patch(
        "afang.database.ohlcv_data_collector.fetch_most_recent_data"
    )
    mocked_fetch_most_recent_data.return_value = 70

    # mock the fetch_older_data function
    mocked_fetch_older_data = mocker.patch(
        "afang.database.ohlcv_data_collector.fetch_older_data"
    )
    mocked_fetch_older_data.return_value = 40

    # mock the is_dataset_valid function
    mocked_is_dataset_valid = mocker.patch(
        "afang.database.ohlcv_data_collector.OHLCVDatabase.is_dataset_valid"
    )
    mocked_is_dataset_valid.return_value = True

    fetch_symbol_data(dummy_is_exchange, "test_symbol", ohlcv_root_db_dir)

    assert mocked_get_min_max_timestamp.assert_called
    assert mocked_fetch_initial_data.assert_called
    assert mocked_fetch_most_recent_data.assert_called
    assert mocked_fetch_older_data.assert_called
    assert mocked_is_dataset_valid.assert_called


def test_fetch_historical_price_data_no_symbols(dummy_is_exchange, caplog) -> None:
    fetch_historical_price_data(dummy_is_exchange, [])

    assert caplog.records[0].levelname == "WARNING"
    assert "No symbols found to fetch historical price data" in caplog.text


def test_fetch_historical_price_data_no_symbols_with_strategy(
    mocker, dummy_is_exchange, dummy_is_strategy, caplog
) -> None:
    dummy_is_exchange.exchange_symbols = ["test_symbol"]
    mocked_fetch_symbol_data = mocker.patch(
        "afang.database.ohlcv_data_collector.fetch_symbol_data",
        return_value=True,
    )

    fetch_historical_price_data(dummy_is_exchange, [], strategy=dummy_is_strategy)

    # test to assert that if a strategy with an exchange watchlist is provided,
    # there will be symbols whose data is to be fetched therefore there will
    # be no warning/error log.
    assert mocked_fetch_symbol_data.assert_called
    assert not caplog.text


def test_fetch_historical_price_data_unknown_symbol(dummy_is_exchange, caplog):
    fetch_historical_price_data(dummy_is_exchange, ["unknown_symbol"])

    assert caplog.records[0].levelname == "ERROR"
    assert (
        "test_exchange unknown_symbol: provided symbol not present in the exchange"
        in caplog.text
    )


def test_fetch_historical_price_data(mocker, dummy_is_exchange) -> None:
    mocked_fetch_symbol_data = mocker.patch(
        "afang.database.ohlcv_data_collector.fetch_symbol_data",
        return_value=True,
    )

    fetch_historical_price_data(dummy_is_exchange, ["test_symbol"])

    assert mocked_fetch_symbol_data.assert_called
