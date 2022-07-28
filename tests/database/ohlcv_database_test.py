import os
from typing import List, Tuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from afang.database.ohlcv_database import OHLCVDatabase


@pytest.fixture
def ohlcv_write_data() -> List[Tuple[float, float, float, float, float, float]]:
    return [
        (
            1657510380000,
            10.2,
            12.5,
            9.5,
            10.2,
            100.5,
        ),  # Monday, July 11, 2022 3:33:00 AM
        (
            1657510320000,
            10.5,
            12.5,
            9.5,
            10.2,
            100.5,
        ),  # Monday, July 11, 2022 3:32:00 AM
    ]


def test_symbol_database_creation(ohlcv_root_db_dir) -> None:
    # test that database is created if it doesn't yet exist.
    OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    assert os.path.exists(f"{ohlcv_root_db_dir}/test_exchange") is True
    assert os.path.exists(f"{ohlcv_root_db_dir}/test_exchange/test_symbol.h5") is True

    # test that a new symbol database can be created under the same exchange.
    OHLCVDatabase("test_exchange", "another_test_symbol", ohlcv_root_db_dir)
    assert os.path.exists(f"{ohlcv_root_db_dir}/test_exchange") is True
    assert (
        os.path.exists(f"{ohlcv_root_db_dir}/test_exchange/another_test_symbol.h5")
        is True
    )


def test_symbol_dataset_creation(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    assert "test_symbol" not in ohlcv_db.hf.keys()

    ohlcv_db.create_dataset("test_symbol")
    assert "test_symbol" in ohlcv_db.hf.keys()


def test_write_to_non_existent_dataset(ohlcv_root_db_dir, caplog) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.write_data("non_existent_symbol", [])

    assert caplog.records[0].levelname == "WARNING"
    assert (
        "non_existent_symbol: no dataset exists for symbol in database" in caplog.text
    )


@pytest.mark.parametrize(
    "min_ts, max_ts, expect_log, expected_log_text, expected_data_len",
    [
        (1657510320000, 1657510380000, True, "no data to insert into database", 2),
        (1657510320000, 1657510320000, True, "does not match length of input data", 2),
        (1657510300000, 1657510300000, False, "", 4),
    ],
)
def test_write_to_dataset(
    mocker,
    ohlcv_root_db_dir,
    caplog,
    ohlcv_write_data,
    min_ts,
    max_ts,
    expect_log,
    expected_log_text,
    expected_data_len,
) -> None:
    # set up dataset for a symbol and write initial data into the dataset.
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    ohlcv_db.write_data("test_symbol", ohlcv_write_data)

    # mock the return value of the get_min_max_timestamp function.
    def mock_get_min_max_timestamp(_self, _symbol):
        return min_ts, max_ts

    mocker.patch(
        "afang.database.ohlcv_database.OHLCVDatabase.get_min_max_timestamp",
        mock_get_min_max_timestamp,
    )

    # attempt to write the test data into the dataset.
    ohlcv_db.write_data("test_symbol", ohlcv_write_data)
    dataset_data_len = len(ohlcv_db.hf.get("test_symbol")[:])

    assert dataset_data_len == expected_data_len
    if expect_log:
        assert expected_log_text in caplog.text


def test_get_min_max_timestamp_no_dataset(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    assert ohlcv_db.get_min_max_timestamp("test_symbol") == (None, None)


def test_get_min_max_timestamp_empty_dataset(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    assert ohlcv_db.get_min_max_timestamp("test_symbol") == (None, None)


@pytest.mark.parametrize(
    "data, min_ts, max_ts",
    [
        ([(1657510320000, 10.5, 12.5, 9.5, 10.2, 100.5)], 1657510320000, 1657510320000),
        (
            [
                (1657510320000, 10.5, 12.5, 9.5, 10.2, 100.5),
                (1657510380000, 10.2, 12.5, 9.5, 10.2, 100.5),
            ],
            1657510320000,
            1657510380000,
        ),
    ],
)
def test_get_min_max_timestamp(ohlcv_root_db_dir, data, min_ts, max_ts) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    ohlcv_db.write_data("test_symbol", data)

    assert ohlcv_db.get_min_max_timestamp("test_symbol") == (min_ts, max_ts)


def test_get_data_no_dataset(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    data = ohlcv_db.get_data("test_symbol", 0, 100)
    assert data is None


def test_get_data_empty_dataset(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    data = ohlcv_db.get_data("test_symbol", 0, 100)
    assert data is None


def test_get_data(ohlcv_root_db_dir, caplog) -> None:
    ohlcv_data: List[Tuple[float, float, float, float, float, float]] = [
        (5, 40.2, 92.5, 8.5, 12.2, 190.5),
        (1, 10.2, 12.5, 9.5, 14.2, 120.5),
        (4, 70.2, 42.5, 1.5, 15.2, 160.5),
        (2, 30.2, 72.5, 4.5, 16.2, 130.5),
        (3, 60.2, 62.5, 2.5, 18.2, 150.5),
    ]

    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    ohlcv_db.write_data("test_symbol", ohlcv_data)
    data = ohlcv_db.get_data("test_symbol", 2, 4)

    expected_data = {
        "open": [30.2, 60.2, 70.2],
        "high": [72.5, 62.5, 42.5],
        "low": [4.5, 2.5, 1.5],
        "close": [16.2, 18.2, 15.2],
        "volume": [130.5, 150.5, 160.5],
    }
    ts_data = [
        "1970-01-01 00:00:00.002",
        "1970-01-01 00:00:00.003",
        "1970-01-01 00:00:00.004",
    ]
    expected_data_df = pd.DataFrame(data=expected_data)
    expected_data_df.set_index(pd.DatetimeIndex(ts_data), drop=True, inplace=True)
    expected_data_df.index.name = "timestamp"

    assert_frame_equal(data, expected_data_df, check_dtype=True)


def test_is_data_valid_no_dataset(ohlcv_root_db_dir, caplog) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    is_valid = ohlcv_db.is_dataset_valid("test_symbol")

    assert "no dataset exists for symbol in database" in caplog.text
    assert not is_valid


def test_is_data_valid_empty_dataset(ohlcv_root_db_dir) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    is_valid = ohlcv_db.is_dataset_valid("test_symbol")

    assert not is_valid


@pytest.mark.parametrize(
    "data, expect_is_valid",
    [
        (
            [
                (1657510560000, 10.5, 12.5, 9.5, 10.2, 100.5),
                (1657510380000, 10.2, 12.5, 9.5, 10.2, 100.5),
                (1657510320000, 10.5, 12.5, 9.5, 10.2, 100.5),
                (1657510500000, 10.5, 12.5, 9.5, 10.2, 100.5),
            ],
            True,
        ),
        (
            [
                (1657510560000, 10.5, 12.5, 9.5, 10.2, 100.5),
                (1657510500000, 10.2, 12.5, 9.5, 10.2, 100.5),
                (1657510320000, 10.5, 12.5, 9.5, 10.2, 100.5),
                (1657510500000, 10.5, 12.5, 9.5, 10.2, 100.5),
            ],
            False,
        ),
    ],
)
def test_is_data_valid(ohlcv_root_db_dir, caplog, data, expect_is_valid) -> None:
    ohlcv_db = OHLCVDatabase("test_exchange", "test_symbol", ohlcv_root_db_dir)
    ohlcv_db.create_dataset("test_symbol")
    ohlcv_db.write_data("test_symbol", data)
    is_valid = ohlcv_db.is_dataset_valid("test_symbol")

    assert is_valid == expect_is_valid
