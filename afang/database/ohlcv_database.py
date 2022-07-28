import logging
import os
import pathlib
import time
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVDatabase:
    """Interface to store, retrieve, and manipulate price data in a HDF5
    database."""

    def __init__(
        self, exchange: str, symbol: str, root_db_dir: Optional[str] = None
    ) -> None:
        """Initialize the OHLCVDatabase class.

        :param exchange: name of exchange to use.
        :param symbol: symbol to use.
        :param root_db_dir: root database directory.
        """

        if not root_db_dir:
            root_db_dir = f"{pathlib.Path(__file__).parents[2]}/data/ohlcv"

        # Create exchange database directory if it does not exist.
        exchange_db_dir = f"{root_db_dir}/{exchange}"
        if not os.path.exists(exchange_db_dir):
            os.mkdir(exchange_db_dir)

        self.hf = h5py.File(f"{exchange_db_dir}/{symbol}.h5", "a")
        self.hf.flush()

    def create_dataset(self, symbol: str) -> None:
        """Create a dataset in the exchange's HDF5 database to contain symbol
        price data information if one does not already exist.

        :param symbol: symbol to create a dataset for.

        :return: None
        """

        if symbol in self.hf.keys():
            return

        self.hf.create_dataset(
            name=symbol, shape=(0, 6), maxshape=(None, 6), dtype="float64"
        )
        self.hf.flush()

    def get_min_max_timestamp(
        self, symbol: str
    ) -> Union[Tuple[None, None], Tuple[float, float]]:
        """Fetch the minimum(oldest) and maximum(latest) timestamp in a
        symbol's stored price data. If no dataset exists for the provided
        symbol, (None, None) will be returned.

        :param symbol: name of symbol to use.

        :return: Union[Tuple[None, None], Tuple[float, float]]
        """

        if symbol not in self.hf.keys():
            return None, None

        existing_data = self.hf.get(symbol)[:]
        if not len(existing_data):
            return None, None

        min_timestamp = min(existing_data, key=lambda row: row[0])[0]
        max_timestamp = max(existing_data, key=lambda row: row[0])[0]

        return min_timestamp, max_timestamp

    def write_data(
        self, symbol: str, data: List[Tuple[float, float, float, float, float, float]]
    ) -> None:
        """Write OHLCV price data into a symbol's dataset within an exchange's
        HDF5 database.

        :param symbol: symbol whose price data is to be stored.
        :param data: list of OHLCV tuples to be stored into the symbol's dataset.

        :return: None
        """

        if symbol not in self.hf.keys():
            logger.warning("%s: no dataset exists for symbol in database", symbol)
            return

        if not len(data):
            return

        min_timestamp, max_timestamp = self.get_min_max_timestamp(symbol)
        if min_timestamp is None:
            min_timestamp = float("inf")
            max_timestamp = 0

        filtered_data = []
        for d in data:
            if d[0] < min_timestamp:
                filtered_data.append(d)
            elif d[0] > max_timestamp:
                filtered_data.append(d)

        if not len(filtered_data):
            logger.warning("%s: no data to insert into database", symbol)
            return

        if len(filtered_data) != len(data):
            logger.warning(
                "%s: length of filtered data(%s) does not match length of input data(%s)",
                symbol,
                len(filtered_data),
                len(data),
            )
            return

        data_array = np.array(filtered_data)

        self.hf.get(symbol).resize(
            self.hf.get(symbol).shape[0] + data_array.shape[0], axis=0
        )
        idx = -data_array.shape[0]
        self.hf.get(symbol)[idx:] = data_array
        self.hf.flush()

    def get_data(
        self, symbol: str, from_time: int, to_time: int
    ) -> Optional[pd.DataFrame]:
        """Retrieve price data from the database for a given symbol sorted by
        timestamps and within the from_time and to_time parameters.

        :param symbol: name of symbol whose data is to be retrieved.
        :param from_time: UNIX timestamp in ms which retrieved price data should not be older than.
        :param to_time: UNIX timestamp in ms which retrieved price data should not be newer than.

        :return: Optional[pd.DataFrame]
        """

        if symbol not in self.hf.keys():
            logger.warning("%s: no dataset exists for symbol in database", symbol)
            return None

        query_start_time = time.time()

        existing_data = self.hf.get(symbol)[:]
        if not len(existing_data):
            return None

        data = sorted(existing_data, key=lambda row: row[0])
        data = np.array(data)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = df[(df.timestamp >= from_time) & (df.timestamp <= to_time)]

        df.timestamp = pd.to_datetime(df.timestamp.values.astype(np.int64), unit="ms")
        df.set_index("timestamp", drop=True, inplace=True)

        query_time = round(time.time() - query_start_time, 2)

        logger.info(
            "Retrieved %s %s data candles from the database in %s seconds",
            len(df.index),
            symbol,
            query_time,
        )

        return df

    @classmethod
    def _is_unique(
        cls, df_col: pd.Series, allowed_vals: Any, astype: str = "timedelta64[m]"
    ) -> List[str]:
        """Check if all values in a pandas' series are of a specific allowed
        set. This function will return a list of disallowed items.

        :param df_col: dataframe column.
        :param allowed_vals: dict of values allowed in the pandas' series.
        :param astype: type to cast disallowed values to.

        :return: List[str]
        """

        value_counts = df_col.value_counts()
        disallowed_values = set(df_col.unique()) - allowed_vals

        verbose_disallowed_values = list()
        for disallowed_value in disallowed_values:
            verbose_disallowed_values.append(
                f"{disallowed_value.astype(astype)} ({value_counts[disallowed_value]})"
            )

        return verbose_disallowed_values

    def is_dataset_valid(self, symbol: str) -> bool:
        """Ensure that persisted price data of a given symbol is valid i.e.
        timestamps are strictly 1 or 2 minute(s) apart.

        :param symbol: name of symbol whose data is to be validated.

        :return: bool
        """

        if symbol not in self.hf.keys():
            logger.warning("%s: no dataset exists for symbol in database", symbol)
            return False

        existing_data = self.hf.get(symbol)[:]
        if not len(existing_data):
            return False

        data = sorted(existing_data, key=lambda row: row[0])
        data = np.array(data)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df.timestamp = pd.to_datetime(df.timestamp.values.astype(np.int64), unit="ms")

        df["open-time-diff"] = df["timestamp"].shift(-1) - df["timestamp"]
        allowed_values = {
            np.timedelta64(60000000000, "ns"),  # 1 minute
            np.timedelta64(120000000000, "ns"),  # 2 minutes
        }
        disallowed_values = self._is_unique(df["open-time-diff"][:-1], allowed_values)
        if disallowed_values:
            logger.warning(
                "%s: invalid symbol dataset. disallowed values: %s",
                symbol,
                disallowed_values,
            )
            return False

        logger.info("%s: symbol dataset is valid", symbol)
        return True
