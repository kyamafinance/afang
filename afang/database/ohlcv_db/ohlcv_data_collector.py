import logging
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from operator import attrgetter
from typing import List, Optional, Tuple, Union

from afang.database.ohlcv_db.ohlcv_database import OHLCVDatabase
from afang.exchanges.is_exchange import IsExchange
from afang.strategies.is_strategy import IsStrategy
from afang.utils.util import milliseconds_to_datetime

logger = logging.getLogger(__name__)


def fetch_initial_data(
    exchange: IsExchange, symbol: str, ohlcv_db: OHLCVDatabase
) -> Union[Tuple[None, None], Tuple[float, float]]:
    """Fetch the most recent historical price data for a symbol from a given
    exchange and store this data in the provided ohlcv database.

    :param exchange: an instance of an interface of the exchange to use
        to fetch historical price data.
    :param symbol: name of the symbol whose historical price data should
        be fetched.
    :param ohlcv_db: an instance of an interface of an ohlcv database.
    :return: Union[Tuple[None, None], Tuple[float, float]]
    """

    data = exchange.get_historical_candles(symbol, end_time=int(time.time() * 1000))

    if data is None:
        logger.warning(
            "%s %s: fetch initial data request failed", exchange.display_name, symbol
        )
        return None, None

    if len(data) < 2:
        logger.warning("%s %s: no initial data found", exchange.display_name, symbol)
        return None, None

    data = sorted(data, key=attrgetter("open_time"))
    data = data[:-1]  # removing the last candle because it is likely unfinished.

    logger.info(
        "%s %s: collected %s initial data candles from %s to %s",
        exchange.display_name,
        symbol,
        len(data),
        milliseconds_to_datetime(int(data[0].open_time)),
        milliseconds_to_datetime(int(data[-1].open_time)),
    )

    oldest_timestamp, most_recent_timestamp = data[0].open_time, data[-1].open_time

    ohlcv_db.write_data(symbol, data)

    return oldest_timestamp, most_recent_timestamp


def fetch_most_recent_data(
    exchange: IsExchange,
    symbol: str,
    ohlcv_db: OHLCVDatabase,
    most_recent_timestamp: float,
    query_limit: float,
    write_limit: int,
) -> Optional[float]:
    """Fetch a symbol's most recent historical price data from an exchange
    after a given timestamp and store this data in the database.

    :param exchange: an instance of an interface of the exchange to use
        to fetch historical price data.
    :param symbol: name of the symbol whose historical price data should
        be fetched.
    :param ohlcv_db: an instance of an interface to an OHLCV database.
    :param most_recent_timestamp: UNIX timestamp in ms after which
        historical price data will be fetched.
    :param query_limit: rate limit of how long to sleep between HTTP
        requests.
    :param write_limit: threshold of how many candles to fetch before
        saving them to the DB.
    :return: Optional[float]
    """

    data_to_insert = []
    _most_recent_timestamp = most_recent_timestamp

    while True:
        data = exchange.get_historical_candles(
            symbol, start_time=int(_most_recent_timestamp)
        )
        data = sorted(data, key=attrgetter("open_time"))

        if data is None:
            time.sleep(4)  # Pause in the event an error occurs during data collection.
            continue

        if len(data) < 2:
            break

        data = data[:-1]  # removing the last candle because it is likely unfinished.

        if data[-1].open_time <= _most_recent_timestamp:
            logger.warning(
                "%s %s: most recent fetched data timestamp(%s) not newer than "
                "existing most recent timestamp(%s)",
                exchange.display_name,
                symbol,
                milliseconds_to_datetime(int(data[-1].open_time)),
                milliseconds_to_datetime(int(_most_recent_timestamp)),
            )
            return None

        _most_recent_timestamp = data[-1].open_time

        logger.info(
            "%s %s: collected %s most recent data candles from %s to %s",
            exchange.display_name,
            symbol,
            len(data),
            milliseconds_to_datetime(int(data[0].open_time)),
            milliseconds_to_datetime(int(data[-1].open_time)),
        )

        data_to_insert += data
        if len(data_to_insert) >= write_limit:
            ohlcv_db.write_data(symbol, data_to_insert)
            data_to_insert.clear()

        time.sleep(query_limit)

    if data_to_insert:
        ohlcv_db.write_data(symbol, data_to_insert)
        data_to_insert.clear()

    return _most_recent_timestamp


def fetch_older_data(
    exchange: IsExchange,
    symbol: str,
    ohlcv_db: OHLCVDatabase,
    oldest_timestamp: float,
    query_limit: float,
    write_limit: int,
) -> Optional[float]:
    """Fetch a symbol's historical price data from an exchange before a given
    timestamp and store this data in the database.

    :param exchange: an instance of an interface of the exchange to use
        to fetch historical price data.
    :param symbol: name of the symbol whose historical price data should
        be fetched.
    :param ohlcv_db: an instance of an interface to an OHLCV database.
    :param oldest_timestamp: UNIX timestamp in ms before which
        historical price data will be fetched.
    :param query_limit: rate limit of how long to sleep between HTTP
        requests.
    :param write_limit: threshold of how many candles to fetch before
        saving them to the DB.
    :return: Optional[float]
    """

    data_to_insert = []
    _oldest_timestamp = oldest_timestamp

    while True:
        data = exchange.get_historical_candles(symbol, end_time=int(_oldest_timestamp))
        data = sorted(data, key=attrgetter("open_time"))

        if data is None:
            time.sleep(4)  # Pause in the event an error occurs during data collection.
            continue

        if not len(data):
            logger.info(
                "%s %s: older data collection stopped because no data before %s was found",
                exchange.display_name,
                symbol,
                milliseconds_to_datetime(int(_oldest_timestamp)),
            )
            break

        if data[0].open_time >= _oldest_timestamp:
            logger.warning(
                "%s %s: fetched data oldest timestamp(%s) not older than existing oldest timestamp(%s)",
                exchange.display_name,
                symbol,
                milliseconds_to_datetime(int(data[0].open_time)),
                milliseconds_to_datetime(int(_oldest_timestamp)),
            )
            return None

        _oldest_timestamp = data[0].open_time

        logger.info(
            "%s %s: collected %s older data candles from %s to %s",
            exchange.display_name,
            symbol,
            len(data),
            milliseconds_to_datetime(int(data[0].open_time)),
            milliseconds_to_datetime(int(data[-1].open_time)),
        )

        data_to_insert += data
        if len(data_to_insert) >= write_limit:
            ohlcv_db.write_data(symbol, data_to_insert)
            data_to_insert.clear()

        time.sleep(query_limit)

    if data_to_insert:
        ohlcv_db.write_data(symbol, data_to_insert)
        data_to_insert.clear()

    return _oldest_timestamp


def fetch_symbol_data(
    exchange: IsExchange,
    symbol: str,
    root_db_dir: Optional[str] = None,
) -> Optional[bool]:
    """Collect all historical price data for a given symbol from a given
    exchange and store this data in the database. Returns an optional bool on
    whether the symbol dataset contains valid data.

    :param exchange: an instance of an interface of the exchange to use
        to fetch historical price data.
    :param symbol: name of the symbol whose historical price data should
        be fetched.
    :param root_db_dir: path to the intended root OHLCV database
        directory.
    :return: Optional[bool]
    """

    if symbol not in exchange.exchange_symbols:
        logger.warning(
            "%s %s: provided symbol not present in the exchange",
            exchange.display_name,
            symbol,
        )
        return None

    ohlcv_db = OHLCVDatabase(exchange, symbol, root_db_dir)
    ohlcv_db.create_dataset(symbol)

    oldest_timestamp, most_recent_timestamp = ohlcv_db.get_min_max_timestamp(symbol)

    # Initial data request.
    if oldest_timestamp is None:
        oldest_timestamp, most_recent_timestamp = fetch_initial_data(
            exchange,
            symbol,
            ohlcv_db,
        )

        if oldest_timestamp is None:
            return None

    client_config = exchange.get_config_params()
    # Rate limit of how long to sleep between HTTP requests.
    query_limit = client_config.get("query_limit", 1)
    # Threshold of how many candles to fetch before saving them to the DB.
    write_limit = client_config.get("write_limit", 10000)

    # Get most recent data.
    most_recent_timestamp = fetch_most_recent_data(
        exchange,
        symbol,
        ohlcv_db,
        most_recent_timestamp,
        query_limit,
        write_limit,
    )

    if most_recent_timestamp is None:
        return None

    # Get older data.
    oldest_timestamp = fetch_older_data(
        exchange, symbol, ohlcv_db, oldest_timestamp, query_limit, write_limit
    )

    if oldest_timestamp is None:
        return None

    # Validate symbol data
    return ohlcv_db.is_dataset_valid(symbol)


def fetch_historical_price_data(
    exchange: IsExchange,
    symbols: List[str],
    strategy: Optional[IsStrategy] = None,
) -> None:
    """Fetch historical price data for the parsed symbols.

    :param exchange: an instance of an interface of the exchange to use
        to fetch historical price data.
    :param symbols: symbol names to fetch historical price data for.
    :param strategy: optional strategy instance.
    :return: None
    """

    if not symbols and strategy:
        symbols = strategy.config.get("watchlist", dict()).get(exchange.name, [])
    if not symbols:
        logger.warning("No symbols found to fetch historical price data")
        return

    for symbol in symbols:
        if symbol not in exchange.exchange_symbols:
            logger.error(
                "%s %s: provided symbol not present in the exchange",
                exchange.display_name,
                symbol,
            )
            return None

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        executor.map(
            fetch_symbol_data,
            [exchange] * len(symbols),
            [symbol for symbol in symbols],
        )
