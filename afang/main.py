import argparse
import logging
import multiprocessing
import sys
from typing import Optional

from afang.cli_handler import parse_args
from afang.database.backtest_data_collector import collect_all
from afang.exchanges import BinanceExchange, DyDxExchange, IsExchange

logger = logging.getLogger(__name__)


def get_exchange_client(exchange_arg: str) -> Optional[IsExchange]:
    """Get the proper exchange client given the exchange's name.

    :param exchange_arg: name of the exchange client.

    :return: Optional[IsExchange]
    """

    exchange: Optional[IsExchange] = None
    if exchange_arg == "binance":
        exchange = BinanceExchange()
    elif exchange_arg == "dydx":
        exchange = DyDxExchange()

    return exchange


def fetch_historical_price_data(
    exchange: IsExchange, parsed_args: argparse.Namespace
) -> None:
    """Fetch historical price data for the parsed symbols.

    :param exchange: an instance of an interface of the exchange to use to fetch historical price data.
    :param parsed_args: arguments parsed from the CLI.

    :return: None
    """

    symbols = parsed_args.symbols
    if not symbols:
        logger.warning("No symbols found to fetch historical price data")
        return

    client_config = exchange.get_config_params()
    query_limit = client_config.get("query_limit")
    write_limit = client_config.get("write_limit")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for symbol in symbols:
        pool.apply_async(
            collect_all,
            (exchange, symbol, None, query_limit, write_limit),
        )

    pool.close()
    pool.join()


def main(args):
    """Parse command line arguments and run the desired functionality based on
    the provided application mode.

    :param args: command line arguments to parse.

    :return: None
    """

    parsed_args = parse_args(args)

    # Define the exchange client.
    exchange = get_exchange_client(parsed_args.exchange)
    if not exchange:
        logger.warning("Unknown exchange provided: %s", parsed_args.exchange)
        return

    if parsed_args.mode == "data":
        # If the provided mode is data, collect historical price data.
        fetch_historical_price_data(exchange, parsed_args)

    else:
        logger.warning("Unknown mode provided: %s", parsed_args.mode)
        return


if __name__ == "__main__":
    main(args=sys.argv[1:])
