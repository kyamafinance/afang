import logging
import sys
from typing import Optional

from afang.cli_handler import parse_args
from afang.database.backtest_data_collector import fetch_historical_price_data
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
