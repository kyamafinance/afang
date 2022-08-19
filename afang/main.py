import argparse
import logging
import operator
import sys
from typing import Callable, Optional

import afang.strategies as strategies
from afang.cli_handler import parse_args
from afang.database.backtest_data_collector import fetch_historical_price_data
from afang.exchanges import BinanceExchange, DyDxExchange, IsExchange
from afang.strategies.optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


def get_exchange_client(parsed_args: argparse.Namespace) -> Optional[IsExchange]:
    """Get the proper exchange client given the exchange's name.

    :param parsed_args: arguments parsed from the CLI.

    :return: Optional[IsExchange]
    """

    exchange: Optional[IsExchange] = None
    if parsed_args.exchange == "binance":
        exchange = BinanceExchange(testnet=parsed_args.testnet)
    elif parsed_args.exchange == "dydx":
        exchange = DyDxExchange(testnet=parsed_args.testnet)

    return exchange


def get_strategy_instance(strategy_name: str) -> Optional[Callable]:
    """Returns a callable strategy instance. If the strategy name does not
    correspond to a properly defined strategy, a ValueError is raised.

    :param strategy_name: name of the user defined strategy.
    :return: Callable
    """

    if not strategy_name:
        return None

    try:
        return operator.attrgetter(f"{strategy_name}.{strategy_name}")(strategies)
    except AttributeError:
        raise ValueError(f"Unknown strategy name provided: {strategy_name}")


def main(args):
    """Parse command line arguments and run the desired functionality based on
    the provided application mode.

    :param args: command line arguments to parse.

    :return: None
    """

    parsed_args = parse_args(args)

    # Get the exchange client.
    exchange = get_exchange_client(parsed_args)
    if not exchange:
        logger.warning("Unknown exchange provided: %s", parsed_args.exchange)
        return

    # Get the strategy instance if one was specified.
    strategy = get_strategy_instance(parsed_args.strategy)

    if parsed_args.mode == "data":
        # If the provided mode is data, collect historical price data.
        fetch_historical_price_data(
            exchange, parsed_args, strategy=strategy() if strategy else None
        )

    elif parsed_args.mode == "backtest":
        # If the mode provided is backtest, run a backtest on the provided strategy
        strategy().run_backtest(exchange, parsed_args)

    elif parsed_args.mode == "optimize":
        # Optimize trading strategy parameters.
        StrategyOptimizer(strategy, exchange, parsed_args).optimize()

    else:
        logger.warning("Unknown mode provided: %s", parsed_args.mode)
        return


if __name__ == "__main__":
    main(args=sys.argv[1:])
