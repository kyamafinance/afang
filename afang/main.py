import argparse
import logging
import sys
from typing import Callable, Optional

import user_strategies as user_strategies
from afang.cli_handler import parse_args
from afang.database.ohlcv_db.ohlcv_data_collector import fetch_historical_price_data
from afang.exchanges import BinanceExchange, DyDxExchange, IsExchange
from afang.models import Exchange, Mode
from afang.strategies.optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


def get_exchange_client(parsed_args: argparse.Namespace) -> Optional[IsExchange]:
    """Get the proper exchange client given the exchange's name.

    :param parsed_args: arguments parsed from the CLI.
    :return: Optional[IsExchange]
    """

    exchange: Optional[IsExchange] = None
    if parsed_args.exchange == Exchange.binance.value:
        exchange = BinanceExchange(testnet=parsed_args.testnet)
    elif parsed_args.exchange == Exchange.dydx.value:
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
        strategy_module = __import__(
            f"{user_strategies.__name__}.{strategy_name}.{strategy_name}",
            fromlist=[strategy_name],
        )
        strategy_instance = getattr(strategy_module, strategy_name)
        return strategy_instance
    except (ModuleNotFoundError, AttributeError):
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

    if parsed_args.mode == Mode.data.value:
        # If the provided mode is data, collect historical price data.
        fetch_historical_price_data(
            exchange, parsed_args.symbols, strategy=strategy() if strategy else None
        )
        return

    if not strategy:
        logger.warning("Trading strategy not provided")
        return

    if parsed_args.mode == Mode.backtest.value:
        # If the mode provided is backtest, run a backtest on the provided strategy
        strategy().run_backtest(
            exchange,
            parsed_args.symbols,
            parsed_args.timeframe,
            parsed_args.from_time,
            parsed_args.to_time,
        )

    elif parsed_args.mode == Mode.optimize.value:
        # Optimize trading strategy parameters.
        StrategyOptimizer(
            strategy,
            exchange,
            parsed_args.symbols,
            parsed_args.timeframe,
            parsed_args.from_time,
            parsed_args.to_time,
        ).optimize()

    elif parsed_args.mode == Mode.trade.value:
        # Run the trader on the provided strategy.
        strategy().run_trader(
            exchange,
            parsed_args.symbols,
            parsed_args.timeframe,
            demo_mode=parsed_args.demo,
        )

    else:
        logger.warning("Unknown mode provided: %s", parsed_args.mode)
        return


if __name__ == "__main__":
    main(args=sys.argv[1:])
