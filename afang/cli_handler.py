import argparse


def parse_args(args) -> argparse.Namespace:
    """Parse application command line arguments.

    :param args: command line arguments to parse.

    :return: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description="A Python-based platform for backtesting and optimizing automated trading systems"
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="program mode",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--exchange",
        type=str,
        help="exchange to use",
        required=True,
    )
    parser.add_argument(
        "--testnet",
        type=bool,
        help="whether to use the testnet version of the exchange",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=[],
        help="list of symbols to use",
    )
    parser.add_argument("--strategy", type=str, help="strategy to use")
    parser.add_argument(
        "--timeframe",
        type=str,
        help="timeframe to use",
    )
    parser.add_argument(
        "--from-time",
        type=str,
        help="backtest from time. format: (yyyy-mm-dd). if not provided, backtest will start from the beginning of "
        "the available price data",
    )
    parser.add_argument(
        "--to-time",
        type=str,
        help="backtest to time. format: (yyyy-mm-dd). if not provided, backtest will complete at the end of the "
        "available price data",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args
