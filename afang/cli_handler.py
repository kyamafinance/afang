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
        "--symbols",
        type=str,
        nargs="+",
        default=[],
        help="list of symbols to use",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args
