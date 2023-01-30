import argparse
from enum import Enum

import pytest

from afang.exchanges import BinanceExchange, DyDxExchange
from afang.main import get_exchange_client, get_strategy_instance, main
from afang.strategies.SampleStrategy.SampleStrategy import SampleStrategy


@pytest.fixture(autouse=True)
def mock_dydx_get_api_client(mocker):
    mocker.patch("afang.main.DyDxExchange._get_api_client")


@pytest.mark.parametrize(
    "args, expected_log",
    [
        (["-m", "unknown_mode", "-e", "dydx"], "Unknown mode provided"),
        (["-m", "data", "-e", "unknown_exchange"], "Unknown exchange provided"),
    ],
)
def test_unknown_inputs(mocker, args, expected_log, caplog) -> None:
    class MockedMode(Enum):
        data = "data"
        unknown_mode = "unknown_mode"

    class MockedExchange(Enum):
        dydx = "dydx"
        unknown_exchange = "unknown_exchange"

    mocker.patch("afang.cli_handler.Mode", MockedMode)
    mocker.patch("afang.cli_handler.Exchange", MockedExchange)

    main(args)

    assert caplog.records[-1].levelname == "WARNING"
    assert expected_log in caplog.text


def test_fetch_historical_price_data(mocker, dummy_is_exchange) -> None:
    args = ["-m", "data", "-e", "dydx", "--symbols", "BTC-USD"]
    mocked_fetch_historical_price_data = mocker.patch(
        "afang.main.fetch_historical_price_data", return_value=None
    )
    main(args)

    assert mocked_fetch_historical_price_data.assert_called


def test_get_strategy_instance_undefined_strategy() -> None:
    undefined_strategy = "UndefinedStrategy"
    with pytest.raises(
        ValueError, match=f"Unknown strategy name provided: {undefined_strategy}"
    ):
        get_strategy_instance(undefined_strategy)


@pytest.mark.parametrize(
    "parsed_args, expected_exchange_client",
    [
        (argparse.Namespace(exchange="binance", testnet=False), "binance"),
        (
            argparse.Namespace(exchange="binance", testnet=True),
            "binance-testnet",
        ),
        (argparse.Namespace(exchange="dydx", testnet=False), "dydx"),
        (argparse.Namespace(exchange="dydx", testnet=True), "dydx-testnet"),
        (argparse.Namespace(exchange="unknown", testnet=False), None),
        (argparse.Namespace(exchange="unknown", testnet=True), None),
    ],
)
def test_get_exchange_client(mocker, parsed_args, expected_exchange_client) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["BTCUSDT", "ETHUSDT"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._get_symbols",
        mock_get_symbols,
    )

    if expected_exchange_client == "binance":
        expected_exchange_client = BinanceExchange()
    elif expected_exchange_client == "binance-testnet":
        expected_exchange_client = BinanceExchange(testnet=True)
    elif expected_exchange_client == "dydx":
        expected_exchange_client = DyDxExchange()
    elif expected_exchange_client == "dydx-testnet":
        expected_exchange_client = DyDxExchange(testnet=True)

    exchange_client = get_exchange_client(parsed_args)

    if not expected_exchange_client:
        assert expected_exchange_client is None
        return

    assert exchange_client.name == expected_exchange_client.name
    assert exchange_client.testnet == expected_exchange_client.testnet
    assert exchange_client._base_url == expected_exchange_client._base_url


def test_get_strategy_instance() -> None:
    strategy_name = "SampleStrategy"
    assert get_strategy_instance(strategy_name) == SampleStrategy


def test_run_backtest(mocker) -> None:
    args = [
        "-m",
        "backtest",
        "-e",
        "dydx",
        "--symbols",
        "BTC-USD",
        "--strategy",
        "SampleStrategy",
    ]
    mocked_run_backtest = mocker.patch(
        "afang.strategies.is_strategy.IsStrategy.run_backtest", return_value=None
    )
    main(args)

    assert mocked_run_backtest.assert_called


def test_run_strategy_optimization(mocker) -> None:
    args = [
        "-m",
        "optimize",
        "-e",
        "dydx",
        "--symbols",
        "BTC-USD",
        "--strategy",
        "SampleStrategy",
    ]

    mocked_strategy_optimize = mocker.patch(
        "afang.main.StrategyOptimizer.optimize", return_value=None
    )
    main(args)

    assert mocked_strategy_optimize.assert_called


def test_run_strategy_trader(mocker) -> None:
    args = [
        "-m",
        "trade",
        "-e",
        "dydx",
        "--symbols",
        "BTC-USD",
        "--strategy",
        "SampleStrategy",
    ]

    mocked_strategy_trader = mocker.patch(
        "afang.strategies.is_strategy.IsStrategy.run_trader", return_value=None
    )
    main(args)

    assert mocked_strategy_trader.assert_called
