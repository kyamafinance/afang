import pytest

from afang.main import get_strategy_instance, main
from afang.strategies.SampleStrategy.SampleStrategy import SampleStrategy


@pytest.mark.parametrize(
    "args, expected_log",
    [
        (["-m", "unknown_mode", "-e", "dydx"], "Unknown mode provided"),
        (["-m", "data", "-e", "unknown_exchange"], "Unknown exchange provided"),
    ],
)
def test_unknown_inputs(args, expected_log, caplog) -> None:
    main(args)

    assert caplog.records[0].levelname == "WARNING"
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
