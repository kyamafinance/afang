import pytest

from afang.main import main


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
    mocked_collect_all = mocker.patch(
        "afang.main.fetch_historical_price_data", return_value=None
    )
    main(args)

    assert mocked_collect_all.assert_called
