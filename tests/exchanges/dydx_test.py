from typing import Any, Dict

import pytest

from afang.exchanges.dydx import DyDxExchange


def test_dydx_exchange_init(mocker) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["BTCUSDT", "ETHUSDT"]

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._get_symbols",
        mock_get_symbols,
    )

    dydx_exchange = DyDxExchange()
    assert dydx_exchange.name == "dydx"
    assert dydx_exchange._base_url == "https://api.dydx.exchange"
    assert dydx_exchange.symbols == ["BTCUSDT", "ETHUSDT"]
    assert dydx_exchange.get_config_params() == {
        "query_limit": 0.2,
        "write_limit": 20000,
    }


@pytest.mark.parametrize(
    "req_response, expected_symbols",
    [
        (
            {
                "markets": {
                    "BTC-USD": {
                        "market": "LINK-USD",
                        "type": "PERPETUAL",
                    },
                    "LINK-USD": {
                        "market": "LINK-USD",
                        "type": "NOT_PERPETUAL",
                    },
                    "LTC-USD": {
                        "market": "LINK-USD",
                        "type": "PERPETUAL",
                    },
                },
            },
            ["BTC-USD", "LTC-USD"],
        ),
        ({}, []),
        (None, []),
    ],
)
def test_get_symbols(mocker, req_response, expected_symbols) -> None:
    # mock the return value of the _make_request function.
    def mock_make_request(_self, _endpoint: str, _query_parameters: Dict) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._make_request",
        mock_make_request,
    )

    dydx_exchange = DyDxExchange()
    assert dydx_exchange.symbols == expected_symbols


@pytest.mark.parametrize(
    "req_response, expected_candles",
    [
        (
            {
                "candles": [
                    {
                        "startedAt": "2021-01-05T00:00:00.000Z",
                        "updatedAt": "2021-01-05T00:00:00.000Z",
                        "market": "BTC-USD",
                        "resolution": "1DAY",
                        "low": "12",
                        "high": "11",
                        "open": "10",
                        "close": "13",
                        "baseTokenVolume": "1.002",
                        "trades": "3",
                        "usdVolume": "14",
                        "startingOpenInterest": "28",
                    },
                    {
                        "startedAt": "2020-01-05T00:00:00.000Z",
                        "updatedAt": "2021-01-05T00:00:00.000Z",
                        "market": "BTC-USD",
                        "resolution": "1DAY",
                        "low": "3",
                        "high": "2",
                        "open": "1",
                        "close": "4",
                        "baseTokenVolume": "1.002",
                        "trades": "3",
                        "usdVolume": "5",
                        "startingOpenInterest": "28",
                    },
                ]
            },
            [
                (1578182400000.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                (1609804800000, 10.0, 11.0, 12.0, 13.0, 14.0),
            ],
        ),
        (None, None),
        ({}, None),
    ],
)
def test_get_historical_data(mocker, req_response, expected_candles) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._get_symbols",
        mock_get_symbols,
    )

    # mock the return value of the _make_request function.
    def mock_make_request(_self, _endpoint: str, _query_parameters: Dict) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._make_request",
        mock_make_request,
    )

    dydx_exchange = DyDxExchange()
    assert dydx_exchange.get_historical_data("test_symbol", 2, 100) == expected_candles