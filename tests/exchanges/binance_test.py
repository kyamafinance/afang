from typing import Any, Dict

import pytest

from afang.exchanges.binance import BinanceExchange


def test_binance_exchange_init(mocker) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["BTCUSDT", "ETHUSDT"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    binance_exchange = BinanceExchange()
    assert binance_exchange.name == "binance"
    assert binance_exchange._base_url == "https://fapi.binance.com"
    assert binance_exchange.symbols == ["BTCUSDT", "ETHUSDT"]


@pytest.mark.parametrize(
    "req_response, expected_symbols",
    [
        (
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "contractType": "PERPETUAL",
                    },
                    {
                        "symbol": "LINKBTC",
                        "contractType": "SPOT",
                    },
                    {
                        "symbol": "LTCUSDT",
                        "contractType": "PERPETUAL",
                    },
                ]
            },
            ["BTCUSDT", "LTCUSDT"],
        ),
        (None, []),
    ],
)
def test_get_symbols(mocker, req_response, expected_symbols) -> None:
    # mock the return value of the _make_request function.
    def mock_make_request(_self, _endpoint: str, _query_parameters: Dict) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request",
        mock_make_request,
    )

    binance_exchange = BinanceExchange()
    assert binance_exchange.symbols == expected_symbols


@pytest.mark.parametrize(
    "req_response, expected_candles",
    [
        (
            [
                [
                    1,
                    "1.5",
                    "2.5",
                    "3.5",
                    "4.5",
                    "5.5",
                    900,
                    "6.4",
                    30,
                    "7.8",
                    "9.8",
                    "7.7",
                ],
                [
                    9,
                    "4.5",
                    "4.5",
                    "2.5",
                    "6.5",
                    "7.5",
                    540,
                    "4.4",
                    50,
                    "1.8",
                    "2.8",
                    "1.7",
                ],
            ],
            [(1, 1.5, 2.5, 3.5, 4.5, 5.5), (9, 4.5, 4.5, 2.5, 6.5, 7.5)],
        ),
        (None, None),
    ],
)
def test_get_historical_data(mocker, req_response, expected_candles) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    # mock the return value of the _make_request function.
    def mock_make_request(_self, _endpoint: str, _query_parameters: Dict) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request",
        mock_make_request,
    )

    binance_exchange = BinanceExchange()
    assert (
        binance_exchange.get_historical_data("test_symbol", 0, 100) == expected_candles
    )
