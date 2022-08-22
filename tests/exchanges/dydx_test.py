from typing import Any, Dict

import pytest

from afang.exchanges.dydx import DyDxExchange
from afang.exchanges.models import Candle, HTTPMethod, Symbol
from afang.models import Timeframe


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
    assert dydx_exchange.testnet is False
    assert dydx_exchange._base_url == "https://api.dydx.exchange"
    assert dydx_exchange.symbols == ["BTCUSDT", "ETHUSDT"]
    assert dydx_exchange.get_config_params() == {
        "query_limit": 0.2,
        "write_limit": 20000,
    }

    dydx_exchange_testnet = DyDxExchange(testnet=True)
    assert dydx_exchange_testnet.testnet is True
    assert dydx_exchange_testnet._base_url == "https://api.stage.dydx.exchange"


@pytest.mark.parametrize(
    "req_response, expected_symbols",
    [
        (
            {
                "markets": {
                    "BTC-USD": {
                        "market": "BTC-USD",
                        "type": "PERPETUAL",
                        "baseAsset": "BTC",
                        "quoteAsset": "USD",
                        "stepSize": "0.1",
                        "tickSize": "0.01",
                    },
                    "LINK-USD": {
                        "market": "LINK-USD",
                        "type": "NOT_PERPETUAL",
                    },
                    "LTC-USD": {
                        "market": "LTC-USD",
                        "type": "PERPETUAL",
                        "baseAsset": "LTC",
                        "quoteAsset": "USD",
                        "stepSize": "0.1",
                        "tickSize": "0.01",
                    },
                },
            },
            {
                "BTC-USD": Symbol(
                    name="BTC-USD",
                    base_asset="BTC",
                    quote_asset="USD",
                    price_decimals=2,
                    quantity_decimals=1,
                    tick_size=0.01,
                    step_size=0.1,
                ),
                "LTC-USD": Symbol(
                    name="LTC-USD",
                    base_asset="LTC",
                    quote_asset="USD",
                    price_decimals=2,
                    quantity_decimals=1,
                    tick_size=0.01,
                    step_size=0.1,
                ),
            },
        ),
        ({}, dict()),
        (None, dict()),
    ],
)
def test_get_symbols(mocker, req_response, expected_symbols) -> None:
    # mock the return value of the _make_request function.
    def mock_make_request(
        _self, _method: HTTPMethod, _endpoint: str, _query_parameters: Dict
    ) -> Any:
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
                Candle(1578182400000, 1.0, 2.0, 3.0, 4.0, 5.0),
                Candle(1609804800000, 10.0, 11.0, 12.0, 13.0, 14.0),
            ],
        ),
        (None, None),
        ({}, None),
    ],
)
def test_get_historical_candles(mocker, req_response, expected_candles) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._get_symbols",
        mock_get_symbols,
    )

    # mock the return value of the _make_request function.
    def mock_make_request(
        _self, _method: HTTPMethod, _endpoint: str, _query_parameters: Dict
    ) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.dydx.DyDxExchange._make_request",
        mock_make_request,
    )

    dydx_exchange = DyDxExchange()
    assert (
        dydx_exchange.get_historical_candles("test_symbol", 2, 100) == expected_candles
    )


def test_get_historical_candles_unknown_timeframe(mocker, caplog) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    binance_exchange = DyDxExchange()
    binance_exchange.get_historical_candles("test_symbol", 2, 100, Timeframe.M3)

    assert caplog.records[0].levelname == "ERROR"
    assert (
        "dydx cannot fetch historical candles in 3m intervals. Please use another timeframe"
        in caplog.text
    )
