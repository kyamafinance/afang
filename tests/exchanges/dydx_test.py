from types import SimpleNamespace
from typing import Any, Dict

import pytest
from dydx3 import DydxApiError

from afang.exchanges.dydx import DyDxExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
)
from afang.models import Mode, Timeframe


@pytest.fixture(autouse=True)
def mock_dydx_get_api_client(mocker):
    mocker.patch("afang.exchanges.dydx.DyDxExchange._get_api_client")


@pytest.fixture
def mock_dydx_api_client():
    class DydxApiErrorResponse:
        def __init__(self):
            self.status_code = 400
            self.text = "DydxApiErrorResponse"

        @classmethod
        def json(cls):
            return {}

    class Private:
        def __init__(self, should_raise_exception):
            self.should_raise_exception = should_raise_exception

        def create_order(self, **_kwargs):
            if self.should_raise_exception:
                raise DydxApiError(response=DydxApiErrorResponse())

            order = {"order": {"id": "12345"}}
            data = dict(data=order)
            return SimpleNamespace(**data)

        @classmethod
        def get_fills(cls, _market, _order_id, _limit):
            fills = {
                "fills": [
                    {"size": "10", "price": "11", "fee": 2},
                    {"size": "12", "price": "13", "fee": 1},
                ]
            }
            data = dict(data=fills)
            return SimpleNamespace(**data)

        def get_order_by_id(self, _order_id):
            if self.should_raise_exception:
                raise DydxApiError(response=DydxApiErrorResponse())

            order = {
                "order": {
                    "size": "12",
                    "remainingSize": "10",
                    "side": "BUY",
                    "type": "MARKET",
                    "price": "102",
                    "status": "OPEN",
                    "timeInForce": "GTT",
                }
            }
            data = dict(data=order)
            return SimpleNamespace(**data)

        def cancel_order(self, _order_id):
            if self.should_raise_exception:
                raise DydxApiError(response=DydxApiErrorResponse())

            return True

    class ApiClient:
        def __init__(self, should_raise_exception):
            self.should_raise_exception = should_raise_exception
            self.private = Private(self.should_raise_exception)

    return ApiClient


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
    assert dydx_exchange.mode is None
    assert dydx_exchange.display_name == "dydx"
    assert dydx_exchange.testnet is False
    assert dydx_exchange._base_url == "https://api.dydx.exchange"
    assert dydx_exchange._wss_url == "wss://api.dydx.exchange/v3/ws"
    assert dydx_exchange.exchange_symbols == ["BTCUSDT", "ETHUSDT"]
    assert dydx_exchange.get_config_params() == {
        "query_limit": 0.2,
        "write_limit": 20000,
    }

    dydx_exchange_testnet = DyDxExchange(testnet=True)
    assert dydx_exchange_testnet.mode is None
    assert dydx_exchange_testnet.display_name == "dydx-testnet"
    assert dydx_exchange_testnet.testnet is True
    assert dydx_exchange_testnet._base_url == "https://api.stage.dydx.exchange"
    assert dydx_exchange_testnet._wss_url == "wss://api.stage.dydx.exchange/v3/ws"

    dydx_exchange_data_mode = DyDxExchange(mode=Mode.data)
    assert dydx_exchange_data_mode.mode == Mode.data


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
    assert dydx_exchange.exchange_symbols == expected_symbols


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
        "afang.exchanges.dydx.DyDxExchange._get_symbols",
        mock_get_symbols,
    )

    dydx_exchange = DyDxExchange()
    dydx_exchange.get_historical_candles("test_symbol", 2, 100, Timeframe.M3)

    assert caplog.records[0].levelname == "ERROR"
    assert (
        "dydx cannot fetch historical candles in 3m intervals. Please use another timeframe"
        in caplog.text
    )


def test_get_api_client(mocker) -> None:
    mocked_dydx3_client = mocker.patch("afang.exchanges.dydx.Client")
    mocked_client_get_account = mocker.patch(
        "afang.exchanges.dydx.Client.private.get_account"
    )

    dydx_exchange = DyDxExchange()
    dydx_exchange._get_api_client()

    assert mocked_dydx3_client.assert_called_once
    assert mocked_client_get_account.assert_called_once


@pytest.mark.parametrize("should_raise_exception", [True, False])
def test_place_order(caplog, mock_dydx_api_client, should_raise_exception) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.mode = Mode.trade
    dydx_exchange._api_client = mock_dydx_api_client(should_raise_exception)
    order_id = dydx_exchange.place_order(
        "BTC-USD", OrderSide.BUY, 0.001, OrderType.LIMIT, 10
    )

    if should_raise_exception:
        assert caplog.records[-1].levelname == "ERROR"
        assert (
            "DydxApiError raised when attempting to place new BTC-USD order"
            in caplog.text
        )
        return

    assert order_id == "12345"


@pytest.mark.parametrize("should_raise_exception", [True, False])
def test_get_order_by_id(caplog, mock_dydx_api_client, should_raise_exception) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._api_client = mock_dydx_api_client(should_raise_exception)
    order = dydx_exchange.get_order_by_id("BTC-USD", "12345")

    if should_raise_exception:
        assert caplog.records[-1].levelname == "ERROR"
        assert (
            "DydxApiError raised when attempting to get BTC-USD order status with ID 12345"
            in caplog.text
        )
        return

    assert order == Order(
        symbol="BTC-USD",
        order_id="12345",
        side=OrderSide.BUY,
        original_price=102.0,
        average_price=12.09090909090909,
        original_quantity=12.0,
        executed_quantity=2.0,
        remaining_quantity=10.0,
        order_type=OrderType.MARKET,
        order_status="OPEN",
        time_in_force="GTT",
        commission=3,
    )


@pytest.mark.parametrize("should_raise_exception", [True, False])
def test_cancel_order(caplog, mock_dydx_api_client, should_raise_exception) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._api_client = mock_dydx_api_client(should_raise_exception)
    order_placed = dydx_exchange.cancel_order("BTC-USD", "12345")

    if should_raise_exception:
        assert caplog.records[-1].levelname == "ERROR"
        assert (
            "DydxApiError raised when attempting to cancel BTC-USD order with ID 12345"
            in caplog.text
        )
        return

    assert order_placed is True
