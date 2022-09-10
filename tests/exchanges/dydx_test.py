import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import websocket
from dydx3 import DydxApiError

from afang.exchanges.dydx import DyDxExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
    SymbolBalance,
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

        def get_account(self):
            if self.should_raise_exception:
                raise DydxApiError(response=DydxApiErrorResponse())

            account = {"account": {"openPositions": {"BTC-USD": {"size": "300"}}}}
            data = dict(data=account)
            return SimpleNamespace(**data)

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

        def sign(self, **kwargs):
            pass

    class ApiClient:
        def __init__(self, should_raise_exception=False):
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
    dydx_exchange.trading_price_data["BTC-USD"] = [
        Candle(
            open_time=1,
            open=1,
            high=2,
            low=3,
            close=3,
            volume=4,
        )
    ]
    dydx_exchange._api_client = mock_dydx_api_client(should_raise_exception)
    order_id = dydx_exchange.place_order(
        "BTC-USD", OrderSide.BUY, 0.001, OrderType.MARKET, 10
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


def test_wss_subscribe_trades_stream() -> None:
    class MockWSS:
        def __init__(self):
            self.send_count = 0

        def send(self, _val):
            self.send_count += 1

    mock_wss = MockWSS()

    dydx_exchange = DyDxExchange()
    dydx_exchange._wss = mock_wss
    dydx_exchange.trading_symbols = {
        "BTC-USD": Symbol(
            base_asset="BTC",
            quote_asset="USD",
            name="BTC-USD",
            price_decimals=2,
            quantity_decimals=2,
            step_size=2,
            tick_size=2,
        ),
        "ETH-USD": Symbol(
            base_asset="ETH",
            quote_asset="USD",
            name="ETH-USD",
            price_decimals=2,
            quantity_decimals=2,
            step_size=2,
            tick_size=2,
        ),
    }
    dydx_exchange._wss_subscribe_trades_stream()

    assert mock_wss.send_count == 2


def test_wss_subscribe_markets_stream() -> None:
    class MockWSS:
        def __init__(self):
            self.send_count = 0

        def send(self, _val):
            self.send_count += 1

    mock_wss = MockWSS()

    dydx_exchange = DyDxExchange()
    dydx_exchange._wss = mock_wss
    dydx_exchange._wss_subscribe_markets_stream()

    assert mock_wss.send_count == 1


def test_wss_subscribe_accounts_stream(mock_dydx_api_client) -> None:
    class MockWSS:
        def __init__(self):
            self.send_count = 0

        def send(self, _val):
            self.send_count += 1

    mock_wss = MockWSS()

    dydx_exchange = DyDxExchange()
    dydx_exchange._wss = mock_wss
    dydx_exchange._api_client = mock_dydx_api_client()
    dydx_exchange._wss_subscribe_accounts_stream()

    assert mock_wss.send_count == 1


def test_wss_on_open(mocker) -> None:
    mocked_subscribe_trades_stream = mocker.patch.object(
        DyDxExchange, "_wss_subscribe_trades_stream"
    )
    mocked_subscribe_markets_stream = mocker.patch.object(
        DyDxExchange, "_wss_subscribe_markets_stream"
    )
    mocked_subscribe_accounts_stream = mocker.patch.object(
        DyDxExchange, "_wss_subscribe_accounts_stream"
    )

    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_on_open(websocket.WebSocketApp("fake-url"))

    assert mocked_subscribe_trades_stream.assert_called_once
    assert mocked_subscribe_markets_stream.assert_called_once
    assert mocked_subscribe_accounts_stream.assert_called_once


def test_wss_on_close(caplog) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_on_close(websocket.WebSocketApp("fake-url"))

    assert caplog.records[0].levelname == "WARNING"
    assert "wss connection closed" in caplog.text


def test_wss_on_error(caplog) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_on_error(websocket.WebSocketApp("fake-url"), "message")

    assert caplog.records[0].levelname == "ERROR"
    assert "wss connection error" in caplog.text


def test_update_collateral_balance() -> None:
    dydx_exchange = DyDxExchange()

    # no dydx_exchange._symbol_total_pos_size
    dydx_exchange._update_collateral_balance(run_forever=False)
    assert "USD" not in dydx_exchange.trading_symbol_balance
    dydx_exchange._symbol_total_pos_size = {"BTC-USD": 100, "ETH-USD": 200}

    # no dydx_exchange._quote_balance
    dydx_exchange._update_collateral_balance(run_forever=False)
    assert "USD" not in dydx_exchange.trading_symbol_balance
    dydx_exchange._quote_balance = 400

    # no dydx_exchange._oracle_prices
    dydx_exchange._update_collateral_balance(run_forever=False)
    assert "USD" not in dydx_exchange.trading_symbol_balance
    dydx_exchange._oracle_prices = {"BTC-USD": 20, "ETH-USD": 50}

    dydx_exchange._update_collateral_balance(run_forever=False)
    assert "USD" in dydx_exchange.trading_symbol_balance
    assert dydx_exchange.trading_symbol_balance["USD"] == SymbolBalance(
        name="USD", wallet_balance=12400
    )


def test_wss_handle_oracle_price_update_no_contents() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_oracle_price_update({})
    assert not dydx_exchange._oracle_prices


def test_wss_handle_oracle_price_update() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_oracle_price_update(
        {"contents": {"markets": {"BTC-USD": {"oraclePrice": 20}}}}
    )

    assert dydx_exchange._oracle_prices["BTC-USD"] == 20


def test_wss_handle_collateral_balance_update_no_contents() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_collateral_balance_update({})
    assert not dydx_exchange.trading_symbol_balance


def test_wss_handle_collateral_balance_update() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_collateral_balance_update(
        {
            "contents": {
                "account": {
                    "quoteBalance": 300,  # should be overridden by the value in accounts
                },
                "accounts": [
                    {"accountNumber": 0, "quoteBalance": 500},
                    {"accountNumber": 1},
                ],
                "positions": [
                    {"market": "BTC-USD", "size": 100},
                    {"market": "ETH-USD", "size": 100},
                ],
            }
        }
    )

    assert "BTC-USD" in dydx_exchange._symbol_total_pos_size
    assert "ETH-USD" in dydx_exchange._symbol_total_pos_size


def test_wss_handle_collateral_balance_update_with_equity() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_collateral_balance_update(
        {
            "contents": {
                "account": {
                    "quoteBalance": 300,  # should be overridden by the value in accounts
                    "equity": 700,
                },
                "accounts": [
                    {"accountNumber": 0, "quoteBalance": 500},
                    {"accountNumber": 1},
                ],
            }
        }
    )

    assert not dydx_exchange._symbol_total_pos_size
    assert "USD" in dydx_exchange.trading_symbol_balance
    assert dydx_exchange.trading_symbol_balance["USD"].wallet_balance == 700


def test_wss_handle_order_update_no_contents() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_order_update({})
    assert not dydx_exchange.active_orders


def test_wss_handle_order_update() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.active_orders["12345"] = Order(
        symbol="BTC-USD",
        order_id="12345",
        side=OrderSide.SELL,
        original_price=13.4,
        average_price=12.77,
        original_quantity=12.5,
        executed_quantity=2.5,
        remaining_quantity=10,
        order_type=OrderType.MARKET,
        order_status="FILLED",
        time_in_force="GTT",
        commission=4,
    )
    dydx_exchange._wss_handle_order_update(
        {
            "contents": {
                "orders": [
                    {
                        "market": "BTC-USD",
                        "price": "12",
                        "id": "12345",
                        "size": "50",
                        "remainingSize": "10",
                        "side": "SELL",
                        "type": "MARKET",
                        "status": "FILLED",
                        "timeInForce": "GTT",
                    }
                ],
                "fills": [
                    {"orderId": "12345", "size": "5", "price": "13.0", "fee": "1"}
                ],
            }
        }
    )

    assert dydx_exchange.active_orders["12345"] == Order(
        symbol="BTC-USD",
        order_id="12345",
        side=OrderSide.SELL,
        original_price=12.0,
        average_price=12.923333333333332,
        original_quantity=50.0,
        executed_quantity=40.0,
        remaining_quantity=10,
        order_type=OrderType.MARKET,
        order_status="FILLED",
        time_in_force="GTT",
        commission=5,
    )


def test_wss_handle_candlestick_update_no_contents() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_handle_candlestick_update({})
    assert not dydx_exchange.trading_price_data


def test_wss_handle_candlestick_update_few_trading_price_data(caplog) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.trading_price_data["BTC-USD"] = []

    with pytest.raises(ValueError):
        dydx_exchange._wss_handle_candlestick_update(
            {"id": "BTC-USD", "contents": {"trades": []}}
        )

        assert caplog.records[0].levelname == "ERROR"
        assert (
            "candles cannot be updated because there are too few initial candles"
            in caplog.text
        )


def test_wss_handle_candlestick_update_update_current_candle() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.trading_price_data["BTC-USD"] = [
        Candle(
            open_time=1662774258000,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
        ),
        Candle(
            open_time=1662774318000,
            open=6,
            high=7,
            low=8,
            close=9,
            volume=10,
        ),
    ]

    dydx_exchange._wss_handle_candlestick_update(
        {
            "id": "BTC-USD",
            "contents": {
                "trades": [
                    {
                        "createdAt": "2022-09-10T01:45:18.000Z",
                        "price": "30",
                        "size": "10",
                    }
                ]
            },
        }
    )

    assert len(dydx_exchange.trading_price_data["BTC-USD"]) == 2
    assert dydx_exchange.trading_price_data["BTC-USD"][-1] == Candle(
        open_time=1662774318000,
        open=6,
        high=30,
        low=8,
        close=30,
        volume=20,
    )


def test_wss_handle_candlestick_update_add_new_candle() -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.trading_price_data["BTC-USD"] = [
        Candle(
            open_time=1662774258000,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
        ),
        Candle(
            open_time=1662774318000,
            open=6,
            high=7,
            low=8,
            close=9,
            volume=10,
        ),
    ]

    dydx_exchange._wss_handle_candlestick_update(
        {
            "id": "BTC-USD",
            "contents": {
                "trades": [
                    {
                        "createdAt": "2022-09-10T01:46:20.000Z",
                        "price": "30",
                        "size": "10",
                    }
                ]
            },
        }
    )

    assert len(dydx_exchange.trading_price_data["BTC-USD"]) == 2
    assert dydx_exchange.trading_price_data["BTC-USD"][-1] == Candle(
        open_time=1662774378000,
        open=30,
        high=30,
        low=30,
        close=30,
        volume=10,
    )


def test_wss_handle_candlestick_update_invalid_candle(caplog) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.trading_timeframe = Timeframe.M1
    dydx_exchange.trading_price_data["BTC-USD"] = [
        Candle(
            open_time=1662774258000,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
        ),
        Candle(
            open_time=1662774318000,
            open=6,
            high=7,
            low=8,
            close=9,
            volume=10,
        ),
    ]

    dydx_exchange._wss_handle_candlestick_update(
        {
            "id": "BTC-USD",
            "contents": {
                "trades": [
                    {
                        "createdAt": "2022-09-10T01:49:20.000Z",
                        "price": "30",
                        "size": "10",
                    }
                ]
            },
        }
    )

    assert len(dydx_exchange.trading_price_data["BTC-USD"]) == 2
    assert dydx_exchange.trading_price_data["BTC-USD"][-1] == Candle(
        open_time=1662774318000,
        open=6,
        high=7,
        low=8,
        close=9,
        volume=10,
    )
    assert caplog.records[0].levelname == "ERROR"
    assert (
        "candles cannot be updated because candle(s) are potentially missing"
        in caplog.text
    )


def test_wss_on_message_no_channel(mocker) -> None:
    mocked_oracle_price_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_oracle_price_update"
    )
    mocked_collateral_balance_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_collateral_balance_update"
    )
    mocked_order_update = mocker.patch.object(DyDxExchange, "_wss_handle_order_update")
    mocked_candlestick_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_candlestick_update"
    )

    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_on_message(websocket.WebSocketApp("fake-url"), "{}")

    assert mocked_oracle_price_update.assert_not_called
    assert mocked_collateral_balance_update.assert_not_called
    assert mocked_order_update.assert_not_called
    assert mocked_candlestick_update.assert_not_called


@pytest.mark.parametrize("msg_channel", ["v3_markets", "v3_accounts", "v3_trades"])
def test_wss_on_message(mocker, msg_channel) -> None:
    mocked_oracle_price_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_oracle_price_update"
    )
    mocked_collateral_balance_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_collateral_balance_update"
    )
    mocked_order_update = mocker.patch.object(DyDxExchange, "_wss_handle_order_update")
    mocked_candlestick_update = mocker.patch.object(
        DyDxExchange, "_wss_handle_candlestick_update"
    )

    msg = {"channel": msg_channel}
    msg_str = json.dumps(msg)
    dydx_exchange = DyDxExchange()
    dydx_exchange._wss_on_message(websocket.WebSocketApp("fake-url"), msg_str)

    if msg_channel == "v3_markets":
        assert mocked_oracle_price_update.assert_called_once
    elif msg_channel == "v3_accounts":
        assert mocked_collateral_balance_update.assert_called_once
        assert mocked_order_update.assert_called_once
    elif msg_channel == "v3_trades":
        assert mocked_candlestick_update.assert_called_once


def test_start_wss(mocker) -> None:
    mocked_threading_thread = mocker.patch("afang.exchanges.dydx.threading.Thread")

    dydx_exchange = DyDxExchange()
    dydx_exchange._start_wss()

    assert mocked_threading_thread.assert_called_once


@pytest.mark.parametrize("should_raise_exception", [True, False])
def test_populate_initial_position_sizes(
    caplog, mock_dydx_api_client, should_raise_exception
) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange._api_client = mock_dydx_api_client(should_raise_exception)
    dydx_exchange._populate_initial_position_sizes()

    if should_raise_exception:
        assert caplog.records[0].levelname == "ERROR"
        assert (
            "DydxApiError raised when attempting to populate initial position sizes"
            in caplog.text
        )
        return

    assert dydx_exchange._symbol_total_pos_size["BTC-USD"] == 300


def test_setup_exchange_for_trading(mocker) -> None:
    mocked_populate_trading_symbols = mocker.patch.object(
        DyDxExchange, "_populate_trading_symbols"
    )
    mocked_populate_trading_timeframe = mocker.patch.object(
        DyDxExchange, "_populate_trading_timeframe"
    )
    mocked_populate_initial_pos_sizes = mocker.patch.object(
        DyDxExchange, "_populate_initial_position_sizes"
    )
    mocked_threading_thread = mocker.patch("afang.exchanges.dydx.threading.Thread")
    mocked_populate_initial_trading_price_data = mocker.patch.object(
        DyDxExchange, "_populate_initial_trading_price_data"
    )
    mocked_start_wss = mocker.patch.object(DyDxExchange, "_start_wss")

    dydx_exchange = DyDxExchange()
    dydx_exchange.setup_exchange_for_trading(["BTC-USD"], Timeframe.M1)

    assert mocked_populate_trading_symbols.assert_called_once
    assert mocked_populate_trading_timeframe.assert_called_once
    assert mocked_populate_initial_pos_sizes.assert_called_once
    assert mocked_threading_thread.assert_called_once
    assert mocked_populate_initial_trading_price_data.assert_called_once
    assert mocked_start_wss.assert_called_once


def test_change_initial_leverage(caplog) -> None:
    dydx_exchange = DyDxExchange()
    dydx_exchange.change_initial_leverage(["BTC-USD"], 5)

    assert dydx_exchange.symbol_leverage["BTC-USD"] == 5
    assert caplog.records[0].levelname == "INFO"
    assert "changed BTC-USD initial leverage to 5" in caplog.text
