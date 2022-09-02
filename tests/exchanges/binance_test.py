from typing import Any, Dict

import pytest

from afang.exchanges.binance import BinanceExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
)
from afang.models import Mode, Timeframe


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
    assert binance_exchange.mode is None
    assert binance_exchange.display_name == "binance"
    assert binance_exchange.testnet is False
    assert binance_exchange._base_url == "https://fapi.binance.com"
    assert binance_exchange._wss_url == "wss://fstream.binance.com/ws"
    assert binance_exchange.exchange_symbols == ["BTCUSDT", "ETHUSDT"]
    assert binance_exchange.get_config_params() == {
        "query_limit": 1.1,
        "write_limit": 10000,
    }

    binance_exchange_testnet = BinanceExchange(testnet=True)
    assert binance_exchange_testnet.mode is None
    assert binance_exchange_testnet.display_name == "binance-testnet"
    assert binance_exchange_testnet.testnet is True
    assert binance_exchange_testnet._base_url == "https://testnet.binancefuture.com"
    assert binance_exchange_testnet._wss_url == "wss://stream.binancefuture.com/ws"

    binance_exchange_data_mode = BinanceExchange(mode=Mode.data)
    assert binance_exchange_data_mode.mode == Mode.data


@pytest.mark.parametrize(
    "req_response, expected_symbols",
    [
        (
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "contractType": "PERPETUAL",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                            {"filterType": "LOT_SIZE", "stepSize": "1"},
                        ],
                    },
                    {
                        "symbol": "LINKBTC",
                        "contractType": "SPOT",
                    },
                    {
                        "symbol": "LTCUSDT",
                        "contractType": "PERPETUAL",
                        "baseAsset": "LTC",
                        "quoteAsset": "USDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                            {"filterType": "LOT_SIZE", "stepSize": "1"},
                        ],
                    },
                ]
            },
            {
                "BTCUSDT": Symbol(
                    name="BTCUSDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    price_decimals=4,
                    quantity_decimals=0,
                    tick_size=0.0001,
                    step_size=1,
                ),
                "LTCUSDT": Symbol(
                    name="LTCUSDT",
                    base_asset="LTC",
                    quote_asset="USDT",
                    price_decimals=4,
                    quantity_decimals=0,
                    tick_size=0.0001,
                    step_size=1,
                ),
            },
        ),
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
        "afang.exchanges.binance.BinanceExchange._make_request",
        mock_make_request,
    )

    binance_exchange = BinanceExchange()
    assert binance_exchange.exchange_symbols == expected_symbols


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
                    "600.4",
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
                    "400.4",
                    50,
                    "1.8",
                    "2.8",
                    "1.7",
                ],
            ],
            [
                Candle(1, 1.5, 2.5, 3.5, 4.5, 600.4),
                Candle(9, 4.5, 4.5, 2.5, 6.5, 400.4),
            ],
        ),
        (None, None),
    ],
)
def test_get_historical_candles(mocker, req_response, expected_candles) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    # mock the return value of the _make_request function.
    def mock_make_request(
        _self, _method: HTTPMethod, _endpoint: str, _query_parameters: Dict
    ) -> Any:
        return req_response

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request",
        mock_make_request,
    )

    binance_exchange = BinanceExchange()
    assert (
        binance_exchange.get_historical_candles("test_symbol", 2, 100)
        == expected_candles
    )


def test_get_historical_candles_unknown_timeframe(mocker, caplog) -> None:
    # mock the return value of the _get_symbols function.
    def mock_get_symbols(_self):
        return ["test_symbol"]

    mocker.patch(
        "afang.exchanges.binance.BinanceExchange._get_symbols",
        mock_get_symbols,
    )

    binance_exchange = BinanceExchange()
    binance_exchange.get_historical_candles("test_symbol", 2, 100, Timeframe.M3)

    assert caplog.records[0].levelname == "ERROR"
    assert (
        "binance cannot fetch historical candles in 3m intervals. Please use another timeframe"
        in caplog.text
    )


def test_generate_authed_request_signature(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    dummy_req_params = dict()
    dummy_req_params["symbol"] = "BTCUSDT"
    dummy_req_params["orderId"] = "12345"
    dummy_req_params["timestamp"] = "1661431820499"

    binance_exchange = BinanceExchange()
    binance_exchange._SECRET_KEY = "stBntGeUiVVSOpNPQAQF5qF8Xu6CG-i-d4Stzeu7"

    signature = binance_exchange._generate_authed_request_signature(dummy_req_params)
    assert (
        signature == "a461aa77cfb2c6c81ee9152705fc5cd72cb79b8a39a28d87102bd8cc4d47157c"
    )


@pytest.mark.parametrize("response", [({"orderId": "12345"}), None])
def test_place_order(mocker, response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    order_id = binance_exchange.place_order(
        "BTCUSDT", OrderSide.BUY, 0.001, OrderType.LIMIT, 10
    )

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once
    assert order_id is None if not response else response["orderId"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        {
            "origQty": "10",
            "executedQty": "5",
            "side": "BUY",
            "type": "MARKET",
            "price": "15",
            "avgPrice": "12",
            "status": "PARTIALLY_FILLED",
            "timeInForce": "GTC",
        },
        {
            "origQty": "10",
            "executedQty": "5",
            "side": "SELL",
            "type": "LIMIT",
            "price": "15",
            "avgPrice": "12",
            "status": "PARTIALLY_FILLED",
            "timeInForce": "GTC",
        },
    ],
)
def test_get_order_by_id(mocker, response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    order = binance_exchange.get_order_by_id("BTCUSDT", "12345")

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once

    if not response:
        assert order is None
        return

    order_quantity = float(response["origQty"])
    executed_quantity = float(response["executedQty"])
    remaining_quantity = order_quantity - executed_quantity

    order_side = OrderSide.UNKNOWN
    if response["side"] == "BUY":
        order_side = OrderSide.BUY
    elif response["side"] == "SELL":
        order_side = OrderSide.SELL

    order_type = OrderType.UNKNOWN
    if response["type"] == "LIMIT":
        order_type = OrderType.LIMIT
    elif response["type"] == "MARKET":
        order_type = OrderType.MARKET

    assert order == Order(
        symbol="BTCUSDT",
        order_id="12345",
        side=order_side,
        original_price=float(response["price"]),
        average_price=float(response["avgPrice"]),
        original_quantity=order_quantity,
        executed_quantity=executed_quantity,
        remaining_quantity=remaining_quantity,
        order_type=order_type,
        order_status=response["status"],
        time_in_force=response["timeInForce"],
        commission=None,
    )


@pytest.mark.parametrize("response", [None, {"orderId": "12345"}])
def test_cancel_order(mocker, response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    cancellation_successful = binance_exchange.cancel_order("BTCUSDT", "12345")

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once
    assert cancellation_successful is False if not response else True
