import json
from typing import Any, Dict

import pytest
import websocket

from afang.exchanges.binance import BinanceExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
    SymbolBalance,
)
from afang.models import Timeframe


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
    assert binance_exchange_testnet.display_name == "binance-testnet"
    assert binance_exchange_testnet.testnet is True
    assert binance_exchange_testnet._base_url == "https://testnet.binancefuture.com"
    assert binance_exchange_testnet._wss_url == "wss://stream.binancefuture.com/ws"


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
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

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
    "order_type, response",
    [
        (
            OrderType.MARKET,
            {
                "symbol": "BTCUSDT",
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0004",
            },
        ),
        (
            OrderType.LIMIT,
            {
                "symbol": "BTCUSDT",
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0004",
            },
        ),
        (OrderType.LIMIT, None),
    ],
)
def test_get_user_commission_rate(mocker, order_type, response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    user_commission_rate = binance_exchange.get_user_commission_rate(
        "BTCUSDT", order_type
    )

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once

    if not response:
        assert user_commission_rate is None
    elif order_type == OrderType.MARKET:
        assert user_commission_rate == float(response["takerCommissionRate"])
    elif order_type == OrderType.LIMIT:
        assert user_commission_rate == float(response["makerCommissionRate"])


@pytest.mark.parametrize(
    "response, expected_commission",
    [
        (None, None),
        (
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
            0.024,
        ),
        (
            {
                "origQty": "10",
                "executedQty": "10",
                "side": "SELL",
                "type": "LIMIT",
                "price": "15",
                "avgPrice": "12",
                "status": "FILLED",
                "timeInForce": "GTC",
            },
            0.048,
        ),
    ],
)
def test_get_order_by_id(mocker, response, expected_commission) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocker.patch(
        "afang.exchanges.binance.BinanceExchange.get_user_commission_rate",
        return_value=0.0004,
    )
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
        commission=expected_commission,
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


@pytest.mark.parametrize(
    "response", [None, {"assets": [{"asset": "BTC", "walletBalance": "11"}]}]
)
def test_get_asset_balances(mocker, caplog, response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    binance_exchange._get_asset_balances()

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once

    if not response:
        assert caplog.records[0].levelname == "ERROR"
        assert "error while fetching asset balances" in caplog.text
        return

    assert binance_exchange.trading_symbol_balance["BTC"] == SymbolBalance(
        name="BTC", wallet_balance=11.0
    )


def test_subscribe_wss_candlestick_stream(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    class MockWSS:
        def send(self, _val):
            pass

    mock_wss = MockWSS()

    binance_exchange = BinanceExchange()
    binance_exchange._wss = mock_wss
    binance_exchange.trading_timeframe = Timeframe.M15
    binance_exchange.trading_symbols = {
        "BTC": Symbol(
            name="BTC",
            base_asset="BTC",
            quote_asset="BTC",
            price_decimals=2,
            quantity_decimals=2,
            tick_size=2,
            step_size=2,
        ),
        "ETH": Symbol(
            name="ETH",
            base_asset="ETH",
            quote_asset="ETH",
            price_decimals=2,
            quantity_decimals=2,
            tick_size=2,
            step_size=2,
        ),
    }
    binance_exchange._subscribe_wss_candlestick_stream()

    assert binance_exchange._wss_stream_id == 3


def test_wss_on_open(mocker, caplog) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_subscribe_wss_candlestick_stream = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._subscribe_wss_candlestick_stream"
    )

    binance_exchange = BinanceExchange()
    binance_exchange._wss_on_open(websocket.WebSocketApp("fake-url"))

    assert caplog.records[0].levelname == "INFO"
    assert "wss connection opened" in caplog.text
    assert mocked_subscribe_wss_candlestick_stream.assert_called_once


def test_wss_on_close(caplog, mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    binance_exchange = BinanceExchange()
    binance_exchange._wss_on_close(websocket.WebSocketApp("fake-url"), "500", "error")

    assert caplog.records[0].levelname == "WARN"
    assert "wss connection closed" in caplog.text


def test_wss_on_error(mocker, caplog) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._start_wss")
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    binance_exchange = BinanceExchange()
    binance_exchange._wss_on_error(websocket.WebSocketApp("fake-url"), "this message")

    assert caplog.records[0].levelname == "ERROR"
    assert "wss connection error: this message" in caplog.text


def test_wss_handle_listen_key_expired(mocker, caplog) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    binance_exchange = BinanceExchange()
    binance_exchange._wss_handle_listen_key_expired("this message")

    assert caplog.records[0].levelname == "ERROR"
    assert "wss listen key expired" in caplog.text


def test_wss_handle_margin_call(caplog, mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    binance_exchange = BinanceExchange()
    binance_exchange._wss_handle_margin_call({"p": [{"s": "BTC"}, {"s": "ETH"}]})

    assert caplog.records[0].levelname == "WARN"
    assert "position risk ratio is too high for symbols: BTC, ETH" in caplog.text


@pytest.mark.parametrize(
    "msg_data, expected_symbol_balances",
    [
        ({}, {}),
        ({"a": {}}, {}),
        (
            {"a": {"B": [{"a": "BTC", "wb": "32"}]}},
            {"BTC": SymbolBalance(name="BTC", wallet_balance=32)},
        ),
    ],
)
def test_wss_handle_asset_balance_update(
    mocker, msg_data, expected_symbol_balances
) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    binance_exchange = BinanceExchange()
    binance_exchange._wss_handle_asset_balance_update(msg_data)

    assert binance_exchange.trading_symbol_balance == expected_symbol_balances


def test_wss_handle_order_update(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    msg_data = {
        "o": {
            "s": "BTC",
            "i": 12345,
            "p": "13.4",
            "ap": "12.77",
            "q": "12.5",
            "z": "2.5",
            "S": "SELL",
            "o": "LIMIT",
            "n": 2,
            "X": "FILLED",
            "f": "GTC",
        }
    }

    binance_exchange = BinanceExchange()
    binance_exchange._active_orders["12345"] = Order(
        symbol="BTC",
        order_id="12345",
        side=OrderSide.SELL,
        original_price=43,
        average_price=42,
        original_quantity=12,
        executed_quantity=2,
        remaining_quantity=10,
        order_type=OrderType.LIMIT,
        order_status="OPEN",
        time_in_force="GTC",
        commission=2,
    )
    binance_exchange._wss_handle_order_update(msg_data)

    assert binance_exchange._active_orders["12345"] == Order(
        symbol="BTC",
        order_id="12345",
        side=OrderSide.SELL,
        original_price=13.4,
        average_price=12.77,
        original_quantity=12.5,
        executed_quantity=2.5,
        remaining_quantity=10,
        order_type=OrderType.LIMIT,
        order_status="FILLED",
        time_in_force="GTC",
        commission=4,
    )


def test_wss_handle_candlestick_update_no_k(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    binance_exchange = BinanceExchange()
    binance_exchange.trading_price_data["BTC"] = [
        Candle(
            open_time=1,
            open=2,
            high=3,
            low=4,
            close=5,
            volume=6,
        )
    ]

    binance_exchange._wss_handle_candlestick_update({})
    assert len(binance_exchange.trading_price_data["BTC"]) == 1


@pytest.mark.parametrize("msg_data_open_time, expected_candle_len", [(1, 1), (2, 1)])
def test_wss_handle_candlestick_update(
    mocker, msg_data_open_time, expected_candle_len
) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")

    msg_data: Dict = {
        "ps": "BTC",
        "k": {
            "t": msg_data_open_time,
            "o": "3",
            "h": "4",
            "l": "5",
            "c": "6",
            "q": "7",
        },
    }

    binance_exchange = BinanceExchange()
    binance_exchange.trading_price_data["BTC"] = [
        Candle(
            open_time=1,
            open=2,
            high=3,
            low=4,
            close=5,
            volume=6,
        )
    ]

    binance_exchange._wss_handle_candlestick_update(msg_data)
    assert len(binance_exchange.trading_price_data["BTC"]) == expected_candle_len
    assert binance_exchange.trading_price_data["BTC"][-1] == Candle(
        open_time=msg_data_open_time,
        open=3,
        high=4,
        low=5,
        close=6,
        volume=float(msg_data["k"]["q"]),
    )


@pytest.mark.parametrize(
    "msg_data",
    [
        {},
        {"e": "listenKeyExpired"},
        {"e": "MARGIN_CALL"},
        {"e": "ACCOUNT_UPDATE"},
        {"e": "ORDER_TRADE_UPDATE"},
        {"e": "continuous_kline"},
    ],
)
def test_wss_on_message(mocker, msg_data) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_handle_lk_expired = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._wss_handle_listen_key_expired"
    )
    mocked_handle_margin_call = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._wss_handle_margin_call"
    )
    mocked_handle_asset_balance_update = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._wss_handle_asset_balance_update"
    )
    mocked_handle_order_update = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._wss_handle_order_update"
    )
    mocked_handle_candlestick_update = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._wss_handle_candlestick_update"
    )

    binance_exchange = BinanceExchange()
    binance_exchange._wss_on_message(
        websocket.WebSocketApp("fake-url"), json.dumps(msg_data)
    )

    if "e" not in msg_data:
        return
    event_type = msg_data["e"]

    if event_type == "listenKeyExpired":
        assert mocked_handle_lk_expired.assert_called_once
    if event_type == "MARGIN_CALL":
        assert mocked_handle_margin_call.assert_called_once
    if event_type == "ACCOUNT_UPDATE":
        assert mocked_handle_asset_balance_update.assert_called_once
    if event_type == "ORDER_TRADE_UPDATE":
        assert mocked_handle_order_update.assert_called_once
    if event_type == "continuous_kline":
        assert mocked_handle_candlestick_update.assert_called_once


@pytest.mark.parametrize(
    "response, expected_output", [(None, None), ({"listenKey": 12345}, 12345)]
)
def test_fetch_wss_listen_key(mocker, response, expected_output) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request", return_value=response
    )

    binance_exchange = BinanceExchange()
    listen_key = binance_exchange._fetch_wss_listen_key()

    assert mocked_make_request.assert_called_once
    assert listen_key == expected_output


@pytest.mark.parametrize(
    "listen_key_result, expected_listen_key", [(None, None), (12345, 12345)]
)
def test_keep_wss_alive(mocker, listen_key_result, expected_listen_key) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_fetch_listen_key = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._fetch_wss_listen_key",
        return_value=listen_key_result,
    )

    binance_exchange = BinanceExchange()
    binance_exchange._keep_wss_alive(run_forever=False)

    assert mocked_fetch_listen_key.assert_called_once
    assert binance_exchange._wss_listen_key == expected_listen_key


def test_start_wss_no_listen_key(mocker, caplog) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_fetch_lk = mocker.patch.object(
        BinanceExchange, "_fetch_wss_listen_key", return_value=None
    )

    binance_exchange = BinanceExchange()
    binance_exchange._start_wss()

    assert mocked_fetch_lk.assert_called_once
    assert binance_exchange._wss is None
    assert caplog.records[0].levelname == "ERROR"
    assert "could not start wss due to lack of a valid listen key" in caplog.text


def test_start_wss(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_threading_thread = mocker.patch("afang.exchanges.binance.threading.Thread")
    mocked_fetch_lk = mocker.patch.object(
        BinanceExchange, "_fetch_wss_listen_key", return_value=12345
    )

    binance_exchange = BinanceExchange()
    binance_exchange._start_wss()

    assert mocked_fetch_lk.assert_called_once
    assert mocked_threading_thread.call_count == 2
    assert type(binance_exchange._wss) == websocket.WebSocketApp


def test_setup_exchange_for_trading(mocker) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_populate_trading_symbols = mocker.patch.object(
        BinanceExchange, "_populate_trading_symbols"
    )
    mocked_populate_trading_timeframe = mocker.patch.object(
        BinanceExchange, "_populate_trading_timeframe"
    )
    mocked_get_asset_balances = mocker.patch.object(
        BinanceExchange, "_get_asset_balances"
    )
    mocked_populate_initial_trading_data = mocker.patch.object(
        BinanceExchange, "_populate_initial_trading_price_data"
    )
    mocked_start_wss = mocker.patch.object(BinanceExchange, "_start_wss")

    binance_exchange = BinanceExchange()
    binance_exchange.setup_exchange_for_trading(["BTC"], Timeframe.M15)

    assert mocked_populate_trading_symbols.assert_called_once
    assert mocked_populate_trading_timeframe.assert_called_once
    assert mocked_get_asset_balances.assert_called_once
    assert mocked_populate_initial_trading_data.assert_called_once
    assert mocked_start_wss.assert_called_once


@pytest.mark.parametrize("request_response", [None, {"leverage": 3}])
def test_change_initial_leverage(mocker, caplog, request_response) -> None:
    mocker.patch("afang.exchanges.binance.BinanceExchange._get_symbols")
    mocked_make_request = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._make_request",
        return_value=request_response,
    )
    mocked_generate_authed_request_signature = mocker.patch(
        "afang.exchanges.binance.BinanceExchange._generate_authed_request_signature"
    )

    binance_exchange = BinanceExchange()
    binance_exchange.change_initial_leverage(["BTC"], 3)

    assert mocked_make_request.assert_called_once
    assert mocked_generate_authed_request_signature.assert_called_once

    if not request_response:
        assert caplog.records[0].levelname == "WARN"
        assert "could not change BTC initial leverage to 3" in caplog.text
        return

    assert binance_exchange.symbol_leverage["BTC"] == 3
    assert caplog.records[0].levelname == "INFO"
    assert "changed BTC initial leverage to 3" in caplog.text
