from enum import Enum

import pytest
import requests

from afang.exchanges import IsExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
)
from afang.models import Timeframe


@pytest.fixture
def dummy_order() -> Order:
    return Order(
        symbol="BTCUSDT",
        order_id="12345",
        side=OrderSide.SELL,
        original_price=100,
        average_price=110,
        original_quantity=10,
        executed_quantity=10,
        remaining_quantity=0,
        order_type=OrderType.LIMIT,
        order_status="FILLED",
        time_in_force="GTC",
        commission=9,
    )


def test_is_exchange_initialization(dummy_is_exchange) -> None:
    assert dummy_is_exchange.name == "test_exchange"
    assert dummy_is_exchange.display_name == "test_exchange"
    assert dummy_is_exchange._base_url == "https://dummy.com"
    assert dummy_is_exchange._wss_url == "wss://dummy.com/ws"
    assert dummy_is_exchange.exchange_symbols == dict()
    assert dummy_is_exchange.get_historical_candles("test_symbol", 0, 100) is None
    assert dummy_is_exchange.get_config_params() == {
        "query_limit": 1,
        "write_limit": 50000,
    }


@pytest.mark.parametrize(
    "status_code, exception, expected_response",
    [
        (200, None, {"result": "success"}),
        (400, requests.ConnectionError, None),
        (400, None, None),
    ],
)
def test_is_exchange_make_request(
    requests_mock, dummy_is_exchange, status_code, exception, expected_response
) -> None:
    if exception:
        requests_mock.get(
            "https://dummy.com/endpoint?query=bull&limit=dog", exc=exception
        )
        requests_mock.post(
            "https://dummy.com/endpoint?query=bull&limit=dog", exc=exception
        )
        requests_mock.delete(
            "https://dummy.com/endpoint?query=bull&limit=dog", exc=exception
        )
    else:
        requests_mock.get(
            "https://dummy.com/endpoint?query=bull&limit=dog",
            json={"result": "success"},
            status_code=status_code,
            exc=exception,
        )
        requests_mock.post(
            "https://dummy.com/endpoint?query=bull&limit=dog",
            json={"result": "success"},
            status_code=status_code,
            exc=exception,
        )
        requests_mock.delete(
            "https://dummy.com/endpoint?query=bull&limit=dog",
            json={"result": "success"},
            status_code=status_code,
            exc=exception,
        )

    # GET request
    response = dummy_is_exchange._make_request(
        HTTPMethod.GET, "/endpoint", query_parameters={"query": "bull", "limit": "dog"}
    )
    assert response == expected_response

    # POST request
    response = dummy_is_exchange._make_request(
        HTTPMethod.POST, "/endpoint", query_parameters={"query": "bull", "limit": "dog"}
    )
    assert response == expected_response

    # DELETE request
    response = dummy_is_exchange._make_request(
        HTTPMethod.DELETE,
        "/endpoint",
        query_parameters={"query": "bull", "limit": "dog"},
    )
    assert response == expected_response


def test_is_exchange_make_request_unknown_method(
    caplog, requests_mock, dummy_is_exchange
) -> None:
    requests_mock.get(
        "https://dummy.com/endpoint?query=bull&limit=dog",
        json={"result": "success"},
        status_code=200,
    )

    # noinspection PyShadowingNames
    class HTTPMethod(Enum):
        UNKNOWN = "UNKNOWN"

    response = dummy_is_exchange._make_request(
        HTTPMethod.UNKNOWN,
        "/endpoint",
        query_parameters={"query": "bull", "limit": "dog"},
    )

    assert response is None
    assert caplog.records[0].levelname == "ERROR"
    assert (
        "Unknown HTTP method UNKNOWN provided while making request to /endpoint"
        in caplog.text
    )


def test_populate_trading_symbols_unknown_symbol(dummy_is_exchange, caplog) -> None:
    with pytest.raises(KeyError):
        dummy_is_exchange._populate_trading_symbols(["BTCUSDT"])
        assert caplog.records[-1].levelname == "ERROR"
        assert "symbol BTCUSDT does not exist" in caplog.text


def test_populate_trading_symbols(dummy_is_exchange) -> None:
    dummy_is_exchange.exchange_symbols = {
        "BTCUSDT": Symbol(
            name="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_decimals=2,
            quantity_decimals=2,
            step_size=2,
            tick_size=2,
        )
    }
    dummy_is_exchange._populate_trading_symbols(["BTCUSDT"])

    assert (
        dummy_is_exchange.trading_symbols["BTCUSDT"]
        == dummy_is_exchange.exchange_symbols["BTCUSDT"]
    )


def test_populate_trading_timeframe_unsupported_tf(caplog, dummy_is_exchange) -> None:
    with pytest.raises(ValueError):
        dummy_is_exchange._populate_trading_timeframe(Timeframe.M1, ["M15", "M30"])
        assert caplog.records[-1].levelname == "ERROR"
        assert "timeframe 1m not supported. Try a different timeframe" in caplog.text


def test_populate_trading_timeframe(dummy_is_exchange) -> None:
    dummy_is_exchange._populate_trading_timeframe(Timeframe.M1, ["M15", "M1"])
    assert dummy_is_exchange.trading_timeframe == Timeframe.M1


def test_populate_initial_trading_price_data(mocker, caplog, dummy_is_exchange) -> None:
    mocker.patch.object(
        IsExchange,
        "get_historical_candles",
        return_value=[
            Candle(
                open_time=1,
                open=2,
                high=3,
                low=4,
                close=5,
                volume=6,
            )
        ],
    )

    dummy_is_exchange.trading_symbols = {
        "BTCUSDT": Symbol(
            name="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_decimals=2,
            quantity_decimals=2,
            step_size=2,
            tick_size=2,
        )
    }
    dummy_is_exchange._populate_initial_trading_price_data(num_iterations=1)

    assert len(dummy_is_exchange.trading_price_data["BTCUSDT"]) == 1
    assert caplog.records[-1].levelname == "INFO"
    assert "fetched 1 initial price data candles" in caplog.text


@pytest.mark.parametrize(
    "order_id, active_orders, expected_order, should_log_warning",
    [
        (None, {"12345": dummy_order}, None, False),
        ("12345", {"12345": dummy_order}, dummy_order, False),
        ("12345", dict(), dummy_order, False),
        ("12345", dict(), None, True),
    ],
)
def test_get_exchange_order(
    mocker,
    dummy_is_exchange,
    caplog,
    order_id,
    active_orders,
    expected_order,
    should_log_warning,
) -> None:
    mocker.patch.object(
        IsExchange,
        "get_order_by_id",
        return_value=expected_order,
    )

    dummy_is_exchange._active_orders = active_orders
    order = dummy_is_exchange.get_exchange_order("BTCUSDT", order_id)

    assert order == expected_order
    if should_log_warning:
        assert caplog.records[0].levelname == "WARN"
        assert "Unable to get test_exchange BTCUSDT order with id: 12345" in caplog.text
    if expected_order:
        assert dummy_is_exchange._active_orders["12345"]
    else:
        assert "12345" not in dummy_is_exchange._active_orders or not order_id
