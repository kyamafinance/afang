from enum import Enum

import pytest
import requests

from afang.exchanges.models import HTTPMethod


def test_is_exchange_initialization(dummy_is_exchange) -> None:
    assert dummy_is_exchange.name == "test_exchange"
    assert dummy_is_exchange.display_name == "test_exchange"
    assert dummy_is_exchange._base_url == "https://dummy.com"
    assert dummy_is_exchange._wss_url == "wss://dummy.com/ws"
    assert dummy_is_exchange.symbols == dict()
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
