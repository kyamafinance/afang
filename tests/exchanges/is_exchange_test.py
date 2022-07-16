import pytest
import requests


def test_is_exchange_initialization(dummy_is_exchange) -> None:
    assert dummy_is_exchange.name == "test_exchange"
    assert dummy_is_exchange._base_url == "https://dummy.com"
    assert dummy_is_exchange.symbols == list()
    assert dummy_is_exchange.get_historical_data("test_symbol", 0, 100) is None
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
    else:
        requests_mock.get(
            "https://dummy.com/endpoint?query=bull&limit=dog",
            json={"result": "success"},
            status_code=status_code,
            exc=exception,
        )
    response = dummy_is_exchange._make_request(
        "/endpoint", query_parameters={"query": "bull", "limit": "dog"}
    )

    assert response == expected_response
