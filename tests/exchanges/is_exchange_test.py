from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests

from afang.exchanges.is_exchange import IsExchange


@pytest.fixture
def dummy_is_exchange() -> IsExchange:
    class Dummy(IsExchange):
        def __init__(self, name: str, base_url: str) -> None:
            super().__init__(name, base_url)

        def _get_symbols(self) -> List[str]:
            return super()._get_symbols()

        def _make_request(self, endpoint: str, query_parameters: Dict) -> Any:
            return super()._make_request(endpoint, query_parameters)

        def get_historical_data(
            self,
            _symbol: str,
            _start_time: Optional[int] = None,
            _end_time: Optional[int] = None,
        ) -> Optional[List[Tuple[float, float, float, float, float, float]]]:
            return super().get_historical_data(_symbol, _start_time, _end_time)

    return Dummy(name="test_exchange", base_url="https://dummy.com")


def test_is_exchange_initialization(dummy_is_exchange) -> None:
    assert dummy_is_exchange.name == "test_exchange"
    assert dummy_is_exchange._base_url == "https://dummy.com"
    assert dummy_is_exchange.symbols == list()
    assert dummy_is_exchange.get_historical_data("test_symbol", 0, 100) is None


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
