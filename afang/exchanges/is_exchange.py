import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger()


class IsExchange(ABC):
    """Base interface for any supported exchange."""

    @abstractmethod
    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self._base_url = base_url
        self.symbols = self._get_symbols()

    @abstractmethod
    def _get_symbols(self) -> List[str]:
        """Fetch all the available symbols on the exchange.

        :return: List[str]
        """

        return list()

    def _make_request(self, endpoint: str, query_parameters: Dict) -> Any:
        """Make an unauthenticated GET request to the exchange. If the request
        is successful, a JSON object instance will be returned. If the request
        in unsuccessful, None will be returned.

        :param endpoint: the URL path of the associated GET request.
        :param query_parameters: a dictionary of parameters to pass within the query.

        :return: Any
        """

        try:
            response = requests.get(self._base_url + endpoint, params=query_parameters)
        except Exception as e:
            logger.error("Connection error while making request to %s: %s", endpoint, e)
            return None

        if response.status_code == 200:
            return response.json()

        logger.error(
            "Error while making request to %s: %s (status code: %s)",
            endpoint,
            response.json(),
            response.status_code,
        )
        return None

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[List[Tuple[float, float, float, float, float, float]]]:
        """Fetch candlestick bars for a particular symbol from the exchange. If
        start_time and end_time are not provided, the most recent klines are
        returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.

        :return: Optional[List[Tuple[float, float, float, float, float, float]]]
        """

        return None
