import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from afang.exchanges.models import Candle, HTTPMethod
from afang.models import Timeframe

logger = logging.getLogger(__name__)


class IsExchange(ABC):
    """Base interface for any supported exchange."""

    @abstractmethod
    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self._base_url = base_url
        self.symbols = self._get_symbols()

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

            - query_limit: rate limit of how long to sleep between HTTP requests.
            - write_limit: threshold of how many candles to fetch before saving them to the DB.

        :return: dict
        """

        return {"query_limit": 1, "write_limit": 50000}

    @abstractmethod
    def _get_symbols(self) -> List[str]:
        """Fetch all the available symbols on the exchange.

        :return: List[str]
        """

        return list()

    def _make_request(
        self, method: HTTPMethod, endpoint: str, query_parameters: Dict
    ) -> Any:
        """Make an unauthenticated GET request to the exchange. If the request
        is successful, a JSON object instance will be returned. If the request
        in unsuccessful, None will be returned.

        :param method: HTTP method to be used to make the request.
        :param endpoint: the URL path of the associated GET request.
        :param query_parameters: a dictionary of parameters to pass within the query.

        :return: Any
        """

        try:
            if method == HTTPMethod.GET:
                response = requests.get(
                    self._base_url + endpoint, params=query_parameters
                )
            elif method == HTTPMethod.POST:
                response = requests.post(
                    self._base_url + endpoint, params=query_parameters
                )
            elif method == HTTPMethod.DELETE:
                response = requests.delete(
                    self._base_url + endpoint, params=query_parameters
                )
            else:
                logger.error(
                    "Unknown HTTP method %s provided while making request to %s",
                    method.name,
                    endpoint,
                )
                return None
        except Exception as e:
            logger.error(
                "Connection error while making %s request to %s: %s",
                method.name,
                endpoint,
                e,
            )
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
    def get_historical_candles(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeframe: Optional[Timeframe] = Timeframe.M1,
    ) -> Optional[List[Candle]]:
        """Fetch candlestick bars for a particular symbol from the exchange. If
        start_time and end_time are not provided, the most recent klines are
        returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param timeframe: optional. timeframe to download historical candles.

        :return: Optional[List[Candle]]
        """

        return None
