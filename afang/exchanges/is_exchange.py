import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

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

logger = logging.getLogger(__name__)


class IsExchange(ABC):
    """Base interface for any supported exchange."""

    @abstractmethod
    def __init__(
        self,
        name: str,
        mode: Optional[Mode],
        testnet: bool,
        base_url: str,
        wss_url: str,
    ) -> None:
        self.name = name
        self.display_name = name + "-testnet" if testnet else name
        self.mode = mode
        self.testnet = testnet
        self._base_url = base_url
        self._wss_url = wss_url
        self.exchange_symbols: Dict[str, Symbol] = self._get_symbols()
        self.symbol_leverage: Dict[str, int] = dict()
        self.trading_symbols: Dict[str, Symbol] = dict()
        self.trading_timeframe: Optional[Timeframe] = None
        self.trading_price_data: Dict[str, pd.DataFrame] = dict()
        self.active_orders: Dict[str, Order] = dict()
        self.trading_symbol_balance: Dict[str, SymbolBalance] = dict()

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

            - query_limit: rate limit of how long to sleep between HTTP requests.
            - write_limit: threshold of how many candles to fetch before saving them to the DB.

        :return: dict
        """

        return {"query_limit": 1, "write_limit": 50000}

    @abstractmethod
    def _get_symbols(self) -> Dict[str, Symbol]:
        """Fetch all the available symbols on the exchange.

        :return: Dict[str, Symbol]
        """

        return dict()

    def _make_request(
        self,
        method: HTTPMethod,
        endpoint: str,
        query_parameters: Dict,
        headers: Optional[Dict] = None,
    ) -> Any:
        """Make an unauthenticated GET request to the exchange. If the request
        is successful, a JSON object instance will be returned. If the request
        in unsuccessful, None will be returned.

        :param method: HTTP method to be used to make the request.
        :param endpoint: the URL path of the associated GET request.
        :param query_parameters: a dictionary of parameters to pass within the query.
        :param headers: optional headers to send with the request.

        :return: Any
        """

        try:
            if method == HTTPMethod.GET:
                response = requests.get(
                    self._base_url + endpoint, params=query_parameters, headers=headers
                )
            elif method == HTTPMethod.POST:
                response = requests.post(
                    self._base_url + endpoint, params=query_parameters, headers=headers
                )
            elif method == HTTPMethod.DELETE:
                response = requests.delete(
                    self._base_url + endpoint, params=query_parameters, headers=headers
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
            "Error while making %s request to %s: %s (status code: %s)",
            method.name,
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
        timeframe: Timeframe = Timeframe.M1,
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

    @abstractmethod
    def place_order(
        self,
        symbol_name: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float] = None,
        **_kwargs
    ) -> Optional[str]:
        """Place a new order for a specified symbol on the exchange. Returns
        the order ID if order placement was successful.

        :param symbol_name: name of symbol.
        :param side: order side.
        :param quantity: order quantity.
        :param order_type: order type.
        :param price: optional. order price.
        :param _kwargs:
            post_only bool: order will only be allowed if it will enter the order book.
                            NOTE: post_only orders may override the time in force if specified.
            dydx_limit_fee: float: highest accepted fee for the trade on the dYdX exchange.
        :return: Optional[str]
        """

        return None

    @abstractmethod
    def get_order_by_id(self, symbol_name: str, order_id: str) -> Optional[Order]:
        """Query an order by ID.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to query.
        :return: Optional[Order]
        """

        return None

    @abstractmethod
    def cancel_order(self, symbol_name: str, order_id: str) -> bool:
        """Cancel an active order on the exchange. Returns a bool on whether
        order cancellation was successful.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to cancel.
        :return: bool
        """

        return False
