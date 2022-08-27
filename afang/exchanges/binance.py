import hashlib
import hmac
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv

from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
)
from afang.models import Mode, Timeframe
from afang.utils.util import get_float_precision

load_dotenv()
logger = logging.getLogger(__name__)


class TimeframeMapping(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"


class BinanceExchange(IsExchange):
    """Interface to run exchange functions on Binance USDT Futures."""

    def __init__(self, mode: Optional[Mode] = None, testnet: bool = False) -> None:
        """
        :param testnet: whether to use the testnet version of the exchange.
        """

        name = "binance"
        base_url = "https://fapi.binance.com"
        wss_url = "wss://fstream.binance.com/ws"
        if testnet:
            base_url = "https://testnet.binancefuture.com"
            wss_url = "wss://stream.binancefuture.com/ws"

        super().__init__(name, mode, testnet, base_url, wss_url)

        # Binance exchange environment variables.
        self._API_KEY = os.environ.get("BINANCE_API_KEY")
        self._SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY")

        # Headers to be applied to authenticated requests.
        self._headers = {"X-MBX-APIKEY": self._API_KEY}

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

        :return: dict
        """

        return {"query_limit": 1.1, "write_limit": 10000}

    def _get_symbols(self) -> Dict[str, Symbol]:
        """Fetch all Binance USDT Futures symbols.

        :return: Dict[str, Symbol]
        """

        symbols: Dict[str, Symbol] = dict()
        params: Dict = dict()
        endpoint = "/fapi/v1/exchangeInfo"

        data = self._make_request(HTTPMethod.GET, endpoint, params)
        if data is None:
            return symbols

        for symbol in data.get("symbols"):
            if symbol.get("contractType") == "PERPETUAL":
                symbol_name = symbol.get("symbol")
                tick_size: float = 0
                step_size: float = 0

                for symbol_filter in symbol.get("filters"):
                    if symbol_filter.get("filterType") == "PRICE_FILTER":
                        tick_size = symbol_filter.get("tickSize")
                    if symbol_filter.get("filterType") == "LOT_SIZE":
                        step_size = symbol_filter.get("stepSize")

                symbols[symbol_name] = Symbol(
                    name=symbol_name,
                    base_asset=symbol.get("baseAsset"),
                    quote_asset=symbol.get("quoteAsset"),
                    price_decimals=get_float_precision(tick_size),
                    quantity_decimals=get_float_precision(step_size),
                    tick_size=float(tick_size),
                    step_size=float(step_size),
                )

        return symbols

    def get_historical_candles(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeframe: Timeframe = Timeframe.M1,
    ) -> Optional[List[Candle]]:
        """Fetch candlestick bars for a particular symbol from the Binance
        exchange. If start_time and end_time are not sent, the most recent
        klines are returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param timeframe: optional. timeframe to download historical candles.

        :return: Optional[List[Candle]]
        """

        try:
            tf_interval: str = TimeframeMapping[timeframe.name].value
        except KeyError:
            logger.error(
                "%s cannot fetch historical candles in %s intervals. Please use another timeframe",
                self.name,
                timeframe.value,
            )
            return None

        params: Dict = dict()
        params["symbol"] = symbol
        params["interval"] = tf_interval
        params["limit"] = "1500"

        if start_time:
            start_time += (
                60000  # adding a minute due to how binance API returns results.
            )
            params["startTime"] = str(start_time)
        if end_time:
            end_time -= (
                60000  # subtracting a minute due to how binance API returns results.
            )
            params["endTime"] = str(end_time)

        endpoint = "/fapi/v1/klines"

        raw_candles = self._make_request(HTTPMethod.GET, endpoint, params)
        if raw_candles is None:
            return None

        candles: List[Candle] = []
        for candle in raw_candles:
            candles.append(
                Candle(
                    open_time=int(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[7]),
                )
            )

        return candles

    def _generate_authed_request_signature(self, req_params: Dict) -> str:
        """Generate a signature to be used to authenticate private HTTP
        endpoints.

        :param req_params: request query string parameters.
        :return: str
        """

        return hmac.new(
            self._SECRET_KEY.encode(), urlencode(req_params).encode(), hashlib.sha256
        ).hexdigest()

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
                            NOTE: post_only orders will override the time in force if specified.
        :return: Optional[str]
        """

        params: Dict = dict()
        params["symbol"] = symbol_name
        params["side"] = side.value
        params["quantity"] = str(quantity)
        params["type"] = order_type.value

        if price and order_type != "MARKET":
            params["price"] = str(price)
        if order_type.value != "MARKET":
            time_in_force = "GTX" if _kwargs.get("post_only", False) else "GTC"
            params["timeInForce"] = time_in_force

        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._generate_authed_request_signature(params)

        endpoint = "/fapi/v1/order"
        response = self._make_request(HTTPMethod.POST, endpoint, params, self._headers)
        if not response:
            return None

        return response["orderId"]

    def get_order_by_id(self, symbol_name: str, order_id: str) -> Optional[Order]:
        """Query an order by ID.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to query.
        :return: Optional[Order]
        """

        params: Dict = dict()
        params["symbol"] = symbol_name
        params["orderId"] = order_id
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._generate_authed_request_signature(params)

        endpoint = "/fapi/v1/order"
        response = self._make_request(
            HTTPMethod.GET, endpoint, params, headers=self._headers
        )
        if not response:
            logger.error(
                "Error while fetching %s order info by ID %s", symbol_name, order_id
            )
            return None

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

        return Order(
            symbol=symbol_name,
            order_id=order_id,
            side=order_side,
            price=float(response["price"]),
            average_price=float(response["avgPrice"]),
            quantity=order_quantity,
            executed_quantity=executed_quantity,
            remaining_quantity=remaining_quantity,
            order_type=order_type,
            order_status=response["status"],
            time_in_force=response["timeInForce"],
        )

    def cancel_order(self, symbol_name: str, order_id: str) -> bool:
        """Cancel an active order on the exchange. Returns a bool on whether
        order cancellation was successful.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to cancel.
        :return: bool
        """

        params: Dict = dict()
        params["symbol"] = symbol_name
        params["orderId"] = order_id
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._generate_authed_request_signature(params)

        endpoint = "/fapi/v1/order"
        response = self._make_request(
            HTTPMethod.DELETE, endpoint, params, headers=self._headers
        )
        if response:
            return True

        return False
