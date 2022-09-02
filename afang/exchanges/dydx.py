import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import dateutil.parser
import dydx3
import pytz
from dotenv import load_dotenv
from dydx3 import Client
from dydx3.errors import DydxApiError

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
    M1 = "1MIN"
    M5 = "5MINS"
    M15 = "15MINS"
    M30 = "30MINS"
    H1 = "1HOUR"
    H4 = "4HOURS"
    D1 = "1DAY"


class DyDxExchange(IsExchange):
    """Interface to run exchange functions on DyDx Futures."""

    def __init__(self, mode: Optional[Mode] = None, testnet: bool = False) -> None:
        """
        :param testnet: whether to use the testnet version of the exchange.
        """

        name = "dydx"
        base_url = "https://api.dydx.exchange"
        wss_url = "wss://api.dydx.exchange/v3/ws"
        if testnet:
            base_url = "https://api.stage.dydx.exchange"
            wss_url = "wss://api.stage.dydx.exchange/v3/ws"

        super().__init__(name, mode, testnet, base_url, wss_url)

        # dYdX exchange environment variables.
        self._DYDX_API_KEY = os.environ.get("DYDX_API_KEY")
        self._DYDX_API_SECRET = os.environ.get("DYDX_API_SECRET")
        self._DYDX_API_PASSPHRASE = os.environ.get("DYDX_API_PASSPHRASE")
        self._DYDX_STARK_PRIVATE_KEY = os.environ.get("DYDX_STARK_PRIVATE_KEY")
        self._DYDX_ETHEREUM_ADDRESS = os.environ.get("DYDX_ETHEREUM_ADDRESS")

        # dYdX API client.
        self._account_position_id: Optional[str] = None
        self._api_client: Optional[dydx3.Client] = self._get_api_client()

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

        :return: dict
        """

        return {"query_limit": 0.2, "write_limit": 20000}

    def _get_symbols(self) -> Dict[str, Symbol]:
        """Fetch all DyDx Futures symbols.

        :return: List[str]
        """

        symbols: Dict[str, Symbol] = dict()
        params: Dict = dict()
        endpoint = "/v3/markets"

        data = self._make_request(HTTPMethod.GET, endpoint, params)
        if data is None:
            return symbols

        if "markets" not in data:
            return symbols

        for symbol_name in data.get("markets"):
            symbol = data.get("markets").get(symbol_name)
            if symbol.get("type") == "PERPETUAL":
                symbols[symbol_name] = Symbol(
                    name=symbol_name,
                    base_asset=symbol.get("baseAsset"),
                    quote_asset=symbol.get("quoteAsset"),
                    price_decimals=get_float_precision(symbol.get("tickSize")),
                    quantity_decimals=get_float_precision(symbol.get("stepSize")),
                    tick_size=float(symbol.get("tickSize")),
                    step_size=float(symbol.get("stepSize")),
                )

        return symbols

    def get_historical_candles(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timeframe: Timeframe = Timeframe.M1,
    ) -> Optional[List[Candle]]:
        """Fetch candlestick bars for a particular symbol from the DyDx
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

        candle_fetch_limit = 100

        params = dict()
        params["resolution"] = tf_interval
        params["limit"] = str(candle_fetch_limit)

        timezone = pytz.UTC
        if start_time:
            start_time_seconds = int(start_time / 1000)
            end_time_seconds = int(start_time / 1000) + (candle_fetch_limit * 60)
            params["fromISO"] = datetime.fromtimestamp(
                start_time_seconds, timezone
            ).isoformat()
            params["toISO"] = datetime.fromtimestamp(
                end_time_seconds, timezone
            ).isoformat()
        if end_time:
            end_time_seconds = int(end_time / 1000)
            params["toISO"] = datetime.fromtimestamp(
                end_time_seconds, timezone
            ).isoformat()

        endpoint = f"/v3/candles/{symbol}"

        raw_candles = self._make_request(HTTPMethod.GET, endpoint, params)
        if raw_candles is None:
            return None

        if "candles" not in raw_candles:
            return None

        raw_candles = list(reversed(raw_candles["candles"]))  # reverse candle order

        candles: List[Candle] = []
        for candle in raw_candles:
            candles.append(
                Candle(
                    open_time=int(
                        dateutil.parser.isoparse(candle["startedAt"]).timestamp() * 1000
                    ),
                    open=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"]),
                    volume=float(candle["usdVolume"]),
                )
            )

        return candles

    def _get_api_client(self) -> Optional[dydx3.Client]:
        """Get a dYdX API client instance and account position ID.

        :return: Optional[dydx3.Client]
        """

        # If the mode does not involve authenticated exchange operations,
        # there is no need to attempt instantiating an API client.
        if self.mode != Mode.trade:
            return None

        # Get API client instance.
        try:
            api_client = Client(
                host=self._base_url,
                network_id=3 if self.testnet else 1,
                api_key_credentials={
                    "key": self._DYDX_API_KEY,
                    "secret": self._DYDX_API_SECRET,
                    "passphrase": self._DYDX_API_PASSPHRASE,
                },
                stark_private_key=self._DYDX_STARK_PRIVATE_KEY,
                default_ethereum_address=self._DYDX_ETHEREUM_ADDRESS,
            )
        except ValueError:
            logger.warning(
                "dYdX API client could not be set up probably due to a lack of exchange credentials. "
                "Ignore for unauthenticated operations."
            )
            return None

        # Get account position ID.
        try:
            account_response = api_client.private.get_account()
            self._account_position_id = account_response.data["account"]["positionId"]
            return api_client
        except DydxApiError as dydx_api_err:
            logger.warning(
                "DydxApiError raised when attempting to get account position ID: %s "
                "Ignore for unauthenticated operations.",
                dydx_api_err.msg,
            )

        return None

    def place_order(
        self,
        symbol_name: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float] = None,
        **_kwargs,
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
            dydx_limit_fee: float: highest accepted fee for the trade.
        :return: Optional[str]
        """

        order_params: Dict = dict()
        order_params["position_id"] = self._account_position_id
        order_params["market"] = symbol_name
        order_params["side"] = side.value
        order_params["order_type"] = order_type.value
        order_params["size"] = str(quantity)
        # TODO: If no price was provided and market order, default to current market price + 100.
        order_params["price"] = str(price)
        order_params["expiration_epoch_seconds"] = time.time() + 604800  # 1W
        order_params["post_only"] = _kwargs.get("post_only", False)
        order_params["limit_fee"] = _kwargs.get("dydx_limit_fee", 0.0005)

        time_in_force = "GTT" if order_type.value != "MARKET" else "FOK"
        order_params["time_in_force"] = time_in_force

        try:
            order = self._api_client.private.create_order(**order_params)
            return order.data["order"]["id"]
        except DydxApiError as dydx_api_err:
            logger.error(
                "DydxApiError raised when attempting to place new %s order: %s",
                symbol_name,
                dydx_api_err,
            )

        return None

    def get_order_by_id(self, symbol_name: str, order_id: str) -> Optional[Order]:
        """Query an order by ID.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to query.
        :return: Optional[Order]
        """

        try:
            order_res = self._api_client.private.get_order_by_id(order_id)
            fills_res = self._api_client.private.get_fills(symbol_name, order_id, 100)
            order = order_res.data["order"]
            fills = fills_res.data["fills"]

            average_price: float = 0.0
            total_filled_quantity = sum(float(fill["size"]) for fill in fills)
            for fill in fills:
                average_price += (float(fill["size"]) / total_filled_quantity) * float(
                    fill["price"]
                )

            order_quantity = float(order["size"])
            remaining_quantity = float(order["remainingSize"])
            executed_quantity = order_quantity - remaining_quantity

            order_side = OrderSide.UNKNOWN
            if order["side"] == "BUY":
                order_side = OrderSide.BUY
            elif order["side"] == "SELL":
                order_side = OrderSide.SELL

            order_type = OrderType.UNKNOWN
            if order["type"] == "LIMIT":
                order_type = OrderType.LIMIT
            elif order["type"] == "MARKET":
                order_type = OrderType.MARKET

            return Order(
                symbol=symbol_name,
                order_id=order_id,
                side=order_side,
                original_price=float(order["price"]),
                average_price=average_price,
                original_quantity=order_quantity,
                executed_quantity=executed_quantity,
                remaining_quantity=remaining_quantity,
                order_type=order_type,
                order_status=order["status"],
                time_in_force=order["timeInForce"],
            )
        except DydxApiError as dydx_api_err:
            logger.error(
                "DydxApiError raised when attempting to get %s order status with ID %s: %s",
                symbol_name,
                order_id,
                dydx_api_err,
            )

        return None

    def cancel_order(self, symbol_name: str, order_id: str) -> bool:
        """Cancel an active order on the exchange. Returns a bool on whether
        order cancellation was successful.

        :param symbol_name: name of symbol.
        :param order_id: ID of the order to cancel.
        :return: bool
        """

        try:
            self._api_client.private.cancel_order(order_id)
            return True
        except DydxApiError as dydx_api_err:
            logger.error(
                "DydxApiError raised when attempting to cancel %s order with ID %s: %s",
                symbol_name,
                order_id,
                dydx_api_err,
            )

        return False
