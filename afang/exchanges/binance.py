import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import websocket
from dotenv import load_dotenv

from afang.exchanges.is_exchange import ExchangeTimeframeMapping, IsExchange
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
from afang.utils.util import get_float_precision

load_dotenv()
logger = logging.getLogger(__name__)


class TimeframeMapping(ExchangeTimeframeMapping):
    """Maps application recognized timeframe names to their corresponding
    values on the exchange."""

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

        self._wss_stream_id: int = 1
        self._wss_listen_key: Optional[str] = None
        self._wss: Optional[websocket.WebSocketApp] = None

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
            original_price=float(response["price"]),
            average_price=float(response["avgPrice"]),
            original_quantity=order_quantity,
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

    def _get_asset_balances(self) -> None:
        """Get the wallet balances of all assets on the exchange.

        :return: None
        """

        params: Dict = dict()
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._generate_authed_request_signature(params)

        endpoint = "/fapi/v2/account"
        response = self._make_request(
            HTTPMethod.GET, endpoint, params, headers=self._headers
        )
        if not response:
            logger.error("%s: error while fetching asset balances", self.display_name)
            return None

        for asset in response["assets"]:
            self.trading_symbol_balance[asset["asset"]] = SymbolBalance(
                name=asset["asset"], wallet_balance=float(asset["walletBalance"])
            )

    def _subscribe_wss_candlestick_stream(self) -> None:
        """Subscribe to the exchange wss candlestick stream for all provided
        symbols for the given timeframe.

        :return: None
        """

        wss_data: Dict[str, Any] = dict()
        wss_data["method"] = "SUBSCRIBE"
        wss_data["params"] = list()

        for symbol in self.trading_symbols:
            wss_data["params"].append(
                f"{symbol.lower()}_perpetual@continuousKline_{self.trading_timeframe.value}"
            )
            wss_data["id"] = self._wss_stream_id
            self._wss_stream_id += 1
            self._wss.send(json.dumps(wss_data))

    def _wss_on_open(self, _ws: websocket.WebSocketApp) -> None:
        """Runs when the websocket connection is opened.

        :param _ws: instance of websocket connection.
        :return: None
        """

        logger.info("%s: wss connection opened", self.display_name)
        self._subscribe_wss_candlestick_stream()

    def _wss_on_close(self, _ws: websocket.WebSocketApp) -> None:
        """Runs when the websocket connection is closed.

        :param _ws: instance of websocket connection.
        :return: None
        """

        logger.warning("%s: wss connection closed", self.display_name)

    def _wss_on_error(self, _ws: websocket.WebSocketApp, msg: str) -> None:
        """Runs when there is an error in websocket connection.

        :param _ws: instance of websocket connection.
        :param msg: error message.
        :return: None
        """

        logger.error("%s: wss connection error: %s", self.display_name, msg)

    def _wss_handle_listen_key_expired(self, _msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that the wss
        listen key has expired.

        :param _msg_data: corresponding websocket message.
        :return: None
        """

        logger.error("%s: wss listen key expired", self.display_name)

    def _wss_handle_margin_call(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that a user's
        position risk ratio is too high.

        :param msg_data: corresponding websocket message.
        :return: None
        """

        logger.warning(
            "%s: position risk ratio is too high for symbols: %s",
            self.display_name,
            ", ".join([position["s"] for position in msg_data["p"]]),
        )

    def _wss_handle_asset_balance_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to an asset's balance.

        :param msg_data: corresponding websocket message.
        :return: None
        """

        if "a" not in msg_data or "B" not in msg_data["a"]:
            return

        assets = msg_data["a"]["B"]
        for asset in assets:
            self.trading_symbol_balance[asset["a"]] = SymbolBalance(
                name=asset["a"], wallet_balance=float(asset["wb"])
            )

    def _wss_handle_order_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to an order.

        :param msg_data: corresponding websocket message.
        :return: None
        """

        msg_order = msg_data["o"]
        msg_order_id = str(msg_order["i"])

        order_quantity = float(msg_order["q"])
        executed_quantity = float(msg_order["z"])
        remaining_quantity = order_quantity - executed_quantity

        order_side = OrderSide.UNKNOWN
        if msg_order["S"] == "BUY":
            order_side = OrderSide.BUY
        elif msg_order["S"] == "SELL":
            order_side = OrderSide.SELL

        order_type = OrderType.UNKNOWN
        if msg_order["o"] == "LIMIT":
            order_type = OrderType.LIMIT
        elif msg_order["o"] == "MARKET":
            order_type = OrderType.MARKET

        prev_order_commission = 0.0
        if msg_order_id in self.active_orders:
            prev_order = self.active_orders[msg_order_id]
            prev_order_commission = prev_order.commission
        current_order_commission = float(msg_order["n"]) if "n" in msg_order else 0.0
        current_commission_tally = prev_order_commission + current_order_commission

        updated_order = Order(
            symbol=msg_order["s"],
            order_id=msg_order_id,
            side=order_side,
            original_price=float(msg_order["p"]) if float(msg_order["p"]) else None,
            average_price=float(msg_order["ap"]),
            original_quantity=order_quantity,
            executed_quantity=executed_quantity,
            remaining_quantity=remaining_quantity,
            order_type=order_type,
            order_status=msg_order["X"],
            time_in_force=msg_order["f"],
            commission=current_commission_tally,
        )
        self.active_orders[msg_order_id] = updated_order

    def _wss_handle_candlestick_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to a symbol's candlestick.

        :param msg_data: corresponding websocket message.
        :return: None
        """

        if "k" not in msg_data:
            return

        candlestick_data = msg_data["k"]
        candlestick_symbol = msg_data["ps"]

        updated_candlestick = Candle(
            open_time=int(candlestick_data["t"]),
            open=float(candlestick_data["o"]),
            high=float(candlestick_data["h"]),
            low=float(candlestick_data["l"]),
            close=float(candlestick_data["c"]),
            volume=float(candlestick_data["q"]),
        )

        most_recent_symbol_candlestick = self.trading_price_data[candlestick_symbol][-1]
        last_recorded_candlestick_open_time = most_recent_symbol_candlestick.open_time
        if last_recorded_candlestick_open_time == updated_candlestick.open_time:
            self.trading_price_data[candlestick_symbol][-1] = updated_candlestick
        else:
            self.trading_price_data[candlestick_symbol].append(updated_candlestick)
            self.trading_price_data[candlestick_symbol].pop(0)

    def _wss_on_message(self, _ws: websocket.WebSocketApp, msg: str) -> None:
        """Runs whenever a message is received by the websocket connection.

        :param _ws: instance of websocket connection.
        :param msg: received message.
        :return: None
        """

        msg_data = json.loads(msg)
        if "e" not in msg_data:
            return None

        event_type = msg_data["e"]

        if event_type == "listenKeyExpired":
            self._wss_handle_listen_key_expired(msg_data)
        elif event_type == "MARGIN_CALL":
            self._wss_handle_margin_call(msg_data)
        elif event_type == "ACCOUNT_UPDATE":
            self._wss_handle_asset_balance_update(msg_data)
        elif event_type == "ORDER_TRADE_UPDATE":
            self._wss_handle_order_update(msg_data)
        elif event_type == "continuous_kline":
            self._wss_handle_candlestick_update(msg_data)

    def _fetch_wss_listen_key(self) -> Optional[str]:
        """Fetch the account's listen key and extend its validity for 60
        minutes.

        :return: Optional[str]
        """

        params: Dict = dict()
        endpoint = "/fapi/v1/listenKey"

        response = self._make_request(
            HTTPMethod.POST, endpoint, params, headers=self._headers
        )
        if not response:
            logger.error("%s: error fetching wss listen key", self.display_name)
            return None

        return response["listenKey"]

    def _keep_wss_alive(self) -> None:
        """Periodically extend the validity period of the wss listen key.

        :return: None
        """

        while True:
            wss_listen_key = self._fetch_wss_listen_key()
            if not wss_listen_key:
                logger.warning(
                    "%s: could not extend the validity of the wss listen key. Retrying...",
                    self.display_name,
                )
                time.sleep(5)
                continue

            self._wss_listen_key = wss_listen_key
            time.sleep(40 * 60)

    def _start_wss(self) -> None:
        """Open a websocket connection to the exchange.

        :return: None
        """

        self._wss_listen_key = self._fetch_wss_listen_key()
        if not self._wss_listen_key:
            logger.error(
                "%s: could not start wss due to lack of a valid listen key",
                self.display_name,
            )
            return None

        self._wss = websocket.WebSocketApp(
            f"{self._wss_url}/{self._wss_listen_key}",
            on_open=self._wss_on_open,
            on_close=self._wss_on_close,
            on_message=self._wss_on_message,
            on_error=self._wss_on_error,
        )
        wss_thread = threading.Thread(target=self._wss.run_forever)
        wss_thread.start()

        # Keep websocket listen key valid.
        wss_listen_key_thread = threading.Thread(target=self._keep_wss_alive)
        wss_listen_key_thread.start()

    def setup_exchange_for_trading(
        self, symbols: List[str], timeframe: Timeframe
    ) -> None:
        """Set up the exchange for live or demo trading.

        :param symbols: exchange symbols to be traded.
        :param timeframe: desired trading timeframe.
        :return: None
        """

        # Populate trading symbols and timeframe.
        self._populate_trading_symbols(symbols)
        supported_exchange_timeframes = [tf.name for tf in TimeframeMapping]
        self._populate_trading_timeframe(timeframe, supported_exchange_timeframes)

        # Populate initial trading symbol balances.
        self._get_asset_balances()

        # Populate initial price data.
        self._populate_initial_trading_price_data(num_iterations=1)

        # Open exchange websocket connection.
        self._start_wss()

    def change_initial_leverage(self, symbols: List[str], leverage: int) -> None:
        """Change initial leverage for specific symbols.

        :param symbols: symbols whose initial leverage will be changed.
        :param leverage: updated leverage.
        :return: None
        """

        for symbol in symbols:
            params = dict()
            params["symbol"] = symbol
            params["leverage"] = str(leverage)
            params["timestamp"] = str(int(time.time() * 1000))
            params["signature"] = self._generate_authed_request_signature(params)

            endpoint = "/fapi/v1/leverage"
            response = self._make_request(
                HTTPMethod.POST, endpoint, params, headers=self._headers
            )
            if not response:
                logger.warning(
                    "%s: could not change %s initial leverage to %s",
                    self.display_name,
                    symbol,
                    leverage,
                )
                continue

            self.symbol_leverage[symbol] = leverage
            logger.info(
                "%s: changed %s initial leverage to %s",
                self.display_name,
                symbol,
                leverage,
            )
