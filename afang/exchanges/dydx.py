import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import dateutil.parser
import dydx3
import pytz
import websocket
from dotenv import load_dotenv
from dydx3 import Client
from dydx3.errors import DydxApiError

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
from afang.models import Timeframe
from afang.utils.util import get_float_precision, round_float_to_precision

load_dotenv()
logger = logging.getLogger(__name__)


class TimeframeMapping(ExchangeTimeframeMapping):
    """Maps application recognized timeframe names to their corresponding
    values on the exchange."""

    M1 = "1MIN"
    M5 = "5MINS"
    M15 = "15MINS"
    M30 = "30MINS"
    H1 = "1HOUR"
    H4 = "4HOURS"
    D1 = "1DAY"


class DyDxExchange(IsExchange):
    """Interface to run exchange functions on DyDx Futures."""

    def __init__(self, testnet: bool = False) -> None:
        """
        :param testnet: whether to use the testnet version of the exchange.
        """
        name = "dydx"
        base_url = "https://api.dydx.exchange"
        wss_url = "wss://api.dydx.exchange/v3/ws"
        if testnet:
            base_url = "https://api.stage.dydx.exchange"
            wss_url = "wss://api.stage.dydx.exchange/v3/ws"

        super().__init__(name, testnet, base_url, wss_url)

        # dYdX exchange environment variables.
        self._DYDX_NETWORK_ID = 1
        self._DYDX_API_KEY = os.environ.get("DYDX_API_KEY")
        self._DYDX_API_SECRET = os.environ.get("DYDX_API_SECRET")
        self._DYDX_API_PASSPHRASE = os.environ.get("DYDX_API_PASSPHRASE")
        self._DYDX_STARK_PRIVATE_KEY = os.environ.get("DYDX_STARK_PRIVATE_KEY")
        self._DYDX_ETHEREUM_ADDRESS = os.environ.get("DYDX_ETHEREUM_ADDRESS")
        if testnet:
            self._DYDX_NETWORK_ID = 5
            self._DYDX_API_KEY = os.environ.get("DYDX_TESTNET_API_KEY")
            self._DYDX_API_SECRET = os.environ.get("DYDX_TESTNET_API_SECRET")
            self._DYDX_API_PASSPHRASE = os.environ.get("DYDX_TESTNET_API_PASSPHRASE")
            self._DYDX_STARK_PRIVATE_KEY = os.environ.get(
                "DYDX_TESTNET_STARK_PRIVATE_KEY"
            )
            self._DYDX_ETHEREUM_ADDRESS = os.environ.get(
                "DYDX_TESTNET_ETHEREUM_ADDRESS"
            )

        # dYdX API client.
        self._account_position_id: Optional[str] = None
        self._api_client: Optional[dydx3.Client] = None

        self._quote_balance: Optional[float] = None
        self._oracle_prices: Dict[str, float] = dict()
        self._symbol_total_pos_size: Dict[str, float] = defaultdict(float)
        self._wss: Optional[websocket.WebSocketApp] = None

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
        try:
            api_client = Client(
                host=self._base_url,
                network_id=self._DYDX_NETWORK_ID,
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

        NOTE: The `price` parameter is required even for MARKET orders.
            The best way to handle this is to pass the price as the current symbol price + 100.

        :param symbol_name: name of symbol.
        :param side: order side.
        :param quantity: order quantity.
        :param order_type: order type.
        :param price: optional. order price.
        :param _kwargs:
            post_only bool: order will only be allowed if it will enter the order book.
            dydx_limit_fee: Optional[float]: highest accepted fee for the trade.
        :return: Optional[str]
        """

        order_params: Dict = dict()
        order_params["position_id"] = self._account_position_id
        order_params["market"] = symbol_name
        order_params["side"] = side.value
        order_params["order_type"] = order_type.value

        precise_order_qty = round_float_to_precision(
            quantity,
            self.exchange_symbols.get(symbol_name).step_size,
        )
        order_params["size"] = str(precise_order_qty)

        if price:
            precise_order_price = round_float_to_precision(
                price,
                self.exchange_symbols.get(symbol_name).tick_size,
            )
            order_params["price"] = str(precise_order_price)
        if (
            order_type.value == "MARKET"
            and symbol_name in self.trading_price_data
            and self.trading_price_data[symbol_name]
        ):
            order_market_price = round_float_to_precision(
                float(self.trading_price_data[symbol_name][-1].close) + 100,
                self.exchange_symbols.get(symbol_name).tick_size,
            )
            order_params["price"] = str(order_market_price)

        order_params["post_only"] = _kwargs.get("post_only", False)
        order_params["expiration_epoch_seconds"] = time.time() + 604800  # 1W
        order_params["limit_fee"] = (
            _kwargs.get("dydx_limit_fee") if _kwargs.get("dydx_limit_fee") else 0.0005
        )

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
        except Exception as e:
            logger.error("Failed to place new %s order: %s", symbol_name, str(e))

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
            commission = sum(float(fill["fee"]) for fill in fills)
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
                commission=commission,
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

    def _wss_subscribe_trades_stream(self) -> None:
        """Subscribe to the exchange trades stream.

        :return: None
        """
        for symbol in self.trading_symbols:
            wss_data: Dict[str, Any] = dict()
            wss_data["type"] = "subscribe"
            wss_data["channel"] = "v3_trades"
            wss_data["id"] = symbol
            self._wss.send(json.dumps(wss_data))

    def _wss_subscribe_markets_stream(self) -> None:
        """Subscribe to the exchange wss markets stream.

        :return: None
        """
        wss_data: Dict[str, Any] = dict()
        wss_data["type"] = "subscribe"
        wss_data["channel"] = "v3_markets"

        self._wss.send(json.dumps(wss_data))

    def _wss_subscribe_accounts_stream(self) -> None:
        """Subscribe to the exchange wss accounts stream.

        :return: None
        """
        current_time_iso = datetime.utcnow().isoformat()

        endpoint = "/ws/accounts"
        signature = self._api_client.private.sign(
            request_path=endpoint,
            method=HTTPMethod.GET.value,
            iso_timestamp=current_time_iso,
            data={},
        )

        wss_data: Dict[str, Any] = dict()
        wss_data["type"] = "subscribe"
        wss_data["channel"] = "v3_accounts"
        wss_data["accountNumber"] = "0"
        wss_data["apiKey"] = self._DYDX_API_KEY
        wss_data["passphrase"] = self._DYDX_API_PASSPHRASE
        wss_data["timestamp"] = current_time_iso
        wss_data["signature"] = signature

        self._wss.send(json.dumps(wss_data))

    def _wss_on_open(self, _ws: websocket.WebSocketApp) -> None:
        """Runs when the websocket connection is opened.

        :param _ws: instance of websocket connection.
        :return: None
        """
        logger.info("%s: wss connection opened", self.display_name)
        self._wss_subscribe_trades_stream()
        self._wss_subscribe_markets_stream()
        self._wss_subscribe_accounts_stream()

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

    def _update_collateral_balance(self, run_forever: bool = True) -> None:
        """Constantly updates the collateral balance value.

        - collateral balance value = Q + Σ(Si × Pi) where:
            - Q = quote balance.
            - S = size of the position (positive if long, negative if short).
            - P = oracle price for the market.

        :param run_forever: whether to continuously update collateral balance. used for testing purposes.
        :return: None
        """
        while True:
            if run_forever:
                time.sleep(5)

            if not self._symbol_total_pos_size:
                if run_forever:
                    continue
                break

            if not self._quote_balance:
                if run_forever:
                    continue
                break

            if not self._oracle_prices:
                if run_forever:
                    continue
                break

            total_position_value = 0.0
            for symbol, pos_size in self._symbol_total_pos_size.items():
                total_position_value += self._oracle_prices[symbol] * pos_size

            collateral_balance = self._quote_balance + total_position_value
            self.trading_symbol_balance["USD"] = SymbolBalance(
                name="USD", wallet_balance=collateral_balance
            )

            if not run_forever:
                break

    def _wss_handle_oracle_price_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to a market.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]

        if "markets" in msg_content:
            msg_content = msg_content["markets"]

        for symbol_name, updated_market_info in msg_content.items():
            if "oraclePrice" in updated_market_info:
                self._oracle_prices[symbol_name] = float(
                    updated_market_info["oraclePrice"]
                )

    def __wss_update_quote_balance(self, msg_data: Any) -> None:
        """Updates the quote balance given an accounts stream wss message.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]

        if "account" in msg_content:
            account = msg_content["account"]
            self._quote_balance = float(account["quoteBalance"])

        if "accounts" in msg_content:
            accounts = msg_content["accounts"]
            account_0 = next(
                (account for account in accounts if int(account["accountNumber"]) == 0),
                None,
            )
            if account_0:
                self._quote_balance = float(account_0["quoteBalance"])

    def __wss_update_total_position_size(self, msg_data: Any) -> None:
        """Updates the total position size of all open positions given an
        accounts stream wss message.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]
        symbol_total_pos_size: Dict[str, float] = defaultdict(float)

        if "positions" not in msg_content:
            return None

        for position in msg_content["positions"]:
            symbol_name = position["market"]
            pos_size = float(position["size"])
            symbol_total_pos_size[symbol_name] += pos_size

        self._symbol_total_pos_size.update(symbol_total_pos_size)

    def _wss_handle_collateral_balance_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to the account collateral balance.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]

        self.__wss_update_quote_balance(msg_data)
        if not self._quote_balance:
            return None

        if "USD" not in self.trading_symbol_balance:
            if "account" in msg_content and "equity" in msg_content["account"]:
                equity = msg_content["account"]["equity"]
                self.trading_symbol_balance["USD"] = SymbolBalance(
                    name="USD", wallet_balance=float(equity)
                )
                return None

        self.__wss_update_total_position_size(msg_data)

    def _wss_handle_order_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to an order.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]

        orders = msg_content["orders"] if "orders" in msg_content else list()
        fills = msg_content["fills"] if "fills" in msg_content else list()

        for order in orders:
            order_id = order["id"]

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

            prev_order_commission = 0.0
            prev_order_executed_qty = 0.0
            prev_order_average_price = 0.0
            if order_id in self._active_orders:
                prev_order = self._active_orders[order_id]
                prev_order_commission = prev_order.commission
                prev_order_average_price = prev_order.average_price
                prev_order_executed_qty = prev_order.executed_quantity

            order_fills = [fill for fill in fills if fill["orderId"] == order_id]

            average_price = prev_order_average_price
            prev_executed_qty = prev_order_executed_qty
            for fill in order_fills:
                current_executed_qty = prev_executed_qty + float(fill["size"])
                adjusted_average_price = (
                    prev_executed_qty / current_executed_qty
                ) * average_price
                fill_average_price = (
                    float(fill["size"]) / current_executed_qty
                ) * float(fill["price"])
                average_price = adjusted_average_price + fill_average_price
                prev_executed_qty += float(fill["size"])

            updated_commission = sum(float(fill["fee"]) for fill in order_fills)
            updated_commission += prev_order_commission

            updated_order = Order(
                symbol=order["market"],
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
                commission=updated_commission,
            )
            self._active_orders[order_id] = updated_order

    def _wss_handle_candlestick_update(self, msg_data: Any) -> None:
        """Runs when exchange websocket receives message data that there has
        been an update to a symbol's candlestick.

        :param msg_data: corresponding websocket message.
        :return: None
        """
        if "contents" not in msg_data:
            return None

        msg_content = msg_data["contents"]
        if "trades" not in msg_content:
            return None

        symbol_name = msg_data["id"]
        trades = msg_content["trades"]

        symbol_trading_candles = self.trading_price_data[symbol_name]
        if len(symbol_trading_candles) < 2:
            err_msg = (
                f"{self.display_name}: {symbol_name} candles cannot be updated because there are"
                f" too few initial candles"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        latest_candle_open_time = symbol_trading_candles[-1].open_time
        candle_time_difference = (
            latest_candle_open_time - symbol_trading_candles[-2].open_time
        )
        next_candle_open_time = latest_candle_open_time + candle_time_difference
        next_candle_close_time = next_candle_open_time + candle_time_difference

        for trade in trades:
            trade_time_iso = trade["createdAt"]
            trade_time_timestamp = int(
                dateutil.parser.isoparse(trade_time_iso).timestamp() * 1000
            )
            updated_candle_price = float(trade["price"])
            updated_candle_volume = float(trade["size"])

            if latest_candle_open_time <= trade_time_timestamp < next_candle_open_time:
                # update the current candlestick.
                self.trading_price_data[symbol_name][-1] = Candle(
                    open_time=symbol_trading_candles[-1].open_time,
                    open=symbol_trading_candles[-1].open,
                    high=max(symbol_trading_candles[-1].high, updated_candle_price),
                    low=min(symbol_trading_candles[-1].low, updated_candle_price),
                    close=updated_candle_price,
                    volume=symbol_trading_candles[-1].volume + updated_candle_volume,
                )

            elif next_candle_open_time <= trade_time_timestamp < next_candle_close_time:
                # add a new candlestick.
                self.trading_price_data[symbol_name].append(
                    Candle(
                        open_time=next_candle_open_time,
                        open=updated_candle_price,
                        high=updated_candle_price,
                        low=updated_candle_price,
                        close=updated_candle_price,
                        volume=updated_candle_volume,
                    )
                )
                self.trading_price_data[symbol_name].pop(0)

            elif trade_time_timestamp > next_candle_close_time:
                # missing candle(s) from persisted record.
                logger.error(
                    "%s: %s candles cannot be updated because candle(s) are potentially"
                    " missing from the persisted record."
                    " trade timestamp: %s latest candle open: %s timeframe: %s",
                    self.display_name,
                    symbol_name,
                    trade_time_timestamp,
                    latest_candle_open_time,
                    self.trading_timeframe.value,
                )

    def _wss_on_message(self, _ws: websocket.WebSocketApp, msg: str) -> None:
        """Runs whenever a message is received by the websocket connection.

        :param _ws: instance of websocket connection.
        :param msg: received message.
        :return: None
        """
        msg_data = json.loads(msg)
        if "channel" not in msg_data:
            return None

        channel_name = msg_data["channel"]

        if channel_name == "v3_markets":
            self._wss_handle_oracle_price_update(msg_data)
        elif channel_name == "v3_accounts":
            self._wss_handle_collateral_balance_update(msg_data)
            self._wss_handle_order_update(msg_data)
        elif channel_name == "v3_trades":
            self._wss_handle_candlestick_update(msg_data)

    def _start_wss(self) -> None:
        """Open a websocket connection to the exchange.

        :return: None
        """
        self._wss = websocket.WebSocketApp(
            self._wss_url,
            on_open=self._wss_on_open,
            on_close=self._wss_on_close,
            on_message=self._wss_on_message,
            on_error=self._wss_on_error,
        )
        wss_thread = threading.Thread(target=self._wss.run_forever)
        wss_thread.start()

    def _populate_initial_position_sizes(self) -> None:
        """Populate initial symbol position sizes for all open positions.

        :return: None
        """
        try:
            account = self._api_client.private.get_account()
            account_data = account.data

            if "openPositions" not in account_data["account"]:
                return None

            open_positions = account_data["account"]["openPositions"]
            for symbol_name, open_position in open_positions.items():
                pos_size = float(open_position["size"])
                self._symbol_total_pos_size[symbol_name] += pos_size

        except DydxApiError as dydx_api_err:
            logger.error(
                "DydxApiError raised when attempting to populate initial position sizes: %s",
                dydx_api_err,
            )

    def setup_exchange_for_trading(
        self, symbols: List[str], timeframe: Timeframe
    ) -> None:
        """Set up the exchange for live or demo trading.

        :param symbols: exchange symbols to be traded.
        :param timeframe: desired trading timeframe.
        :return: None
        """

        # Setup API client.
        self._api_client = self._get_api_client()

        # Populate trading symbols and timeframe.
        self._populate_trading_symbols(symbols)
        supported_exchange_timeframes = [tf.name for tf in TimeframeMapping]
        self._populate_trading_timeframe(timeframe, supported_exchange_timeframes)

        # Populate initial position sizes for open positions.
        self._populate_initial_position_sizes()

        # Constantly update collateral balance.
        balance_update_thread = threading.Thread(target=self._update_collateral_balance)
        balance_update_thread.start()

        # Populate initial price data.
        self._populate_initial_trading_price_data(num_iterations=10)

        # Open exchange websocket connection.
        self._start_wss()

    def change_initial_leverage(self, symbols: List[str], leverage: int) -> None:
        """Change initial leverage for specific symbols.

        :param symbols: symbols whose initial leverage will be changed.
        :param leverage: updated leverage.
        :return: None
        """
        for symbol in symbols:
            self.symbol_leverage[symbol] = leverage
            logger.info(
                "%s: changed %s initial leverage to %s",
                self.display_name,
                symbol,
                leverage,
            )
