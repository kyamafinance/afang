import logging
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import dateutil.parser
import pytz
from dotenv import load_dotenv

from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Candle, HTTPMethod, Symbol
from afang.models import Timeframe
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

        self._DYDX_API_KEY = os.environ.get("DYDX_API_KEY")
        self._DYDX_API_SECRET = os.environ.get("DYDX_API_SECRET")
        self._DYDX_API_PASSPHRASE = os.environ.get("DYDX_API_PASSPHRASE")
        self._DYDX_STARK_PRIVATE_KEY = os.environ.get("DYDX_STARK_PRIVATE_KEY")
        self._DYDX_DEFAULT_ETHEREUM_ADDRESS = os.environ.get(
            "DYDX_DEFAULT_ETHEREUM_ADDRESS"
        )
        self.DYDX_CLIENT_API_HOST = os.environ.get("DYDX_CLIENT_API_HOST")

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
