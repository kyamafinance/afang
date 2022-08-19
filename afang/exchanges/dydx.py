import logging
from datetime import datetime
from typing import Dict, List, Optional

import dateutil.parser
import pytz

from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Candle

logger = logging.getLogger(__name__)


class DyDxExchange(IsExchange):
    """Interface to run exchange functions on DyDx Futures."""

    def __init__(self) -> None:
        """Initialize DyDxClient class."""

        name = "dydx"
        base_url = "https://api.dydx.exchange"

        super().__init__(name, base_url)

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

        :return: dict
        """

        return {"query_limit": 0.2, "write_limit": 20000}

    def _get_symbols(self) -> List[str]:
        """Fetch all DyDx Futures symbols.

        :return: List[str]
        """

        symbols: List[str] = []
        params: Dict = dict()
        endpoint = "/v3/markets"

        data = self._make_request(endpoint, params)
        if data is None:
            return symbols

        if "markets" not in data:
            return symbols

        for symbol_name in data.get("markets"):
            symbol = data.get("markets").get(symbol_name)
            if symbol.get("type") == "PERPETUAL":
                symbols.append(symbol_name)

        return symbols

    def get_historical_data(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[List[Candle]]:
        """Fetch candlestick bars for a particular symbol from the DyDx
        exchange. If start_time and end_time are not sent, the most recent
        klines are returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.

        :return: Optional[List[Candle]]
        """

        candle_fetch_limit = 100

        params = dict()
        params["resolution"] = "1MIN"
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

        raw_candles = self._make_request(endpoint, params)
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
