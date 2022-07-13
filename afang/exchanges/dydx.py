import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import dateutil.parser as dp
import pytz

from afang.exchanges.is_exchange import IsExchange

logger = logging.getLogger()


class DyDxExchange(IsExchange):
    """Interface to run exchange functions on DyDx Futures."""

    def __init__(self) -> None:
        """Initialize DyDxClient class."""

        name = "dydx"
        base_url = "https://api.dydx.exchange"

        super().__init__(name, base_url)

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
    ) -> Optional[List[Tuple[float, float, float, float, float, float]]]:
        """Fetch candlestick bars for a particular symbol from the DyDx
        exchange. If start_time and end_time are not sent, the most recent
        klines are returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.

        :return: Optional[List[Tuple[float, float, float, float, float, float]]]
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

        candles = []
        for candle in raw_candles:
            candles.append(
                (
                    float(
                        dp.parse(candle["startedAt"]).timestamp() * 1000
                    ),  # open time
                    float(candle["open"]),  # open
                    float(candle["high"]),  # high
                    float(candle["low"]),  # low
                    float(candle["close"]),  # close
                    float(candle["usdVolume"]),  # volume
                )
            )

        return candles
