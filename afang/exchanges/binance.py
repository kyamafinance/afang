import logging
from typing import Dict, List, Optional

from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Candle

logger = logging.getLogger(__name__)


class BinanceExchange(IsExchange):
    """Interface to run exchange functions on Binance USDT Futures."""

    def __init__(self) -> None:
        name = "binance"
        base_url = "https://fapi.binance.com"

        super().__init__(name, base_url)

    @classmethod
    def get_config_params(cls) -> Dict:
        """Get configuration parameters unique to the exchange.

        :return: dict
        """

        return {"query_limit": 1.1, "write_limit": 10000}

    def _get_symbols(self) -> List[str]:
        """Fetch all Binance USDT Futures symbols.

        :return: List[str]
        """

        symbols: List[str] = []
        params: Dict = dict()
        endpoint = "/fapi/v1/exchangeInfo"

        data = self._make_request(endpoint, params)
        if data is None:
            return symbols

        for symbol in data.get("symbols"):
            if symbol.get("contractType") == "PERPETUAL":
                symbols.append(symbol.get("symbol"))

        return symbols

    def get_historical_data(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[List[Candle]]:
        """Fetch candlestick bars for a particular symbol from the Binance
        exchange. If start_time and end_time are not sent, the most recent
        klines are returned.

        :param symbol: symbol to fetch historical candlestick bars for.
        :param start_time: optional. the start time to begin fetching candlestick bars as a UNIX timestamp in ms.
        :param end_time: optional. the end time to begin fetching candlestick bars as a UNIX timestamp in ms.

        :return: Optional[List[Candle]]
        """

        params: Dict = dict()
        params["symbol"] = symbol
        params["interval"] = "1m"
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

        raw_candles = self._make_request(endpoint, params)
        if raw_candles is None:
            return None

        candles: List[Candle] = []
        for candle in raw_candles:
            candles.append(
                Candle(
                    open_time=float(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[7]),
                )
            )

        return candles
