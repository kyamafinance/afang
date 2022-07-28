import os
import pathlib
import shutil
from abc import ABC
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from afang.exchanges.is_exchange import IsExchange
from afang.strategies.is_strategy import IsStrategy
from afang.strategies.util import TradeLevels


@pytest.fixture()
def ohlcv_root_db_dir() -> str:
    return f"{pathlib.Path(__file__).parent}/testdata/database/ohlcv"


@pytest.fixture(autouse=True)
def delete_ohlcv_databases(ohlcv_root_db_dir) -> Generator:
    yield
    list_dir = os.listdir(ohlcv_root_db_dir)
    for filename in list_dir:
        if filename == ".gitkeep":
            continue

        file_path = os.path.join(ohlcv_root_db_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


@pytest.fixture
def dummy_is_exchange() -> IsExchange:
    class Dummy(IsExchange):
        def __init__(self, name: str, base_url: str) -> None:
            super().__init__(name, base_url)

        @classmethod
        def get_config_params(cls) -> Dict:
            return super().get_config_params()

        def _get_symbols(self) -> List[str]:
            return super()._get_symbols()

        def _make_request(self, endpoint: str, query_parameters: Dict) -> Any:
            return super()._make_request(endpoint, query_parameters)

        def get_historical_data(
            self,
            symbol: str,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
        ) -> Optional[List[Tuple[float, float, float, float, float, float]]]:
            return super().get_historical_data(symbol, start_time, end_time)

    return Dummy(name="test_exchange", base_url="https://dummy.com")


@pytest.fixture
def dummy_is_strategy() -> IsStrategy:
    class Dummy(IsStrategy, ABC):
        def __init__(self, strategy_name: str) -> None:
            super().__init__(strategy_name)

        def read_strategy_config(self) -> Dict:
            return {
                "name": "test_strategy",
                "timeframe": "1h",
                "watchlist": {"test_exchange": ["test_symbol"]},
            }

        def generate_features(self, data: pd.DataFrame) -> None:
            return super().generate_features(data)

        def is_long_trade_signal_present(self, data: Any) -> bool:
            return super().is_long_trade_signal_present(data)

        def is_short_trade_signal_present(self, data: Any) -> bool:
            return super().is_short_trade_signal_present(data)

        def generate_trade_levels(
            self, data: Any, trade_signal_direction: int
        ) -> TradeLevels:
            return super().generate_trade_levels(data, trade_signal_direction)

        def plot_backtest_indicators(self) -> Dict:
            return super().plot_backtest_indicators()

    return Dummy(strategy_name="test_strategy")
