import os
import pathlib
import shutil
from abc import ABC
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import pytest

from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import (
    Candle,
    HTTPMethod,
    Order,
    OrderSide,
    OrderType,
    Symbol,
)
from afang.models import Timeframe
from afang.strategies.is_strategy import IsStrategy
from afang.strategies.models import TradeLevels
from afang.strategies.optimizer import StrategyOptimizer


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


@pytest.fixture()
def optimization_root_dir() -> str:
    return f"{pathlib.Path(__file__).parent}/testdata/optimization"


@pytest.fixture(autouse=True)
def delete_optimization_records(optimization_root_dir) -> Generator:
    yield
    list_dir = os.listdir(optimization_root_dir)
    for filename in list_dir:
        if filename == ".gitkeep":
            continue

        file_path = os.path.join(optimization_root_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


@pytest.fixture
def dummy_is_exchange() -> IsExchange:
    class Dummy(IsExchange):
        def __init__(self, name: str, base_url: str, wss_url: str) -> None:
            super().__init__(name, False, base_url, wss_url)

        @classmethod
        def get_config_params(cls) -> Dict:
            return super().get_config_params()

        def _get_symbols(self) -> Dict[str, Symbol]:
            return super()._get_symbols()

        def _make_request(
            self,
            method: HTTPMethod,
            endpoint: str,
            query_parameters: Dict,
            headers: Optional[Dict] = None,
        ) -> Any:
            return super()._make_request(method, endpoint, query_parameters, headers)

        def get_historical_candles(
            self,
            symbol: str,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
            timeframe: Timeframe = Timeframe.M1,
        ) -> Optional[List[Candle]]:
            return super().get_historical_candles(
                symbol, start_time, end_time, timeframe
            )

        def place_order(
            self,
            symbol_name: str,
            side: OrderSide,
            quantity: float,
            order_type: OrderType,
            price: Optional[float] = None,
            **_kwargs,
        ) -> Optional[str]:
            return super().place_order(
                symbol_name, side, quantity, order_type, price, **_kwargs
            )

        def get_order_by_id(self, symbol_name: str, order_id: str) -> Optional[Order]:
            return super().get_order_by_id(symbol_name, order_id)

        def cancel_order(self, symbol_name: str, order_id: str) -> bool:
            return super().cancel_order(symbol_name, order_id)

    return Dummy(
        name="test_exchange", base_url="https://dummy.com", wss_url="wss://dummy.com/ws"
    )


@pytest.fixture
def dummy_is_strategy_callable() -> Type[IsStrategy]:
    class Dummy(IsStrategy, ABC):
        def __init__(self, strategy_name: Optional[str] = "test_strategy") -> None:
            super().__init__(strategy_name)

        def read_strategy_config(self) -> Dict:
            return {
                "name": "test_strategy",
                "timeframe": "1h",
                "watchlist": {"test_exchange": ["test_symbol"]},
                "parameters": {
                    "RR": 1.5,
                    "ema_period": 200,
                    "macd_signal": 9,
                    "macd_period_fast": 12,
                    "macd_period_slow": 24,
                    "psar_max_val": 0.2,
                    "psar_acceleration": 0.02,
                },
                "optimizer": {
                    "population_size": 4,
                    "num_generations": 5,
                    "objectives": ["average_pnl", "maximum_drawdown"],
                    "parameters": {
                        "RR": {"min": 1.0, "max": 5.0, "type": "float", "decimals": 1},
                        "ema_period": {"min": 100, "max": 800, "type": "int"},
                        "psar_max_val": {
                            "min": 0.05,
                            "max": 0.3,
                            "type": "float",
                            "decimals": 2,
                        },
                        "psar_acceleration": {
                            "min": 0.01,
                            "max": 0.08,
                            "type": "float",
                            "decimals": 2,
                        },
                    },
                },
            }

        def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
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

        def define_optimization_param_constraints(self, parameters: Dict) -> Dict:
            psar_acceleration = min(
                parameters["psar_acceleration"], parameters["psar_max_val"]
            )
            psar_max_val = max(
                parameters["psar_acceleration"], parameters["psar_max_val"]
            )

            parameters["psar_acceleration"] = psar_acceleration
            parameters["psar_max_val"] = psar_max_val

            return parameters

    return Dummy


@pytest.fixture
def dummy_is_strategy(dummy_is_strategy_callable) -> IsStrategy:
    return dummy_is_strategy_callable(strategy_name="test_strategy")


@pytest.fixture
def dummy_is_optimizer(
    dummy_is_exchange, dummy_is_strategy_callable
) -> StrategyOptimizer:
    return StrategyOptimizer(
        dummy_is_strategy_callable, dummy_is_exchange, None, None, None, None
    )
