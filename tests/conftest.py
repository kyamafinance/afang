import os
import pathlib
import shutil
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Tuple

import pytest

from afang.exchanges.is_exchange import IsExchange


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
