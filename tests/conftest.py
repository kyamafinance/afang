import os
import pathlib
import shutil
from collections.abc import Generator

import pytest


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
