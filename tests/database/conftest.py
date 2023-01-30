import os
import pathlib
from collections.abc import Generator

import pytest


@pytest.fixture()
def trades_db_filepath() -> str:
    filepath = (
        f"{pathlib.Path(__file__).parents[1]}/testdata/database/trades/trades.sqlite3"
    )
    return filepath


@pytest.fixture()
def trades_db_test_engine_url(trades_db_filepath) -> str:
    return f"sqlite:///{trades_db_filepath}"


@pytest.fixture(autouse=True)
def delete_trades_database(trades_db_filepath) -> Generator:
    yield
    if os.path.exists(trades_db_filepath):
        os.unlink(trades_db_filepath)
