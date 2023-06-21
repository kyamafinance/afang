import os

from playhouse.sqliteq import SqliteQueueDatabase

from afang.database.trades_db.trades_database import TradesDatabase


def test_trades_database_initialization(trades_db_filepath) -> None:
    trades_db = TradesDatabase(db_name=trades_db_filepath)
    assert os.path.exists(trades_db_filepath)
    assert type(trades_db.database) == SqliteQueueDatabase
