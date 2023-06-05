import logging
import os
import pathlib
from typing import Optional

from peewee import SqliteDatabase

from afang.database.trades_db.models import Order, TradePosition, database

logger = logging.getLogger(__name__)


class TradesDatabase:
    """Interface to store, retrieve, and manipulate user demo/live trade
    data."""

    def __init__(self, db_name: Optional[str] = "trades.sqlite3") -> None:
        """Initialize the TradesDatabase class.

        :param db_name: database name/filepath. optional.
        """

        db_base_dir = os.path.join(pathlib.Path(__file__).parents[3], "data", "trades")
        db_file_path = os.path.join(db_base_dir, db_name)

        database.init(
            database=db_file_path,
            pragmas={
                "journal_mode": "wal",
                "cache_size": -1 * 64000,  # 64MB
                "foreign_keys": 1,
                "ignore_check_constraints": 0,
                "synchronous": 1,
            },
        )

        self.models = [TradePosition, Order]
        self.database: SqliteDatabase = database
        with self.database:
            self.database.create_tables(self.models, safe=True)
