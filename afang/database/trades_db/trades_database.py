import logging
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

        database.init(
            database=db_name,
            pragmas={
                "journal_mode": "wal",
                "cache_size": -1 * 64000,  # 64MB
                "foreign_keys": 1,
                "ignore_check_constraints": 0,
                "synchronous": 1,
            },
        )

        self.database: SqliteDatabase = database
        with self.database:
            self.database.create_tables([TradePosition, Order])
