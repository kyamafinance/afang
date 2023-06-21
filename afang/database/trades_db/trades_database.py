import logging
import os
import pathlib
import time
from typing import Optional

from playhouse.sqliteq import SqliteQueueDatabase

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

        database.init(database=db_file_path)

        self.models = [TradePosition, Order]
        self.database: SqliteQueueDatabase = database

        # Create database tables if needed.
        self.database.start()
        self.database.connect()
        self.database.create_tables(self.models, safe=True)
        time.sleep(0.01)  # ensure tables are created in DB.
        self.database.close()
