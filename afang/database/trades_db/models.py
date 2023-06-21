from datetime import datetime

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
)
from playhouse.sqliteq import SqliteQueueDatabase

database = SqliteQueueDatabase(
    database=None,
    use_gevent=False,
    autostart=False,
    queue_max_size=64,
    results_timeout=5.0,
    thread_safe=True,
    pragmas={
        "journal_mode": "wal",
        "cache_size": -1 * 64000,  # 64MB
        "foreign_keys": 1,
        "ignore_check_constraints": 0,
        "synchronous": 1,
    },
)


class BaseModel(Model):
    class Meta:
        database = database


class TradePosition(BaseModel):
    sequence_id = CharField(null=False, index=True)
    symbol = CharField(null=False, index=True)
    direction = IntegerField(null=False, index=True)
    desired_entry_price = FloatField(null=False)
    is_open = BooleanField(null=False, default=True, index=True)
    holding_time = IntegerField(null=False, default=0)
    open_order_id = CharField(null=False)
    position_qty = FloatField(null=False)
    position_size = FloatField(null=False)
    exchange_display_name = CharField(null=False, index=True)
    entry_time = DateTimeField(null=True)
    entry_price = FloatField(null=True)
    initial_account_balance = FloatField(null=True)
    target_price = FloatField(null=True)
    stop_price = FloatField(null=True)
    close_price = FloatField(null=True)
    executed_qty = FloatField(null=True)
    exit_time = DateTimeField(null=True)
    roe = FloatField(null=True)
    cost_adjusted_roe = FloatField(null=True)
    pnl = FloatField(null=True)
    commission = FloatField(null=True)
    slippage = FloatField(null=True)
    final_account_balance = FloatField(null=True)
    created_at = DateTimeField(null=False, default=datetime.utcnow())


class Order(BaseModel):
    symbol = CharField(null=False, index=True)
    direction = IntegerField(null=False, index=True)
    is_open = BooleanField(null=False, default=True, index=True)
    is_open_order = BooleanField(null=False, index=True)
    order_id = CharField(null=False)
    order_side = CharField(null=False)
    raw_price = FloatField(null=False)
    original_price = FloatField(null=False)
    original_quantity = FloatField(null=False)
    order_type = CharField(null=False)
    exchange_display_name = CharField(null=False)
    time_in_force = CharField(null=True)
    average_price = FloatField(null=True)
    executed_quantity = FloatField(null=True)
    remaining_quantity = FloatField(null=True)
    order_status = CharField(null=True)
    commission = FloatField(null=True)
    position = ForeignKeyField(TradePosition, backref="orders")
