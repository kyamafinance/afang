from datetime import datetime

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
)

database = SqliteDatabase(database=None)


class BaseModel(Model):
    class Meta:
        database = database


class TradePosition(BaseModel):
    symbol = CharField(null=False, index=True)
    direction = IntegerField(null=False, index=True)
    desired_entry_price = FloatField(null=False)
    is_open = BooleanField(null=False, default=True, index=True)
    holding_time = IntegerField(null=False, default=0)
    open_order_id = CharField(null=False)
    position_qty = FloatField(null=False)
    position_size = FloatField(null=False)
    exchange_display_name = CharField(null=False, index=True)
    is_tp_order_active = BooleanField(null=False, default=False)
    is_sl_order_active = BooleanField(null=False, default=False)
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
    original_price = FloatField(null=False)
    original_quantity = FloatField(null=False)
    order_type = CharField(null=False)
    exchange_display_name = CharField(null=False)
    is_take_profit_order = BooleanField(null=True)
    time_in_force = CharField(null=True)
    average_price = FloatField(null=True)
    executed_quantity = FloatField(null=True)
    remaining_quantity = FloatField(null=True)
    order_status = CharField(null=True)
    commission = FloatField(null=True)
    position = ForeignKeyField(TradePosition, backref="orders")
