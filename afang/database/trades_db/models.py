from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from afang.database.trades_db.base import Base


class TradePosition(Base):
    __tablename__ = "position"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    direction = Column(Integer, nullable=False)
    desired_entry_price = Column(Float, nullable=False)
    open_position = Column(Boolean, nullable=False, default=True)
    holding_time = Column(Integer, nullable=False, default=0)
    open_order_id = Column(String, nullable=False)
    position_qty = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    exchange_display_name = Column(String, nullable=False)
    take_profit_order_active = Column(Boolean, nullable=False, default=False)
    stop_loss_order_active = Column(Boolean, nullable=False, default=False)
    entry_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=True)
    initial_account_balance = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    executed_qty = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    roe = Column(Float, nullable=True)
    cost_adjusted_roe = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    commission = Column(Float, nullable=True)
    slippage = Column(Float, nullable=True)
    final_account_balance = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow())
    orders = relationship("Order", back_populates="position", cascade="all, delete")


class Order(Base):
    __tablename__ = "order"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    direction = Column(Integer, nullable=False)
    is_open_order = Column(Boolean, nullable=False)
    order_id = Column(String, nullable=False)
    order_side = Column(String, nullable=False)
    original_price = Column(Float, nullable=False)
    original_quantity = Column(Float, nullable=False)
    order_type = Column(String, nullable=False)
    exchange_display_name = Column(String, nullable=False)
    is_take_profit_order = Column(Boolean, nullable=True)
    time_in_force = Column(String, nullable=True)
    average_price = Column(Float, nullable=True)
    executed_quantity = Column(Float, nullable=True)
    remaining_quantity = Column(Float, nullable=True)
    order_status = Column(String, nullable=True)
    commission = Column(Float, nullable=True)
    complete = Column(Boolean, default=False)
    trade_position_id = Column(Integer, ForeignKey("position.id"))
    position = relationship("TradePosition", back_populates="orders")
