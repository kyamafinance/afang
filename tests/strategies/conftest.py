from typing import List

import pytest

from afang.database.trades_db.models import Order as DBOrder
from afang.database.trades_db.models import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import TradesDatabase
from afang.exchanges.models import Order as ExchangeOrder
from afang.exchanges.models import OrderSide, OrderType


@pytest.fixture
def dummy_db_trade_positions() -> List[dict]:
    return [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "direction": 1,
            "desired_entry_price": 100,
            "open_order_id": "12345",
            "position_qty": 10,
            "position_size": 1000,
            "target_price": 150,
            "stop_price": 80,
            "initial_account_balance": 2000,
            "exchange_display_name": "test_exchange",
        },
        {
            "id": 2,
            "symbol": "BTCUSDT",
            "direction": 1,
            "desired_entry_price": 100,
            "open_order_id": "12345",
            "position_qty": 10,
            "position_size": 1000,
            "target_price": 150,
            "stop_price": 80,
            "initial_account_balance": 2000,
            "exchange_display_name": "test_exchange",
        },
        {
            "id": 3,
            "symbol": "ETHUSDT",
            "direction": 1,
            "desired_entry_price": 100,
            "open_order_id": "22222",
            "position_qty": 10,
            "position_size": 1000,
            "target_price": 150,
            "stop_price": 80,
            "initial_account_balance": 2000,
            "exchange_display_name": "test_exchange",
        },
    ]


@pytest.fixture
def dummy_db_trade_orders() -> List[dict]:
    return [
        {
            "symbol": "BTCUSDT",
            "direction": 1,
            "is_open_order": True,
            "order_id": "12345",
            "order_side": OrderSide.BUY.value,
            "original_price": 100,
            "average_price": 120,
            "original_quantity": 15,
            "executed_quantity": 10,
            "order_type": OrderType.LIMIT.value,
            "commission": 2,
            "exchange_display_name": "test_exchange",
            "position_id": 1,
        },
        {
            "symbol": "BTCUSDT",
            "direction": -1,
            "is_open_order": False,
            "order_id": "678932",
            "order_side": OrderSide.BUY.value,
            "original_price": 100,
            "average_price": 120,
            "original_quantity": 9,
            "executed_quantity": 2,
            "order_type": OrderType.LIMIT.value,
            "commission": 4,
            "exchange_display_name": "test_exchange",
            "position_id": 1,
        },
    ]


@pytest.fixture
def dummy_exchange_orders() -> List[ExchangeOrder]:
    return [
        ExchangeOrder(
            symbol="BTCUSDT",
            order_id="12345",
            side=OrderSide.BUY,
            original_price=100,
            average_price=120,
            original_quantity=15,
            executed_quantity=10,
            remaining_quantity=5,
            order_type=OrderType.LIMIT,
            order_status="PARTIAL",
            time_in_force="GTC",
            commission=2,
        ),
        ExchangeOrder(
            symbol="BTCUSDT",
            order_id="678932",
            side=OrderSide.BUY,
            original_price=100,
            average_price=120,
            original_quantity=9,
            executed_quantity=2,
            remaining_quantity=8,
            order_type=OrderType.LIMIT,
            order_status="PARTIAL",
            time_in_force="GTC",
            commission=4,
        ),
    ]


@pytest.fixture(autouse=True)
def run_around_tests(
    dummy_is_exchange,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
    trades_db_filepath,
    dummy_is_strategy,
):
    # Set-up dummy strategy class.
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.trades_database = TradesDatabase(trades_db_filepath)
    dummy_is_strategy.trades_database.database.connect()

    # Populate dummy database.
    for trade_position in dummy_db_trade_positions:
        DBTradePosition.create(**trade_position)
    for db_order in dummy_db_trade_orders:
        DBOrder.create(**db_order)

    yield

    # Tear down test DB.
    dummy_is_strategy.trades_database.database.close()
