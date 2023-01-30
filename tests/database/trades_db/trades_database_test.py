import os

import pytest
from sqlalchemy.orm import scoped_session

from afang.database.trades_db.models import Order, TradePosition
from afang.database.trades_db.trades_database import (
    TradesDatabase,
    create_session_factory,
)


@pytest.fixture
def dummy_trade_position() -> TradePosition:
    return TradePosition(
        symbol="BTCUSDT",
        direction=1,
        desired_entry_price=10,
        open_order_id="12345",
        position_qty=10,
        position_size=10,
        target_price=20,
        stop_price=5,
        initial_account_balance=20,
    )


@pytest.fixture
def dummy_trade_order() -> Order:
    return Order(
        symbol="BTCUSDT",
        direction=1,
        is_open_order=True,
        order_id="12345",
        order_side="BUY",
        original_price=23,
        original_quantity=10,
        order_type="MARKET",
    )


def test_session_factory_initialization(
    trades_db_filepath, trades_db_test_engine_url, caplog
) -> None:
    TradesDatabase()
    assert caplog.records[0].levelname == "ERROR"
    assert "TradesDatabase requires an initialized session factory" in caplog.text

    # initialize a session factory and ensure that the trades_db is created.
    session_factory = create_session_factory(engine_url=trades_db_test_engine_url)
    assert type(session_factory) is scoped_session
    assert os.path.exists(trades_db_filepath)

    trades_db = TradesDatabase()
    assert trades_db.session is not None


def test_create_new_position(trades_db_test_engine_url, dummy_trade_position) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_position(dummy_trade_position)
    persisted_trade_position = trades_db.fetch_position_by_id(1)
    assert persisted_trade_position == dummy_trade_position


def test_delete_position(trades_db_test_engine_url, dummy_trade_position) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_position(dummy_trade_position)
    trades_db.delete_position(1)
    persisted_trade_position = trades_db.fetch_position_by_id(1)
    assert persisted_trade_position is None


def test_update_position(trades_db_test_engine_url, dummy_trade_position) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_position(dummy_trade_position)
    trades_db.update_position(1, {"symbol": "ETHUSDT", "open_order_id": "6789"})
    persisted_trade_position = trades_db.fetch_position_by_id(1)
    assert persisted_trade_position.symbol == "ETHUSDT"
    assert persisted_trade_position.open_order_id == "6789"


def test_fetch_position_by_id(trades_db_test_engine_url, dummy_trade_position) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_position(dummy_trade_position)
    persisted_trade_position = trades_db.fetch_position_by_id(1)
    assert persisted_trade_position is dummy_trade_position


@pytest.mark.parametrize(
    "filters, limit, expected_persisted_positions",
    [
        (tuple(), -1, 2),
        ((TradePosition.symbol == "LINKUSDT",), -1, 1),
        (tuple(), 1, 1),
        (tuple(), 20, 2),
    ],
)
def test_fetch_positions(
    trades_db_test_engine_url,
    dummy_trade_position,
    filters,
    limit,
    expected_persisted_positions,
) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_position(dummy_trade_position)
    second_dummy_trade_position = TradePosition(
        symbol="LINKUSDT",
        direction=1,
        desired_entry_price=10,
        open_order_id="12345",
        position_qty=10,
        position_size=10,
        target_price=20,
        stop_price=5,
        initial_account_balance=20,
    )
    trades_db.create_new_position(second_dummy_trade_position)
    persisted_positions = trades_db.fetch_positions(filters, limit)

    assert len(persisted_positions) == expected_persisted_positions


def test_create_new_order(trades_db_test_engine_url, dummy_trade_order) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_order(dummy_trade_order)
    persisted_order = trades_db.fetch_order_by_id(1)
    assert persisted_order == dummy_trade_order


def test_update_order(trades_db_test_engine_url, dummy_trade_order) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_order(dummy_trade_order)
    trades_db.update_order(
        1,
        {
            "symbol": "ETHUSDT",
            "is_open_order": False,
        },
    )
    persisted_order = trades_db.fetch_order_by_id(1)
    assert persisted_order.symbol == "ETHUSDT"
    assert not persisted_order.is_open_order


def test_fetch_order_by_id(trades_db_test_engine_url, dummy_trade_order) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_order(dummy_trade_order)
    persisted_order = trades_db.fetch_order_by_id(1)
    assert persisted_order is dummy_trade_order


def test_fetch_order_by_exchange_id(
    trades_db_test_engine_url, dummy_trade_order
) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_order(dummy_trade_order)
    persisted_order = trades_db.fetch_order_by_exchange_id("12345")
    assert persisted_order is dummy_trade_order


@pytest.mark.parametrize(
    "filters, limit, expected_persisted_orders",
    [
        (tuple(), -1, 2),
        ((Order.symbol == "LINKUSDT",), -1, 1),
        (tuple(), 1, 1),
        (tuple(), 20, 2),
    ],
)
def test_fetch_orders(
    trades_db_test_engine_url,
    dummy_trade_order,
    filters,
    limit,
    expected_persisted_orders,
) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    trades_db.create_new_order(dummy_trade_order)
    second_dummy_trade_order = Order(
        symbol="LINKUSDT",
        direction=1,
        is_open_order=True,
        order_id="12345",
        order_side="BUY",
        original_price=23,
        original_quantity=10,
        order_type="MARKET",
    )
    trades_db.create_new_order(second_dummy_trade_order)
    persisted_orders = trades_db.fetch_orders(filters, limit)

    assert len(persisted_orders) == expected_persisted_orders
