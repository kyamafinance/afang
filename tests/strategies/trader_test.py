from typing import List

import pytest

from afang.database.trades_db.models import Order as DBOrder
from afang.database.trades_db.models import TradePosition as DBTradePosition
from afang.database.trades_db.trades_database import (
    TradesDatabase,
    create_session_factory,
)
from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Order, OrderSide, OrderType, Symbol, SymbolBalance


@pytest.fixture
def dummy_db_trade_positions() -> List[DBTradePosition]:
    return [
        DBTradePosition(
            symbol="BTCUSDT",
            direction=1,
            desired_entry_price=10,
            open_order_id="12345",
            position_qty=10,
            position_size=10,
            target_price=20,
            stop_price=5,
            initial_account_balance=20,
        ),
        DBTradePosition(
            symbol="BTCUSDT",
            direction=1,
            desired_entry_price=10,
            open_order_id="67892",
            position_qty=10,
            position_size=10,
            target_price=20,
            stop_price=5,
            initial_account_balance=20,
        ),
    ]


@pytest.fixture
def dummy_db_trade_orders() -> List[Order]:
    return [
        DBOrder(
            symbol="BTCUSDT",
            direction=1,
            is_open_order=True,
            order_id="12345",
            order_side=OrderSide.BUY.value,
            original_price=100,
            original_quantity=10,
            order_type=OrderType.LIMIT.value,
        )
    ]


def test_fetch_symbol_open_trade_positions(
    dummy_db_trade_positions, trades_db_test_engine_url, dummy_is_strategy
) -> None:
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()
    for trade_position in dummy_db_trade_positions:
        trades_db.create_new_position(trade_position)
    trades_db.session.commit()

    open_trade_positions = dummy_is_strategy.fetch_symbol_open_trade_positions(
        "BTCUSDT", trades_db
    )

    assert len(open_trade_positions) == 2
    assert open_trade_positions == dummy_db_trade_positions


@pytest.mark.parametrize(
    "trading_execution_queue_empty, expected_symbol", [(False, "BTCUSDT"), (True, None)]
)
def test_get_next_trading_symbol(
    dummy_is_strategy, trading_execution_queue_empty, expected_symbol
) -> None:
    if not trading_execution_queue_empty:
        dummy_is_strategy.trading_execution_queue.put("BTCUSDT")

    next_trading_symbol = dummy_is_strategy.get_next_trading_symbol(run_forever=False)
    assert next_trading_symbol == expected_symbol


@pytest.mark.parametrize(
    "trading_symbols, expected_symbol",
    [
        (
            {
                "BTCUSDT": Symbol(
                    name="BTCUSDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    price_decimals=3,
                    quantity_decimals=3,
                    tick_size=3,
                    step_size=3,
                )
            },
            "BTCUSDT",
        ),
        (dict(), None),
    ],
)
def test_get_trading_symbol(
    dummy_is_strategy, dummy_is_exchange, trading_symbols, expected_symbol, caplog
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.exchange.trading_symbols = trading_symbols
    trading_symbol = dummy_is_strategy.get_trading_symbol("BTCUSDT")

    if not expected_symbol:
        assert trading_symbol == expected_symbol
        assert caplog.records[0].levelname == "ERROR"
        assert "symbol not found in exchange trading symbols" in caplog.text
    else:
        assert trading_symbol == trading_symbols["BTCUSDT"]


@pytest.mark.parametrize(
    "trading_symbols, trading_symbol_balance, expected_balance, found_quote_balance",
    [
        (
            {
                "BTCUSDT": Symbol(
                    name="BTCUSDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    price_decimals=3,
                    quantity_decimals=3,
                    tick_size=3,
                    step_size=3,
                )
            },
            {"USDT": SymbolBalance(name="USDT", wallet_balance=100)},
            SymbolBalance(name="USDT", wallet_balance=100),
            True,
        ),
        (
            {
                "BTCUSDT": Symbol(
                    name="BTCUSDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    price_decimals=3,
                    quantity_decimals=3,
                    tick_size=3,
                    step_size=3,
                )
            },
            dict(),
            None,
            False,
        ),
        (dict(), SymbolBalance(name="USDT", wallet_balance=100), None, True),
    ],
)
def test_get_quote_asset_balance(
    dummy_is_strategy,
    dummy_is_exchange,
    trading_symbols,
    trading_symbol_balance,
    expected_balance,
    found_quote_balance,
    caplog,
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    dummy_is_strategy.exchange.trading_symbols = trading_symbols
    dummy_is_strategy.exchange.trading_symbol_balance = trading_symbol_balance

    quote_asset_balance = dummy_is_strategy.get_quote_asset_balance("BTCUSDT")
    assert quote_asset_balance == expected_balance
    if not found_quote_balance:
        assert caplog.records[0].levelname == "ERROR"
        assert (
            "quote asset USDT not found in exchange trading symbol balances"
            in caplog.text
        )


@pytest.mark.parametrize(
    "quote_asset_balance, intended_position_size",
    [
        (SymbolBalance(name="USDT", wallet_balance=1000), 100),
        (SymbolBalance(name="USDT", wallet_balance=2000), 150),
    ],
)
def test_get_open_order_position_size(
    dummy_is_strategy, quote_asset_balance, intended_position_size
) -> None:
    dummy_is_strategy.leverage = 5
    dummy_is_strategy.max_amount_per_trade = 150

    position_size = dummy_is_strategy.get_open_order_position_size(quote_asset_balance)
    assert position_size == intended_position_size


@pytest.mark.parametrize(
    "exchange_order_return_vals, expected_output, expected_log",
    [
        ([None, None], 0.0, True),
        (
            [
                Order(
                    symbol="BTCUSDT",
                    order_id="12345",
                    side=OrderSide.BUY,
                    original_price=100,
                    average_price=100,
                    original_quantity=10,
                    executed_quantity=6,
                    remaining_quantity=4,
                    order_type=OrderType.LIMIT,
                    order_status="PARTIAL",
                    time_in_force="GTC",
                    commission=1,
                ),
                None,
            ],
            6,
            False,
        ),
        (
            [
                Order(
                    symbol="BTCUSDT",
                    order_id="12345",
                    side=OrderSide.BUY,
                    original_price=100,
                    average_price=100,
                    original_quantity=10,
                    executed_quantity=6,
                    remaining_quantity=4,
                    order_type=OrderType.LIMIT,
                    order_status="PARTIAL",
                    time_in_force="GTC",
                    commission=1,
                ),
                Order(
                    symbol="BTCUSDT",
                    order_id="67892",
                    side=OrderSide.BUY,
                    original_price=100,
                    average_price=100,
                    original_quantity=20,
                    executed_quantity=12,
                    remaining_quantity=8,
                    order_type=OrderType.LIMIT,
                    order_status="PARTIAL",
                    time_in_force="GTC",
                    commission=1,
                ),
                None,
            ],
            8,
            False,
        ),
    ],
)
def test_get_close_order_qty(
    mocker,
    dummy_is_strategy,
    dummy_is_exchange,
    dummy_db_trade_positions,
    exchange_order_return_vals,
    expected_output,
    expected_log,
    caplog,
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    mocked_get_exchange_order = mocker.patch.object(
        IsExchange, "get_exchange_order", side_effect=exchange_order_return_vals
    )

    close_order_qty = dummy_is_strategy.get_close_order_qty(dummy_db_trade_positions[0])
    assert close_order_qty == expected_output
    assert mocked_get_exchange_order.assert_called

    if expected_log:
        assert caplog.records[0].levelname == "ERROR"
        assert "could not find position open order" in caplog.text


@pytest.mark.parametrize(
    "order_qty, expected_result", [(0, False), (-10, False), (10, True)]
)
def test_is_order_qty_valid(
    dummy_is_strategy, dummy_is_exchange, order_qty, expected_result, caplog
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    symbol = Symbol(
        name="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_decimals=3,
        quantity_decimals=3,
        tick_size=3,
        step_size=3,
    )

    is_order_qty_valid = dummy_is_strategy.is_order_qty_valid(symbol, order_qty)
    assert is_order_qty_valid == expected_result
    if not expected_result:
        assert caplog.records[0].levelname == "ERROR"
        assert "intended order qty is invalid" in caplog.text


@pytest.mark.parametrize(
    "order_price, expected_result", [(0, False), (-10, False), (10, True)]
)
def test_is_order_price_valid(
    dummy_is_strategy, dummy_is_exchange, order_price, expected_result, caplog
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    symbol = Symbol(
        name="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_decimals=3,
        quantity_decimals=3,
        tick_size=3,
        step_size=3,
    )

    is_order_price_valid = dummy_is_strategy.is_order_price_valid(symbol, order_price)
    assert is_order_price_valid == expected_result
    if not expected_result:
        assert caplog.records[0].levelname == "ERROR"
        assert "intended order price is invalid" in caplog.text


@pytest.mark.parametrize(
    "create_new_position_return_val, should_fail", [(Exception, True), (None, False)]
)
def test_create_new_db_position(
    mocker,
    dummy_is_strategy,
    dummy_is_exchange,
    dummy_db_trade_positions,
    trades_db_test_engine_url,
    create_new_position_return_val,
    should_fail,
    caplog,
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()

    if should_fail:
        mocker.patch.object(
            TradesDatabase,
            "create_new_position",
            side_effect=create_new_position_return_val,
        )

    dummy_is_strategy.create_new_db_position(dummy_db_trade_positions[0], trades_db)

    if should_fail:
        assert caplog.records[0].levelname == "ERROR"
        assert "failed to record new trade position to the DB" in caplog.text
    else:
        assert trades_db.fetch_position_by_id(1) == dummy_db_trade_positions[0]


@pytest.mark.parametrize(
    "create_new_order_return_val, should_fail", [(Exception, True), (None, False)]
)
def test_create_new_db_order(
    mocker,
    dummy_is_strategy,
    dummy_is_exchange,
    dummy_db_trade_orders,
    trades_db_test_engine_url,
    create_new_order_return_val,
    should_fail,
    caplog,
) -> None:
    dummy_is_strategy.exchange = dummy_is_exchange
    create_session_factory(engine_url=trades_db_test_engine_url)
    trades_db = TradesDatabase()

    if should_fail:
        mocker.patch.object(
            TradesDatabase,
            "create_new_order",
            side_effect=create_new_order_return_val,
        )

    dummy_is_strategy.create_new_db_order(dummy_db_trade_orders[0], trades_db)

    if should_fail:
        assert caplog.records[0].levelname == "ERROR"
        assert "failed to record new order to the DB" in caplog.text
    else:
        assert trades_db.fetch_order_by_id(1) == dummy_db_trade_orders[0]
