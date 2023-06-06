import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from afang.database.trades_db.models import Order as DBOrder
from afang.database.trades_db.models import TradePosition as DBTradePosition
from afang.exchanges.is_exchange import IsExchange
from afang.exchanges.models import Candle, OrderSide, OrderType, Symbol, SymbolBalance
from afang.models import Timeframe
from afang.strategies.is_strategy import IsStrategy
from afang.strategies.models import TradeLevels
from afang.strategies.trader import Trader


@pytest.mark.parametrize(
    "symbols, expected_count",
    [(["ETHUSDT"], 1), (["BTCUSDT"], 2), (["ETHUSDT", "BTCUSDT"], 3)],
)
def test_fetch_symbol_open_trade_positions(
    dummy_db_trade_positions,
    dummy_is_strategy,
    symbols,
    expected_count,
) -> None:
    open_trade_positions = dummy_is_strategy.fetch_open_trade_positions(symbols)
    assert len(open_trade_positions) == expected_count


@pytest.mark.parametrize("order_id, expected_found", [("12345", True), ("6789", False)])
def test_fetch_order_by_exchange_id(
    dummy_db_trade_positions,
    dummy_db_trade_orders,
    dummy_is_strategy,
    order_id,
    expected_found,
    caplog,
) -> None:
    order = dummy_is_strategy.fetch_order_by_exchange_id(order_id)

    if expected_found:
        assert order.order_id == order_id
        return
    assert caplog.records[0].levelname == "WARNING"
    assert "order not found in DB" in caplog.text


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
    dummy_is_strategy, trading_symbols, expected_symbol, caplog
) -> None:
    dummy_is_strategy.exchange.trading_symbols = trading_symbols
    trading_symbol = dummy_is_strategy.get_trading_symbol("BTCUSDT")

    if not expected_symbol:
        assert trading_symbol == expected_symbol
        assert caplog.records[0].levelname == "ERROR"
        assert "symbol not found in exchange trading symbols" in caplog.text
    else:
        assert trading_symbol == trading_symbols["BTCUSDT"]


@pytest.mark.parametrize(
    "on_demo_mode, trading_symbols, trading_symbol_balance, expected_balance, found_quote_balance",
    [
        (
            True,
            None,
            None,
            1000,
            True,
        ),
        (
            False,
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
            100,
            True,
        ),
        (
            False,
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
        (False, dict(), SymbolBalance(name="USDT", wallet_balance=100), None, True),
    ],
)
def test_get_quote_asset_wallet_balance(
    dummy_is_strategy,
    on_demo_mode,
    trading_symbols,
    trading_symbol_balance,
    expected_balance,
    found_quote_balance,
    caplog,
) -> None:
    dummy_is_strategy.exchange.trading_symbols = trading_symbols
    dummy_is_strategy.exchange.trading_symbol_balance = trading_symbol_balance
    dummy_is_strategy.on_demo_mode = on_demo_mode
    dummy_is_strategy.initial_test_account_balance = 1000

    quote_asset_balance = dummy_is_strategy.get_quote_asset_wallet_balance("BTCUSDT")
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
        (1000, 100),
        (2000, 150),
    ],
)
def test_get_open_order_position_size(
    dummy_is_strategy, quote_asset_balance, intended_position_size
) -> None:
    dummy_is_strategy.leverage = 5
    dummy_is_strategy.max_amount_per_trade = 150

    position_size = dummy_is_strategy.get_open_order_position_size(quote_asset_balance)
    assert position_size == intended_position_size


def test_get_close_order_qty(
    mocker,
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
    dummy_exchange_orders,
    caplog,
) -> None:
    mocked_get_exchange_order = mocker.patch.object(
        IsExchange, "get_exchange_order", side_effect=dummy_exchange_orders
    )

    db_position = DBTradePosition.get(DBTradePosition.id == 1)
    close_order_qty = dummy_is_strategy.get_close_order_qty(db_position)
    assert close_order_qty == 8
    assert mocked_get_exchange_order.assert_called


@pytest.mark.parametrize(
    "order_qty, expected_result", [(0, False), (-10, False), (10, True)]
)
def test_is_order_qty_valid(
    dummy_is_strategy, order_qty, expected_result, caplog
) -> None:
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
    dummy_is_strategy, order_price, expected_result, caplog
) -> None:
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


def test_update_closed_order_in_db(
    dummy_is_strategy,
    dummy_exchange_orders,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    db_order = DBOrder.get(DBOrder.id == 1)
    dummy_is_strategy.update_closed_order_in_db(dummy_exchange_orders[0], db_order)
    updated_db_order = DBOrder.get(DBOrder.id == 1)

    assert db_order.is_open
    assert not updated_db_order.is_open


@pytest.mark.parametrize(
    "db_position_order_found, db_position_order_closed, exchange_position_order_found, has_remaining_qty",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, False, False, False),
        (True, False, True, True),
    ],
)
def test_cancel_position_order(
    mocker,
    dummy_is_strategy,
    dummy_db_trade_orders,
    dummy_exchange_orders,
    db_position_order_found,
    db_position_order_closed,
    exchange_position_order_found,
    has_remaining_qty,
    caplog,
) -> None:
    mocked_exchange_cancel_order = mocker.patch.object(IsExchange, "cancel_order")
    mocked_closed_order_update = mocker.patch.object(
        Trader, "update_closed_order_in_db"
    )

    if db_position_order_found:
        dummy_position_order = DBOrder.get(DBOrder.id == 1)
        if db_position_order_closed:
            dummy_position_order.is_open = False
            dummy_position_order.save()
        mocker.patch.object(
            Trader,
            "fetch_order_by_exchange_id",
            return_value=dummy_position_order,
        )
    else:
        mocker.patch.object(Trader, "fetch_order_by_exchange_id", return_value=None)

    if exchange_position_order_found:
        dummy_exchange_order = dummy_exchange_orders[0]
        if not has_remaining_qty:
            dummy_exchange_order.remaining_quantity = 0
        mocker.patch.object(
            IsExchange, "get_exchange_order", return_value=dummy_exchange_order
        )
    else:
        mocker.patch.object(IsExchange, "get_exchange_order", return_value=None)

    dummy_is_strategy.cancel_position_order("BTCUSDT", "12345")

    if not db_position_order_found:
        assert caplog.records[0].levelname == "ERROR"
        assert (
            "position order not cancelled because it was not found in the DB."
            in caplog.text
        )
        return
    if not exchange_position_order_found and db_position_order_closed:
        assert not caplog.records
        return
    if not exchange_position_order_found:
        assert caplog.records[0].levelname == "ERROR"
        assert (
            "position order not canceled because order was not found on the exchange."
            in caplog.text
        )
        return
    if exchange_position_order_found:
        assert mocked_closed_order_update.assert_called
    if has_remaining_qty:
        assert mocked_exchange_cancel_order.assert_called
        assert caplog.records[0].levelname == "INFO"
        assert (
            "attempting to cancel position order due to un-executed qty." in caplog.text
        )
        return


def test_is_order_filled(mocker, dummy_is_strategy, dummy_exchange_orders) -> None:
    dummy_order = dummy_exchange_orders[0]
    dummy_order.executed_quantity = 0
    mocker.patch.object(IsExchange, "get_exchange_order", return_value=dummy_order)
    is_order_filled = dummy_is_strategy.is_order_filled("BTCUSDT", "12345")
    assert not is_order_filled

    dummy_order.executed_quantity = 5
    mocker.patch.object(IsExchange, "get_exchange_order", return_value=dummy_order)
    is_order_filled = dummy_is_strategy.is_order_filled("BTCUSDT", "12345")
    assert is_order_filled


@pytest.mark.parametrize(
    "order_found, expected_return_val", [(False, 0.0), (True, 120)]
)
def test_get_order_average_price(
    mocker,
    dummy_is_strategy,
    dummy_exchange_orders,
    order_found,
    expected_return_val,
    caplog,
) -> None:
    if order_found:
        dummy_order = dummy_exchange_orders[0]
        mocker.patch.object(IsExchange, "get_exchange_order", return_value=dummy_order)
    else:
        mocker.patch.object(IsExchange, "get_exchange_order", return_value=None)

    average_price = dummy_is_strategy.get_order_average_price("BTCUSDT", "12345")

    assert average_price == expected_return_val
    if not order_found:
        assert caplog.records[0].levelname == "WARNING"
        assert (
            "Could not get the order average price for BTCUSDT order: 12345"
            in caplog.text
        )


def test_get_symbol_ohlcv_candles_df(dummy_is_strategy, caplog) -> None:
    dummy_is_strategy.timeframe = Timeframe("1m")

    ohlcv_candles = dummy_is_strategy.get_symbol_ohlcv_candles_df("BTCUSDT")
    assert ohlcv_candles.empty
    assert caplog.records[0].levelname == "ERROR"
    assert "price data unavailable" in caplog.text

    dummy_is_strategy.exchange.trading_price_data = {
        "BTCUSDT": [
            Candle(
                open_time=1662774258000,
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
            ),
            Candle(
                open_time=1662774318000,
                open=6,
                high=7,
                low=8,
                close=9,
                volume=10,
            ),
        ]
    }

    ohlcv_data = {
        "open": [1, 6],
        "high": [2, 7],
        "low": [3, 8],
        "close": [4, 9],
        "volume": [5, 10],
    }
    ohlcv_data_df = pd.DataFrame(
        ohlcv_data,
        index=[
            datetime.datetime(2022, 9, 10, 1, 44, 18),
            datetime.datetime(2022, 9, 10, 1, 45, 18),
        ],
    )
    ohlcv_data_df.index.name = "open_time"
    ohlcv_candles = dummy_is_strategy.get_symbol_ohlcv_candles_df("BTCUSDT")
    assert ohlcv_candles.equals(ohlcv_data_df)


def test_get_symbol_current_trading_candle(dummy_is_strategy, caplog) -> None:
    dummy_is_strategy.timeframe = Timeframe("1m")

    current_trading_candle = dummy_is_strategy.get_symbol_current_trading_candle(
        "BTCUSDT", pd.DataFrame()
    )
    assert not current_trading_candle
    assert caplog.records[0].levelname == "ERROR"
    assert "could not fetch current candle data for BTCUSDT 1m" in caplog.text

    ohlcv_data = {
        "open": [1, 6],
        "high": [2, 7],
        "low": [3, 8],
        "close": [4, 9],
        "volume": [5, 10],
    }
    ohlcv_data_df = pd.DataFrame(
        ohlcv_data,
        index=[
            datetime.datetime(2022, 9, 10, 1, 44, 18),
            datetime.datetime(2022, 9, 10, 1, 45, 18),
        ],
    )
    ohlcv_data_df.index.name = "open_time"

    current_trading_candle = dummy_is_strategy.get_symbol_current_trading_candle(
        "BTCUSDT", ohlcv_data_df
    )
    assert current_trading_candle.open == 6
    assert current_trading_candle.high == 7
    assert current_trading_candle.low == 8
    assert current_trading_candle.close == 9
    assert current_trading_candle.volume == 10
    assert current_trading_candle.Index == datetime.datetime(2022, 9, 10, 1, 45, 18)


def test_get_position_total_commission(
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    total_commission = dummy_is_strategy.get_position_total_commission(
        DBTradePosition.get(DBTradePosition.id == 1)
    )
    assert total_commission == 6


def test_get_position_total_slippage(
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    total_slippage = dummy_is_strategy.get_position_total_slippage(
        DBTradePosition.get(DBTradePosition.id == 1)
    )
    assert total_slippage == 160.0


def test_get_position_executed_qty(
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    executed_qty = dummy_is_strategy.get_position_executed_qty(
        DBTradePosition.get(DBTradePosition.id == 1)
    )
    assert executed_qty == 2.0


def test_get_position_close_price(
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    position_close_price = dummy_is_strategy.get_position_close_price(
        DBTradePosition.get(DBTradePosition.id == 1)
    )
    assert position_close_price == 120.0


def test_get_position_pnl(
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    pnl = dummy_is_strategy.get_position_pnl(
        DBTradePosition.get(DBTradePosition.id == 1)
    )
    assert pnl == -966.0


@pytest.mark.parametrize(
    "cost_adjusted, has_open_position_order, expected_roe",
    [
        (False, False, 0.0),
        (
            True,
            True,
            -80.5,
        ),
        (
            False,
            True,
            -80.0,
        ),
    ],
)
def test_get_position_roe(
    mocker,
    dummy_is_strategy,
    dummy_exchange_orders,
    cost_adjusted,
    has_open_position_order,
    expected_roe,
    caplog,
) -> None:
    open_position_order = dummy_exchange_orders[0] if has_open_position_order else None

    mocker.patch.object(
        Trader, "fetch_order_by_exchange_id", return_value=open_position_order
    )

    db_position = DBTradePosition.get(DBTradePosition.id == 1)
    roe = dummy_is_strategy.get_position_roe(db_position, cost_adjusted=cost_adjusted)

    if not has_open_position_order:
        assert caplog.records[0].levelname == "ERROR"
        assert (
            "could not calculate position ROE because the open order could not be found"
            in caplog.text
        )
    assert roe == expected_roe


def test_close_trade_position(
    mocker,
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_db_trade_orders,
) -> None:
    mocker.patch.object(Trader, "get_position_pnl")
    mocker.patch.object(Trader, "get_position_total_slippage")
    mocker.patch.object(Trader, "get_position_close_price")
    mocker.patch.object(Trader, "get_position_roe")
    mocker.patch.object(Trader, "get_position_executed_qty")
    mocker.patch.object(Trader, "get_position_total_commission")

    dummy_is_strategy.exchange.trading_symbols = {
        "BTCUSDT": Symbol(
            name="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_decimals=3,
            quantity_decimals=3,
            tick_size=3,
            step_size=3,
        )
    }
    dummy_is_strategy.exchange.trading_symbol_balance = {
        "USDT": SymbolBalance(name="USDT", wallet_balance=100)
    }

    assert (
        DBTradePosition.select().where(DBTradePosition.is_open.__eq__(True)).count()
        == 3
    )

    db_position = DBTradePosition.get(DBTradePosition.id == 1)
    dummy_is_strategy.close_trade_position(db_position)

    assert (
        DBTradePosition.select().where(DBTradePosition.is_open.__eq__(False)).count()
        == 1
    )
    assert (
        DBTradePosition.select().where(DBTradePosition.is_open.__eq__(True)).count()
        == 2
    )


@pytest.mark.parametrize(
    "on_demo_mode, order_found", [(True, True), (False, True), (False, False)]
)
def test_get_exchange_order(
    mocker, dummy_is_strategy, dummy_exchange_orders, on_demo_mode, order_found, caplog
) -> None:
    mocked_get_exchange_order = mocker.patch.object(IsExchange, "get_exchange_order")
    dummy_is_strategy.demo_mode_exchange_orders = {
        "BTCUSDT": [dummy_exchange_orders[0]]
    }

    if on_demo_mode:
        dummy_is_strategy.on_demo_mode = True
    else:
        dummy_is_strategy.on_demo_mode = False

    if order_found:
        order_id = "12345"
    else:
        order_id = "56789"

    exchange_order = dummy_is_strategy.get_exchange_order("BTCUSDT", order_id)

    if on_demo_mode and order_found:
        assert exchange_order == dummy_exchange_orders[0]
    elif on_demo_mode and not order_found:
        assert not exchange_order
        assert caplog.records[0].levelname == "WARNING"
        assert "Unable to get test_exchange BTCUSDT demo order" in caplog.text
    elif not on_demo_mode:
        assert mocked_get_exchange_order.called


@pytest.mark.parametrize("exchange_place_order_return_val", ["1111", None])
def test_open_trade_position(
    mocker,
    dummy_is_strategy,
    caplog,
    exchange_place_order_return_val,
) -> None:
    dummy_is_strategy.exchange.trading_symbols = {
        "BTCUSDT": Symbol(
            name="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_decimals=3,
            quantity_decimals=3,
            tick_size=0.001,
            step_size=0.001,
        )
    }
    dummy_is_strategy.exchange.trading_symbol_balance = {
        "USDT": SymbolBalance(name="USDT", wallet_balance=100)
    }

    mocked_place_order = mocker.patch.object(
        IsExchange, "place_order", return_value=exchange_place_order_return_val
    )

    trade_position = dummy_is_strategy.open_trade_position(
        "BTCUSDT",
        1,
        TradeLevels(entry_price=10, target_price=12, stop_price=5),
    )

    assert mocked_place_order.assert_called
    assert caplog.records[0].levelname == "INFO"
    assert "attempting to open a new trade position" in caplog.text
    if not exchange_place_order_return_val:
        assert caplog.records[1].levelname == "ERROR"
        assert "failed to place open order on the exchange" in caplog.text
        return
    assert caplog.records[1].levelname == "INFO"
    assert "open order successfully placed on the exchange" in caplog.text
    assert trade_position == DBTradePosition.get(DBTradePosition.id == 4)
    assert len(trade_position.orders) == 1


@pytest.mark.parametrize("is_position_open", [True, False])
def test_place_close_trade_position_order(
    mocker,
    dummy_is_strategy,
    is_position_open,
    caplog,
) -> None:
    dummy_is_strategy.exchange.trading_symbols = {
        "BTCUSDT": Symbol(
            name="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_decimals=3,
            quantity_decimals=3,
            tick_size=0.003,
            step_size=0.003,
        )
    }
    dummy_is_strategy.exchange.trading_symbol_balance = {
        "USDT": SymbolBalance(name="USDT", wallet_balance=100)
    }

    mocker.patch.object(Trader, "is_order_filled", return_value=True)
    mocker.patch.object(IsExchange, "place_order", return_value="11111")
    mocker.patch.object(Trader, "get_close_order_qty", return_value=20)

    db_position = DBTradePosition.get(DBTradePosition.id == 1)
    if not is_position_open:
        db_position.is_open = False
        db_position.save()
    dummy_is_strategy.place_close_trade_position_order(db_position, 20, True)
    db_position = DBTradePosition.get(DBTradePosition.id == 1)

    assert caplog.records[0].levelname == "INFO"
    assert "attempting to place a close trade position order" in caplog.text

    if not is_position_open:
        assert caplog.records[-1].levelname == "ERROR"
        assert "trade position is already closed" in caplog.text
        return

    assert caplog.records[-1].levelname == "INFO"
    assert "close order successfully placed on the exchange" in caplog.text
    assert db_position.is_tp_order_active is True


@pytest.mark.parametrize(
    "direction, close_order_type, current_candle_data",
    [
        (1, OrderType.MARKET, {"close": 22}),
        (1, OrderType.LIMIT, {"close": 12}),
        (1, OrderType.MARKET, {"close": 4.5}),
        (-1, OrderType.MARKET, {"close": 19}),
        (-1, OrderType.LIMIT, {"close": 21}),
        (-1, OrderType.MARKET, {"close": 6}),
    ],
)
def test_handle_open_trade_positions(
    dummy_is_strategy,
    mocker,
    direction,
    close_order_type,
    current_candle_data,
) -> None:
    mocker.patch.object(Trader, "is_order_filled", return_value=True)
    mocked_place_close_trade_position_order = mocker.patch.object(
        Trader, "place_close_trade_position_order"
    )

    symbol_open_positions = [
        DBTradePosition(
            open_position=True,
            symbol="BTCUSDT",
            direction=direction,
            desired_entry_price=10,
            open_order_id="12345",
            position_qty=10,
            position_size=10,
            target_price=20,
            stop_price=5,
            initial_account_balance=20,
            exchange_display_name="test_exchange",
            take_profit_order_active=False,
        )
    ]

    dummy_is_strategy.take_profit_order_type = close_order_type
    dummy_is_strategy.stop_loss_order_type = close_order_type
    dummy_is_strategy.handle_open_trade_positions(
        SimpleNamespace(**current_candle_data), symbol_open_positions
    )

    assert mocked_place_close_trade_position_order.called


@pytest.mark.parametrize(
    "direction, candle_data, is_valid",
    [
        (1, {"close": 16, "open_time": 1}, False),
        (1, {"close": 10, "open_time": 1}, True),
        (-1, {"close": 16, "open_time": 1}, False),
        (-1, {"close": 10, "open_time": 1}, True),
    ],
)
def test_generate_and_verify_trader_trade_levels(
    caplog, dummy_is_strategy, direction, candle_data, is_valid
) -> None:
    dummy_trade_levels = TradeLevels(entry_price=10, stop_price=5, target_price=20)

    def dummy_generate_trade_levels(_data, _trade_signal_direction) -> TradeLevels:
        return dummy_trade_levels

    dummy_is_strategy.generate_trade_levels = dummy_generate_trade_levels

    trade_levels = dummy_is_strategy.generate_and_verify_trader_trade_levels(
        SimpleNamespace(**candle_data), 1
    )

    if not is_valid:
        assert not trade_levels
        assert caplog.records[-1].levelname == "WARNING"
        assert "Generated trade levels are invalid" in caplog.text
        return

    assert trade_levels == dummy_trade_levels


def test_handle_finalized_trade_positions(
    dummy_is_strategy, dummy_exchange_orders, mocker
) -> None:
    mocker.patch.object(Trader, "is_order_filled", return_value=True)
    mocked_cancel_position_order = mocker.patch.object(Trader, "cancel_position_order")
    mocked_close_trade_position = mocker.patch.object(Trader, "close_trade_position")

    exchange_orders = dummy_exchange_orders
    exchange_orders[1].executed_quantity = 10

    mocker.patch.object(
        IsExchange,
        "get_exchange_order",
        side_effect=exchange_orders,
    )

    dummy_is_strategy.handle_finalized_trade_positions(
        [DBTradePosition.get(DBTradePosition.id == 1)]
    )

    assert mocked_close_trade_position.called
    assert mocked_cancel_position_order.called


@pytest.mark.parametrize("exchange_symbol_present", [False, True])
def test_run_symbol_trader(
    dummy_is_strategy,
    mocker,
    caplog,
    exchange_symbol_present,
) -> None:
    dummy_is_strategy.trades_database.database.close()
    dummy_is_strategy.timeframe = Timeframe("1m")
    if exchange_symbol_present:
        dummy_is_strategy.exchange.exchange_symbols = {
            "BTCUSDT": Symbol(
                name="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                price_decimals=3,
                quantity_decimals=3,
                tick_size=0.003,
                step_size=0.003,
            )
        }

    ohlcv_data = {
        "open": [1, 6],
        "high": [2, 7],
        "low": [3, 8],
        "close": [4, 9],
        "volume": [5, 10],
    }
    ohlcv_data_df = pd.DataFrame(
        ohlcv_data,
        index=[
            datetime.datetime(2022, 9, 10, 1, 44, 18),
            datetime.datetime(2022, 9, 10, 1, 45, 18),
        ],
    )
    ohlcv_data_df.index.name = "open_time"
    mocker.patch.object(
        Trader, "get_symbol_ohlcv_candles_df", return_value=ohlcv_data_df
    )
    mocker.patch.object(IsStrategy, "is_long_trade_signal_present", return_value=True)
    mocker.patch.object(Trader, "open_trade_position")

    mocked_handle_open_trade_positions = mocker.patch.object(
        Trader, "handle_open_trade_positions"
    )
    mocked_handle_finalized_trade_positions = mocker.patch.object(
        Trader, "handle_finalized_trade_positions"
    )

    dummy_is_strategy.run_symbol_trader("BTCUSDT")

    if not exchange_symbol_present:
        assert caplog.records[-1].levelname == "ERROR"
        assert "provided symbol not present in the exchange" in caplog.text
        return

    assert mocked_handle_open_trade_positions.called
    assert mocked_handle_finalized_trade_positions.called


def test_run_trader(
    mocker, dummy_is_strategy, dummy_is_exchange, trades_db_filepath
) -> None:
    mocked_setup_exchange_for_trading = mocker.patch.object(
        IsExchange, "setup_exchange_for_trading"
    )
    mocked_change_initial_leverage = mocker.patch.object(
        IsExchange, "change_initial_leverage"
    )
    mocker.patch.object(Trader, "get_next_trading_symbol", return_value=None)

    dummy_is_strategy.run_trader(
        dummy_is_exchange, ["BTCUSDT"], "1m", True, trades_db_filepath
    )

    assert mocked_setup_exchange_for_trading.called
    assert mocked_change_initial_leverage.called


@pytest.mark.parametrize(
    "on_demo_mode, is_exchange_testnet, expected_db_name",
    [
        (True, False, "trades_on-demo-mode.sqlite3"),
        (True, True, "trades_on-demo-mode.sqlite3"),
        (False, True, "trades_on-testnet.sqlite3"),
        (False, False, "trades.sqlite3"),
    ],
)
def test_get_db_name(
    dummy_is_strategy,
    on_demo_mode,
    is_exchange_testnet,
    expected_db_name,
) -> None:
    dummy_is_strategy.exchange.testnet = is_exchange_testnet
    dummy_is_strategy.on_demo_mode = on_demo_mode

    db_name = dummy_is_strategy.get_db_name()

    assert db_name == expected_db_name


def test_add_demo_mode_order(dummy_is_strategy) -> None:
    demo_order_id = dummy_is_strategy.add_demo_mode_order(
        "BTCUSDT", OrderSide.BUY, 10, OrderType.LIMIT, 21.4
    )

    assert len(dummy_is_strategy.demo_mode_exchange_orders["BTCUSDT"]) == 1
    assert (
        dummy_is_strategy.demo_mode_exchange_orders["BTCUSDT"][0].order_id
        == demo_order_id
    )


def test_initialize_demo_mode_orders(dummy_is_strategy) -> None:
    dummy_is_strategy.initialize_demo_mode_orders(["BTCUSDT"])
    assert len(dummy_is_strategy.demo_mode_exchange_orders["BTCUSDT"]) == 2


def test_update_executed_demo_order_in_db(
    dummy_is_strategy, dummy_exchange_orders
) -> None:
    demo_order = dummy_exchange_orders[0]
    demo_order.executed_quantity = 15

    db_order = DBOrder.get(DBOrder.order_id == demo_order.order_id)
    dummy_is_strategy.update_executed_demo_order_in_db(demo_order)
    updated_db_order = DBOrder.get(DBOrder.order_id == demo_order.order_id)

    assert db_order.executed_quantity == 10
    assert updated_db_order.executed_quantity == 15


def test_update_symbol_demo_mode_orders(
    mocker,
    dummy_is_strategy,
    dummy_db_trade_positions,
    dummy_exchange_orders,
) -> None:
    dummy_is_strategy.leverage = 5
    dummy_is_strategy.commission = 0.05

    dummy_exchange_order = dummy_exchange_orders[0]
    dummy_exchange_order.order_type = OrderType.MARKET
    dummy_is_strategy.demo_mode_exchange_orders = {"BTCUSDT": [dummy_exchange_order]}

    query = DBOrder.update(
        {DBOrder.is_open_order: True, DBOrder.order_type: OrderType.MARKET}
    ).where(DBOrder.id == 1)
    query.execute()

    mocked_update_executed_demo_order_in_db = mocker.patch.object(
        Trader, "update_executed_demo_order_in_db"
    )

    dummy_is_strategy.update_symbol_demo_mode_orders(
        "BTCUSDT", SimpleNamespace(**{"close": 20})
    )

    assert mocked_update_executed_demo_order_in_db.called


@pytest.mark.parametrize("on_demo_mode", [True, False])
def test_place_order(mocker, dummy_is_strategy, on_demo_mode) -> None:
    if on_demo_mode:
        dummy_is_strategy.on_demo_mode = True
    else:
        dummy_is_strategy.on_demo_mode = False

    mocked_add_demo_mode_order = mocker.patch.object(
        Trader, "add_demo_mode_order", return_value="12345"
    )
    mocked_exchange_place_order = mocker.patch.object(
        IsExchange, "place_order", return_value="12345"
    )

    dummy_is_strategy.place_order("BTCUSDT", OrderSide.BUY, 10, OrderType.MARKET, 100)

    if on_demo_mode:
        assert mocked_add_demo_mode_order.called
    else:
        assert mocked_exchange_place_order.called


def test_calibrate_position_order_quantities(
    mocker,
    dummy_is_strategy,
    dummy_exchange_orders,
    dummy_db_trade_positions,
) -> None:
    dummy_exchange_orders[0].executed_quantity = 9
    dummy_exchange_orders[0].remaining_quantity = 6
    mocker.patch.object(
        Trader,
        "get_exchange_order",
        side_effect=dummy_exchange_orders,
    )

    mocked_cancel_position_order = mocker.patch.object(Trader, "cancel_position_order")
    mocked_place_close_trade_position_order = mocker.patch.object(
        Trader, "place_close_trade_position_order"
    )

    dummy_is_strategy.calibrate_position_order_quantities(
        DBTradePosition.get(DBTradePosition.id == 1)
    )

    assert mocked_cancel_position_order.called
    assert mocked_place_close_trade_position_order.called
