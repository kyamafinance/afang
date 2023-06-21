from types import SimpleNamespace

import pandas as pd
import pytest

from afang.strategies.is_strategy import DBTradePosition, IsStrategy
from afang.strategies.models import TradeLevels


@pytest.mark.parametrize("is_running_backtest", [True, False])
def test_open_trade_position(mocker, dummy_is_strategy, is_running_backtest) -> None:
    mocked_trader_open_trade_position = mocker.patch.object(
        IsStrategy, "trader__open_trade_position"
    )
    mocked_open_backtest_position = mocker.patch.object(
        IsStrategy, "open_backtest_position"
    )

    dummy_is_strategy.is_running_backtest = True
    if not is_running_backtest:
        dummy_is_strategy.is_running_backtest = False

    dummy_is_strategy.open_trade_position(
        "test_symbol",
        1,
        TradeLevels(
            entry_price=1,
        ),
        SimpleNamespace(**{"Index": pd.to_datetime("2022-01-01 01:00:00")}),
    )

    if is_running_backtest:
        assert mocked_open_backtest_position.called
        assert not mocked_trader_open_trade_position.called
        return

    assert mocked_trader_open_trade_position.called
    assert not mocked_open_backtest_position.called


@pytest.mark.parametrize("is_running_backtest", [True, False])
def test_place_close_trade_position_order(
    mocker, dummy_is_strategy, is_running_backtest
) -> None:
    mocker.patch.object(DBTradePosition, "get_by_id")
    mocked_trader_open_trade_position = mocker.patch.object(
        IsStrategy, "trader__place_close_trade_position_order"
    )
    mocked_open_backtest_position = mocker.patch.object(
        IsStrategy, "close_backtest_position"
    )

    dummy_is_strategy.is_running_backtest = True
    if not is_running_backtest:
        dummy_is_strategy.is_running_backtest = False

    dummy_is_strategy.place_close_trade_position_order(
        1, 1, SimpleNamespace(**{"Index": pd.to_datetime("2022-01-01 01:00:00")})
    )

    if is_running_backtest:
        assert mocked_open_backtest_position.called
        assert not mocked_trader_open_trade_position.called
        return

    assert mocked_trader_open_trade_position.called
    assert not mocked_open_backtest_position.called


@pytest.mark.parametrize(
    "is_position_open, is_running_backtest", [(False, True), (True, False)]
)
def test_update_position_trade_levels(
    mocker, dummy_is_strategy, is_position_open, is_running_backtest, caplog
) -> None:
    position = DBTradePosition.get_by_id(1)
    position.is_open = is_position_open
    mocker.patch.object(DBTradePosition, "get_by_id", return_value=position)
    mocked_cancel_position_order = mocker.patch.object(
        IsStrategy, "cancel_position_order"
    )

    dummy_is_strategy.is_running_backtest = True
    if not is_running_backtest:
        dummy_is_strategy.is_running_backtest = False

    dummy_is_strategy.update_position_trade_levels(
        1,
        TradeLevels(entry_price=1, target_price=10, stop_price=11, sequence_id="12345"),
    )

    if not is_position_open:
        assert caplog.records[-1].levelname == "ERROR"
        assert (
            "trade position trade levels could not be updated. trade position is not open"
            in caplog.text
        )
        return

    if not is_running_backtest:
        assert mocked_cancel_position_order.called
    else:
        assert not mocked_cancel_position_order.called

    updated_position: DBTradePosition = DBTradePosition.get_by_id(1)
    assert updated_position.target_price == 10
    assert updated_position.stop_price == 11
    assert updated_position.sequence_id == "12345"
