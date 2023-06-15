from dataclasses import fields
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from afang.strategies.analyzer import (
    AnalysisStat,
    MonthlyPnLAnalysis,
    StrategyAnalyzer,
    SymbolAnalysisResult,
)


def dummy_trade_position(
    open_position=False,
    direction=1,
    entry_price=100,
    entry_time=datetime(2021, 1, 1),
    target_price=150,
    stop_price=50,
    holding_time=0,
    trade_count=1,
    exit_time=datetime(2022, 1, 1),
    close_price=150,
    initial_account_balance=10000,
    roe=50.0,
    position_size=200.0,
    cost_adjusted_roe=49.85,
    pnl=99.7,
    commission=0.2,
    slippage=0.1,
    final_account_balance=10099.7,
) -> Any:
    trade_position = dict(
        open_position=open_position,
        direction=direction,
        entry_price=entry_price,
        entry_time=entry_time,
        target_price=target_price,
        stop_price=stop_price,
        holding_time=holding_time,
        trade_count=trade_count,
        exit_time=exit_time,
        close_price=close_price,
        initial_account_balance=initial_account_balance,
        roe=roe,
        position_size=position_size,
        cost_adjusted_roe=cost_adjusted_roe,
        pnl=pnl,
        commission=commission,
        slippage=slippage,
        final_account_balance=final_account_balance,
    )

    return SimpleNamespace(**trade_position)


@pytest.fixture
def dummy_strategy_analyzer(dummy_is_strategy, dummy_is_exchange) -> StrategyAnalyzer:
    dummy_is_strategy.symbols = ["test_symbol"]
    dummy_is_strategy.config["exchange"] = dummy_is_exchange
    return StrategyAnalyzer(dummy_is_strategy)


def test_compute_total_net_profit(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_net_profit()

    assert dummy_strategy_analyzer.analysis_results[0].net_profit == AnalysisStat(
        name="Net Profit",
        all_trades=32.0,
        long_trades=13.0,
        short_trades=19.0,
        is_positive_optimization=True,
    )


def test_compute_gross_profit(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=-10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_gross_profit()

    assert dummy_strategy_analyzer.analysis_results[0].gross_profit == AnalysisStat(
        name="Gross Profit",
        all_trades=13.0,
        long_trades=2.5,
        short_trades=10.5,
        is_positive_optimization=True,
    )


def test_compute_gross_loss(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=-10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_gross_loss()

    assert dummy_strategy_analyzer.analysis_results[0].gross_loss == AnalysisStat(
        name="Gross Loss",
        all_trades=-19.0,
        long_trades=-10.5,
        short_trades=-8.5,
        is_positive_optimization=False,
    )


def test_compute_commission(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, commission=1.2),
                dummy_trade_position(direction=-1, commission=0.8),
                dummy_trade_position(direction=1, commission=2.5),
                dummy_trade_position(direction=-1, commission=3.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_commission()

    assert dummy_strategy_analyzer.analysis_results[0].commission == AnalysisStat(
        name="Commission",
        all_trades=8.0,
        long_trades=3.7,
        short_trades=4.3,
        is_positive_optimization=False,
    )


def test_compute_slippage(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, slippage=1.2),
                dummy_trade_position(direction=-1, slippage=0.8),
                dummy_trade_position(direction=1, slippage=2.5),
                dummy_trade_position(direction=-1, slippage=3.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_slippage()

    assert dummy_strategy_analyzer.analysis_results[0].slippage == AnalysisStat(
        name="Slippage",
        all_trades=8.0,
        long_trades=3.7,
        short_trades=4.3,
        is_positive_optimization=False,
    )


def test_compute_profit_factor(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=-10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_gross_profit()
    dummy_strategy_analyzer.compute_gross_loss()
    dummy_strategy_analyzer.compute_profit_factor()

    assert dummy_strategy_analyzer.analysis_results[0].profit_factor == AnalysisStat(
        name="Profit Factor",
        all_trades=0.6842105263157895,
        long_trades=0.23809523809523808,
        short_trades=1.2352941176470589,
        is_positive_optimization=True,
    )


def test_compute_maximum_drawdown(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, final_account_balance=30.0),
                dummy_trade_position(direction=-1, final_account_balance=60.0),
                dummy_trade_position(direction=1, final_account_balance=25.0),
                dummy_trade_position(direction=-1, final_account_balance=30.0),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_maximum_drawdown()

    assert dummy_strategy_analyzer.analysis_results[0].maximum_drawdown == AnalysisStat(
        name="Maximum Drawdown (%)",
        all_trades=58.33333333333333,
        long_trades=16.666666666666664,
        short_trades=50.0,
        is_positive_optimization=False,
    )


def test_compute_total_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1),
                dummy_trade_position(direction=-1),
                dummy_trade_position(direction=1),
                dummy_trade_position(direction=-1),
                dummy_trade_position(direction=-1),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()

    assert dummy_strategy_analyzer.analysis_results[0].total_trades == AnalysisStat(
        name="Total Trades",
        all_trades=5,
        long_trades=2,
        short_trades=3,
        is_positive_optimization=True,
    )


def test_compute_winning_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_winning_trades()

    assert dummy_strategy_analyzer.analysis_results[0].winning_trades == AnalysisStat(
        name="Winning Trades",
        all_trades=3,
        long_trades=2,
        short_trades=1,
        is_positive_optimization=True,
    )


def test_compute_losing_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_losing_trades()

    assert dummy_strategy_analyzer.analysis_results[0].losing_trades == AnalysisStat(
        name="Losing Trades",
        all_trades=2,
        long_trades=0,
        short_trades=2,
        is_positive_optimization=False,
    )


def test_compute_even_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_even_trades()

    assert dummy_strategy_analyzer.analysis_results[0].even_trades == AnalysisStat(
        name="Even Trades",
        all_trades=1,
        long_trades=1,
        short_trades=0,
        is_positive_optimization=False,
    )


def test_compute_percent_profitable(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_even_trades()
    dummy_strategy_analyzer.compute_percent_profitable()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].percent_profitable == AnalysisStat(
        name="Percent Profitable",
        all_trades=50.0,
        long_trades=100.0,
        short_trades=0.0,
        is_positive_optimization=True,
    )


def test_compute_average_roe(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, cost_adjusted_roe=10.5),
                dummy_trade_position(direction=-1, cost_adjusted_roe=-8.5),
                dummy_trade_position(direction=1, cost_adjusted_roe=0),
                dummy_trade_position(direction=-1, cost_adjusted_roe=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_roe()

    assert dummy_strategy_analyzer.analysis_results[0].average_roe == AnalysisStat(
        name="Average ROE (%)",
        all_trades=-2.125,
        long_trades=5.25,
        short_trades=-9.5,
        is_positive_optimization=True,
    )


def test_compute_average_trade_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_pnl()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].average_trade_pnl == AnalysisStat(
        name="Average Trade PnL",
        all_trades=-2.125,
        long_trades=5.25,
        short_trades=-9.5,
        is_positive_optimization=True,
    )


def test_compute_average_winning_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_average_winning_trade()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].average_winning_trade == AnalysisStat(
        name="Average Winning Trade",
        all_trades=7.833333333333333,
        long_trades=6.5,
        short_trades=10.5,
        is_positive_optimization=True,
    )


def test_compute_average_losing_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_average_losing_trade()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].average_losing_trade == AnalysisStat(
        name="Average Losing Trade",
        all_trades=-8.5,
        long_trades=0,
        short_trades=-8.5,
        is_positive_optimization=False,
    )


def test_compute_take_profit_ratio(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=2.5),
                dummy_trade_position(direction=-1, pnl=10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_average_winning_trade()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_average_losing_trade()
    dummy_strategy_analyzer.compute_take_profit_ratio()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].take_profit_ratio == AnalysisStat(
        name="Take Profit Ratio",
        all_trades=-0.9215686274509803,
        long_trades=0,
        short_trades=-1.2352941176470589,
        is_positive_optimization=True,
    )


def test_compute_trade_expectancy(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_even_trades()
    dummy_strategy_analyzer.compute_percent_profitable()
    dummy_strategy_analyzer.compute_average_winning_trade()
    dummy_strategy_analyzer.compute_average_losing_trade()
    dummy_strategy_analyzer.compute_trade_expectancy()

    assert dummy_strategy_analyzer.analysis_results[0].trade_expectancy == AnalysisStat(
        name="Trade Expectancy",
        all_trades=0.5,
        long_trades=10.5,
        short_trades=-9.5,
        is_positive_optimization=True,
    )


def test_compute_max_consecutive_winners(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_max_consecutive_winners()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].max_consecutive_winners == AnalysisStat(
        name="Maximum Consecutive Winners",
        all_trades=1,
        long_trades=1,
        short_trades=0,
        is_positive_optimization=True,
    )


def test_compute_max_consecutive_losers(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_max_consecutive_losers()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].max_consecutive_losers == AnalysisStat(
        name="Maximum Consecutive Losers",
        all_trades=1,
        long_trades=0,
        short_trades=2,
        is_positive_optimization=False,
    )


def test_compute_largest_winning_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_largest_winning_trade()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].largest_winning_trade == AnalysisStat(
        name="Largest Winning Trade",
        all_trades=10.5,
        long_trades=10.5,
        short_trades=-8.5,
        is_positive_optimization=True,
    )


def test_compute_largest_losing_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, pnl=10.5),
                dummy_trade_position(direction=-1, pnl=-8.5),
                dummy_trade_position(direction=1, pnl=0),
                dummy_trade_position(direction=-1, pnl=-10.5),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_largest_losing_trade()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].largest_losing_trade == AnalysisStat(
        name="Largest Losing Trade",
        all_trades=-10.5,
        long_trades=0,
        short_trades=-10.5,
        is_positive_optimization=False,
    )


def test_compute_average_holding_time(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, holding_time=1),
                dummy_trade_position(direction=-1, holding_time=2),
                dummy_trade_position(direction=1, holding_time=3),
                dummy_trade_position(direction=-1, holding_time=4),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_holding_time()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].average_holding_time == AnalysisStat(
        name="Average Holding Time (candles)",
        all_trades=2.5,
        long_trades=2,
        short_trades=3,
        is_positive_optimization=False,
    )


def test_compute_maximum_holding_time(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(direction=1, holding_time=1),
                dummy_trade_position(direction=-1, holding_time=2),
                dummy_trade_position(direction=1, holding_time=3),
                dummy_trade_position(direction=-1, holding_time=4),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_maximum_holding_time()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].maximum_holding_time == AnalysisStat(
        name="Maximum Holding Time",
        all_trades=4,
        long_trades=3,
        short_trades=4,
        is_positive_optimization=False,
    )


def test_compute_monthly_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(
                    direction=1, pnl=10.5, exit_time=datetime(2022, 1, 1)
                ),
                dummy_trade_position(
                    direction=-1, pnl=-8.5, exit_time=datetime(2022, 1, 2)
                ),
                dummy_trade_position(
                    direction=1, pnl=0, exit_time=datetime(2022, 2, 1)
                ),
                dummy_trade_position(
                    direction=-1, pnl=-10.5, exit_time=datetime(2022, 2, 1)
                ),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_monthly_pnl()

    assert dummy_strategy_analyzer.analysis_results[0].monthly_pnl == [
        MonthlyPnLAnalysis(
            month="1-2022", all_trades=2.0, long_trades=10.5, short_trades=-8.5
        ),
        MonthlyPnLAnalysis(
            month="2-2022", all_trades=-10.5, long_trades=0, short_trades=-10.5
        ),
    ]


def test_compute_average_monthly_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        SymbolAnalysisResult(
            symbol="test_symbol",
            trades=[
                dummy_trade_position(
                    direction=1, pnl=10.5, exit_time=datetime(2022, 1, 1)
                ),
                dummy_trade_position(
                    direction=-1, pnl=-8.5, exit_time=datetime(2022, 1, 2)
                ),
                dummy_trade_position(
                    direction=1, pnl=0, exit_time=datetime(2022, 2, 1)
                ),
                dummy_trade_position(
                    direction=-1, pnl=-10.5, exit_time=datetime(2022, 2, 1)
                ),
            ],
        )
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_monthly_pnl()
    dummy_strategy_analyzer.compute_average_monthly_pnl()

    assert dummy_strategy_analyzer.analysis_results[
        0
    ].average_monthly_pnl == AnalysisStat(
        name="Average Monthly PnL",
        all_trades=-4.25,
        long_trades=5.25,
        short_trades=-9.5,
        is_positive_optimization=True,
    )


def test_run_analysis(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.run_analysis()

    analysis_results = dummy_strategy_analyzer.analysis_results
    analysis_results_fields = [field.name for field in fields(analysis_results[0])]

    assert len(analysis_results) == 1
    assert len(analysis_results_fields) == 28
