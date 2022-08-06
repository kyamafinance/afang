from datetime import datetime

import pytest

from afang.strategies.analyzer import StrategyAnalyzer


@pytest.fixture
def dummy_strategy_analyzer(dummy_is_strategy, dummy_is_exchange) -> StrategyAnalyzer:
    dummy_is_strategy.config["exchange"] = dummy_is_exchange
    dummy_is_strategy.trade_positions = {
        "test_symbol": {
            "1": {
                "direction": 1,
                "pnl": 10.5,
                "exit_time": datetime(2022, 1, 1),
                "initial_account_balance": 1000,
                "trade_count": 1,
            }
        }
    }
    return StrategyAnalyzer(dummy_is_strategy)


def test_compute_total_net_profit(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": 8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_net_profit()

    assert dummy_strategy_analyzer.analysis_results[0]["total_net_profit"] == {
        "all_trades": 32.0,
        "long_trades": 13.0,
        "short_trades": 19.0,
    }


def test_compute_gross_profit(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": -10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_gross_profit()

    assert dummy_strategy_analyzer.analysis_results[0]["gross_profit"] == {
        "all_trades": 13.0,
        "long_trades": 2.5,
        "short_trades": 10.5,
    }


def test_compute_gross_loss(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": -10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_gross_loss()

    assert dummy_strategy_analyzer.analysis_results[0]["gross_loss"] == {
        "all_trades": -19.0,
        "long_trades": -10.5,
        "short_trades": -8.5,
    }


def test_compute_commission(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "commission": 1.2},
                {"direction": -1, "commission": 0.8},
                {"direction": 1, "commission": 2.5},
                {"direction": -1, "commission": 3.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_commission()

    assert dummy_strategy_analyzer.analysis_results[0]["commission"] == {
        "all_trades": 8.0,
        "long_trades": 3.7,
        "short_trades": 4.3,
    }


def test_compute_slippage(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "slippage": 1.2},
                {"direction": -1, "slippage": 0.8},
                {"direction": 1, "slippage": 2.5},
                {"direction": -1, "slippage": 3.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_slippage()

    assert dummy_strategy_analyzer.analysis_results[0]["slippage"] == {
        "all_trades": 8.0,
        "long_trades": 3.7,
        "short_trades": 4.3,
    }


def test_compute_profit_factor(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": -10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_gross_profit()
    dummy_strategy_analyzer.compute_gross_loss()
    dummy_strategy_analyzer.compute_profit_factor()

    assert dummy_strategy_analyzer.analysis_results[0]["profit_factor"] == {
        "all_trades": 0.6842105263157895,
        "long_trades": 0.23809523809523808,
        "short_trades": 1.2352941176470589,
    }


def test_compute_maximum_drawdown(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "final_account_balance": 30.0},
                {"direction": -1, "final_account_balance": 60.0},
                {"direction": 1, "final_account_balance": 25.0},
                {"direction": -1, "final_account_balance": 30.0},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_maximum_drawdown()

    assert dummy_strategy_analyzer.analysis_results[0]["maximum_drawdown"] == {
        "all_trades": -58.33333333333333,
        "long_trades": -16.666666666666664,
        "short_trades": -50.0,
        "positive_optimization": True,
    }


def test_compute_total_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1},
                {"direction": -1},
                {"direction": 1},
                {"direction": -1},
                {"direction": -1},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()

    assert dummy_strategy_analyzer.analysis_results[0]["total_trades"] == {
        "all_trades": 5,
        "long_trades": 2,
        "short_trades": 3,
    }


def test_compute_winning_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_winning_trades()

    assert dummy_strategy_analyzer.analysis_results[0]["winning_trades"] == {
        "all_trades": 3,
        "long_trades": 2,
        "short_trades": 1,
    }


def test_compute_losing_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_losing_trades()

    assert dummy_strategy_analyzer.analysis_results[0]["losing_trades"] == {
        "all_trades": 2,
        "long_trades": 0,
        "short_trades": 2,
    }


def test_compute_even_trades(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_even_trades()

    assert dummy_strategy_analyzer.analysis_results[0]["even_trades"] == {
        "all_trades": 1,
        "long_trades": 1,
        "short_trades": 0,
    }


def test_compute_percent_profitable(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_even_trades()
    dummy_strategy_analyzer.compute_percent_profitable()

    assert dummy_strategy_analyzer.analysis_results[0]["percent_profitable"] == {
        "all_trades": 50.0,
        "long_trades": 100.0,
        "short_trades": 0.0,
    }


def test_compute_average_roe(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "cost_adjusted_roe": 10.5},
                {"direction": -1, "cost_adjusted_roe": -8.5},
                {"direction": 1, "cost_adjusted_roe": 0},
                {"direction": -1, "cost_adjusted_roe": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_roe()

    assert dummy_strategy_analyzer.analysis_results[0]["average_roe"] == {
        "all_trades": -2.125,
        "long_trades": 5.25,
        "short_trades": -9.5,
    }


def test_compute_average_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_pnl()

    assert dummy_strategy_analyzer.analysis_results[0]["average_pnl"] == {
        "all_trades": -2.125,
        "long_trades": 5.25,
        "short_trades": -9.5,
        "positive_optimization": True,
    }


def test_compute_average_winning_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_average_winning_trade()

    assert dummy_strategy_analyzer.analysis_results[0]["average_winning_trade"] == {
        "all_trades": 7.833333333333333,
        "long_trades": 6.5,
        "short_trades": 10.5,
    }


def test_compute_average_losing_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_average_losing_trade()

    assert dummy_strategy_analyzer.analysis_results[0]["average_losing_trade"] == {
        "all_trades": -8.5,
        "long_trades": 0,
        "short_trades": -8.5,
    }


def test_compute_take_profit_ratio(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 2.5},
                {"direction": -1, "pnl": 10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_average_winning_trade()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_average_losing_trade()
    dummy_strategy_analyzer.compute_take_profit_ratio()

    assert dummy_strategy_analyzer.analysis_results[0]["take_profit_ratio"] == {
        "all_trades": -0.9215686274509803,
        "long_trades": 0,
        "short_trades": -1.2352941176470589,
    }


def test_compute_trade_expectancy(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_winning_trades()
    dummy_strategy_analyzer.compute_losing_trades()
    dummy_strategy_analyzer.compute_even_trades()
    dummy_strategy_analyzer.compute_percent_profitable()
    dummy_strategy_analyzer.compute_average_winning_trade()
    dummy_strategy_analyzer.compute_average_losing_trade()
    dummy_strategy_analyzer.compute_trade_expectancy()

    assert dummy_strategy_analyzer.analysis_results[0]["trade_expectancy"] == {
        "all_trades": 0.5,
        "long_trades": 10.5,
        "short_trades": -9.5,
    }


def test_compute_max_consecutive_winners(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_max_consecutive_winners()

    assert dummy_strategy_analyzer.analysis_results[0]["max_consecutive_winners"] == {
        "all_trades": 1,
        "long_trades": 1,
        "short_trades": 0,
    }


def test_compute_max_consecutive_losers(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_max_consecutive_losers()

    assert dummy_strategy_analyzer.analysis_results[0]["max_consecutive_losers"] == {
        "all_trades": 1,
        "long_trades": 0,
        "short_trades": 2,
    }


def test_compute_largest_winning_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_largest_winning_trade()

    assert dummy_strategy_analyzer.analysis_results[0]["largest_winning_trade"] == {
        "all_trades": 10.5,
        "long_trades": 10.5,
        "short_trades": -8.5,
    }


def test_compute_largest_losing_trade(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5},
                {"direction": -1, "pnl": -8.5},
                {"direction": 1, "pnl": 0},
                {"direction": -1, "pnl": -10.5},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_largest_losing_trade()

    assert dummy_strategy_analyzer.analysis_results[0]["largest_losing_trade"] == {
        "all_trades": -10.5,
        "long_trades": 0,
        "short_trades": -10.5,
    }


def test_compute_average_holding_time(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "holding_time": 1},
                {"direction": -1, "holding_time": 2},
                {"direction": 1, "holding_time": 3},
                {"direction": -1, "holding_time": 4},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_average_holding_time()

    assert dummy_strategy_analyzer.analysis_results[0]["average_holding_time"] == {
        "all_trades": 2.5,
        "long_trades": 2,
        "short_trades": 3,
    }


def test_compute_maximum_holding_time(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "holding_time": 1},
                {"direction": -1, "holding_time": 2},
                {"direction": 1, "holding_time": 3},
                {"direction": -1, "holding_time": 4},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_maximum_holding_time()

    assert dummy_strategy_analyzer.analysis_results[0]["maximum_holding_time"] == {
        "all_trades": 4,
        "long_trades": 3,
        "short_trades": 4,
    }


def test_compute_monthly_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5, "exit_time": datetime(2022, 1, 1)},
                {"direction": -1, "pnl": -8.5, "exit_time": datetime(2022, 1, 2)},
                {"direction": 1, "pnl": 0, "exit_time": datetime(2022, 2, 1)},
                {"direction": -1, "pnl": -10.5, "exit_time": datetime(2022, 2, 1)},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_monthly_pnl()

    assert dummy_strategy_analyzer.analysis_results[0]["monthly_pnl"] == {
        "all_trades": {"1-2022": 2.0, "2-2022": -10.5},
        "long_trades": {"1-2022": 10.5, "2-2022": 0},
        "short_trades": {"1-2022": -8.5, "2-2022": -10.5},
    }


def test_compute_average_monthly_pnl(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.analysis_results = [
        {
            "symbol": "test_symbol",
            "trades": [
                {"direction": 1, "pnl": 10.5, "exit_time": datetime(2022, 1, 1)},
                {"direction": -1, "pnl": -8.5, "exit_time": datetime(2022, 1, 2)},
                {"direction": 1, "pnl": 0, "exit_time": datetime(2022, 2, 1)},
                {"direction": -1, "pnl": -10.5, "exit_time": datetime(2022, 2, 1)},
            ],
        }
    ]

    dummy_strategy_analyzer.compute_total_trades()
    dummy_strategy_analyzer.compute_monthly_pnl()
    dummy_strategy_analyzer.compute_average_monthly_pnl()

    assert dummy_strategy_analyzer.analysis_results[0]["average_monthly_pnl"] == {
        "all_trades": -4.25,
        "long_trades": 5.25,
        "short_trades": -9.5,
    }


def test_run_analysis(dummy_strategy_analyzer) -> None:
    dummy_strategy_analyzer.run_analysis()

    expected_analysis_result_keys = [
        "symbol",
        "trades",
        "total_net_profit",
        "gross_profit",
        "gross_loss",
        "commission",
        "slippage",
        "profit_factor",
        "maximum_drawdown",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "even_trades",
        "percent_profitable",
        "average_roe",
        "average_pnl",
        "average_winning_trade",
        "average_losing_trade",
        "take_profit_ratio",
        "trade_expectancy",
        "max_consecutive_winners",
        "max_consecutive_losers",
        "largest_winning_trade",
        "largest_losing_trade",
        "average_holding_time",
        "maximum_holding_time",
        "monthly_pnl",
        "average_monthly_pnl",
    ]
    analysis_results = dummy_strategy_analyzer.analysis_results

    assert len(analysis_results) == 2
    assert dummy_strategy_analyzer.analysis_results[0]["symbol"] == "AGGREGATE_DATA"
    assert dummy_strategy_analyzer.analysis_results[1]["symbol"] == "test_symbol"
    assert (
        list(dummy_strategy_analyzer.analysis_results[0].keys())
        == expected_analysis_result_keys
    )
    assert (
        list(dummy_strategy_analyzer.analysis_results[1].keys())
        == expected_analysis_result_keys
    )
