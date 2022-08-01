import pytest

from afang.strategies.analyzer import StrategyAnalyzer


@pytest.fixture
def dummy_strategy_analyzer(dummy_is_strategy) -> StrategyAnalyzer:
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
