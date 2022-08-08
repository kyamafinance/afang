import pytest


def test_generate_initial_population(dummy_is_optimizer) -> None:
    initial_population = dummy_is_optimizer.generate_initial_population()

    assert len(initial_population) == 2
    assert len(dummy_is_optimizer.population_backtest_params) == len(initial_population)

    for profile in initial_population:
        backtest_params = profile.backtest_parameters
        assert 1.0 <= backtest_params["RR"] <= 5.0
        assert 100 <= backtest_params["ema_period"] <= 800
        assert 0.05 <= backtest_params["psar_max_val"] <= 0.3
        assert 0.01 <= backtest_params["psar_acceleration"] <= 0.08
        assert backtest_params["psar_max_val"] >= backtest_params["psar_acceleration"]


@pytest.mark.parametrize(
    "analysis_result, modified_result",
    [
        (
            [
                {
                    "total_trades": {"all_trades": 0},
                    "average_pnl": {"all_trades": 0, "positive_optimization": True},
                    "maximum_drawdown": {
                        "all_trades": 0,
                        "positive_optimization": False,
                    },
                }
            ],
            [
                {
                    "average_pnl": {
                        "all_trades": float("-inf"),
                        "positive_optimization": True,
                    },
                    "maximum_drawdown": {
                        "all_trades": float("inf"),
                        "positive_optimization": False,
                    },
                    "total_trades": {"all_trades": 0},
                }
            ],
        ),
        (
            [
                {
                    "total_trades": {"all_trades": 1},
                    "average_pnl": {"all_trades": 100, "positive_optimization": True},
                    "maximum_drawdown": {
                        "all_trades": 12,
                        "positive_optimization": False,
                    },
                }
            ],
            [
                {
                    "average_pnl": {
                        "all_trades": 100,
                        "positive_optimization": True,
                    },
                    "maximum_drawdown": {
                        "all_trades": 12,
                        "positive_optimization": False,
                    },
                    "total_trades": {"all_trades": 1},
                }
            ],
        ),
    ],
)
def test_evaluate_population(
    mocker, dummy_is_optimizer, analysis_result, modified_result
) -> None:
    mocked_run_backtest = mocker.patch(
        "afang.strategies.backtester.Backtester.run_backtest",
        return_value=analysis_result,
    )
    initial_population = dummy_is_optimizer.generate_initial_population()
    evaluated_population = dummy_is_optimizer.evaluate_population(initial_population)

    assert mocked_run_backtest.assert_called
    for profile in evaluated_population:
        assert profile.backtest_analysis == modified_result
