from typing import Dict, List

import pytest

from afang.strategies.optimizer import BacktestProfile


@pytest.fixture
def run_backtest_side_effect() -> List[List[Dict]]:
    return [
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
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 90, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 16,
                    "positive_optimization": False,
                },
            }
        ],
        [
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 80, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 20,
                    "positive_optimization": False,
                },
            }
        ],
        [
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 30, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 28,
                    "positive_optimization": False,
                },
            }
        ],
    ]


def test_generate_initial_population(dummy_is_optimizer) -> None:
    initial_population = dummy_is_optimizer.generate_initial_population()

    assert len(initial_population) == 4
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


def test_calculate_crowding_distance_empty_population(dummy_is_optimizer) -> None:
    result_population = dummy_is_optimizer.calculate_crowding_distance([])
    assert not result_population


def test_calculate_crowding_distance(
    mocker, dummy_is_optimizer, run_backtest_side_effect
) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.run_backtest",
        side_effect=run_backtest_side_effect,
    )

    initial_population = dummy_is_optimizer.generate_initial_population()
    evaluated_population = dummy_is_optimizer.evaluate_population(initial_population)
    result_population = dummy_is_optimizer.calculate_crowding_distance(
        evaluated_population
    )

    expected_crowding_distances = [
        float("inf"),
        0.7857142857142857,
        1.6071428571428572,
        float("inf"),
    ]
    for i in range(len(result_population)):
        assert result_population[i].crowding_distance == expected_crowding_distances[i]


@pytest.mark.parametrize(
    "population",
    [
        [],
        [BacktestProfile()],
    ],
)
def test_generate_offspring_population_too_small_population(
    dummy_is_optimizer, population
) -> None:
    expected_offspring = dummy_is_optimizer.generate_offspring_population(population)
    assert expected_offspring == population


def test_generate_offspring_population(
    mocker, dummy_is_optimizer, run_backtest_side_effect
) -> None:
    mocker.patch(
        "afang.strategies.backtester.Backtester.run_backtest",
        side_effect=run_backtest_side_effect,
    )

    initial_population = dummy_is_optimizer.generate_initial_population()
    evaluated_population = dummy_is_optimizer.evaluate_population(initial_population)
    evaluated_population = dummy_is_optimizer.calculate_crowding_distance(
        evaluated_population
    )
    offspring_population = dummy_is_optimizer.generate_offspring_population(
        evaluated_population
    )

    assert len(offspring_population) == 4
    assert len(dummy_is_optimizer.population_backtest_params) == 8
    for profile in offspring_population:
        backtest_params = profile.backtest_parameters
        assert 1.0 <= backtest_params["RR"] <= 5.0
        assert 100 <= backtest_params["ema_period"] <= 800
        assert 0.05 <= backtest_params["psar_max_val"] <= 0.3
        assert 0.01 <= backtest_params["psar_acceleration"] <= 0.08
        assert backtest_params["psar_max_val"] >= backtest_params["psar_acceleration"]


@pytest.mark.parametrize(
    "profile_a_analysis, profile_b_analysis, expected_result",
    [
        (
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 100, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 50,
                    "positive_optimization": False,
                },
            },
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 90, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 70,
                    "positive_optimization": False,
                },
            },
            True,
        ),
        (
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 90, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 70,
                    "positive_optimization": False,
                },
            },
            {
                "total_trades": {"all_trades": 1},
                "average_pnl": {"all_trades": 100, "positive_optimization": True},
                "maximum_drawdown": {
                    "all_trades": 50,
                    "positive_optimization": False,
                },
            },
            False,
        ),
    ],
)
def test_is_profile_dominant(
    profile_a_analysis, profile_b_analysis, expected_result, dummy_is_optimizer
) -> None:
    profile_a = BacktestProfile()
    profile_a.backtest_analysis = [profile_a_analysis]
    profile_b = BacktestProfile()
    profile_b.backtest_analysis = [profile_b_analysis]

    objectives = ["average_pnl", "maximum_drawdown"]
    result = dummy_is_optimizer.is_profile_dominant(profile_a, profile_b, objectives)

    assert result == expected_result
