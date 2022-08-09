import csv
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


def test_backtest_profile_reset() -> None:
    backtest_profile = BacktestProfile()
    backtest_profile.front = 1
    backtest_profile.domination_count = 2
    backtest_profile.crowding_distance = 3
    backtest_profile.dominated_profiles = [12]

    backtest_profile.reset()

    assert backtest_profile.front == 0
    assert backtest_profile.domination_count == 0
    assert backtest_profile.crowding_distance == 0.0
    assert backtest_profile.dominated_profiles == list()


def test_backtest_profile_equality() -> None:
    profile_a = BacktestProfile()
    profile_a.backtest_parameters["param1"] = 1
    profile_b = BacktestProfile()
    profile_b.backtest_parameters["param1"] = 2

    assert profile_a != profile_b

    profile_b.backtest_parameters["param1"] = 1

    assert profile_a == profile_b


def test_is_objective_positive_optimization() -> None:
    profile = BacktestProfile()

    with pytest.raises(
        ValueError,
        match="Positive optimization cannot be fetched for an un-analyzed profile",
    ):
        profile.is_objective_positive_optimization("param")

    profile.backtest_analysis = [{"param": {"positive_optimization": True}}]

    assert profile.is_objective_positive_optimization("param")


def test_get_objective_value() -> None:
    profile = BacktestProfile()

    with pytest.raises(
        ValueError, match="Objective value cannot be fetched for an un-analyzed profile"
    ):
        profile.get_objective_value("param")

    profile.backtest_analysis = [{"param": {"all_trades": 1}}]

    assert profile.get_objective_value("param") == 1


def test_set_objective_value() -> None:
    profile = BacktestProfile()

    with pytest.raises(
        ValueError, match="Objective value cannot be set for an un-analyzed profile"
    ):
        profile.set_objective_value("param", 2)

    profile.backtest_analysis = [{"param": {"all_trades": 1}}]

    profile.set_objective_value("param", 2)
    assert profile.get_objective_value("param") == 2


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


def test_non_dominated_sorting(
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

    idx = 0
    population: Dict[int, BacktestProfile] = dict()
    for backtest_profile in evaluated_population:
        population[idx] = backtest_profile
        idx += 1

    fronts = dummy_is_optimizer.non_dominated_sorting(population)

    assert len(fronts) == 4
    for front in fronts:
        assert len(front) == 1


def test_generate_new_population(
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

    idx = 0
    population: Dict[int, BacktestProfile] = dict()
    for backtest_profile in evaluated_population:
        population[idx] = backtest_profile
        idx += 1

    fronts = dummy_is_optimizer.non_dominated_sorting(population)
    new_population = dummy_is_optimizer.generate_new_population(fronts)

    assert len(new_population) == 4
    assert new_population[0].crowding_distance == float("inf")
    assert new_population[1].crowding_distance == 0.7857142857142857
    assert new_population[2].crowding_distance == 1.6071428571428572
    assert new_population[3].crowding_distance == float("inf")


def test_persist_optimization(optimization_root_dir, dummy_is_optimizer) -> None:
    backtest_profile = BacktestProfile()
    backtest_profile.backtest_analysis = [
        {
            "total_trades": {"all_trades": 1},
            "average_pnl": {"all_trades": 100, "positive_optimization": True},
            "maximum_drawdown": {
                "all_trades": 50,
                "positive_optimization": False,
            },
        }
    ]
    backtest_profile.backtest_parameters = {
        "RR": 1.5,
        "ema_period": 250,
        "psar_max_val": 0.1,
        "psar_acceleration": 0.02,
    }
    backtest_profile.front = 1
    backtest_profile.crowding_distance = 1.2

    final_population = [backtest_profile]
    filename = dummy_is_optimizer.persist_optimization(
        final_population, filepath=optimization_root_dir
    )

    expected_persisted_optimization_data = [
        [
            "average_pnl",
            "maximum_drawdown",
            "RR",
            "ema_period",
            "psar_max_val",
            "psar_acceleration",
            "front",
            "crowding distance",
        ],
        ["100", "50", "1.5", "250", "0.1", "0.02", "1", "1.2"],
    ]

    with open(filename) as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            assert row == expected_persisted_optimization_data[idx]


def test_optimize(mocker, dummy_is_optimizer) -> None:
    mocked_run_backtest = mocker.patch(
        "afang.strategies.backtester.Backtester.run_backtest",
        side_effect=run_backtest_side_effect,
    )
    mocked_generate_initial_population = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.generate_initial_population"
    )
    mocked_evaluate_population = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.evaluate_population"
    )
    mocked_calculate_crowding_distance = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.calculate_crowding_distance"
    )
    mocked_generate_offspring_population = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.generate_offspring_population"
    )
    mocked_non_dominated_sorting = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.non_dominated_sorting"
    )
    mocked_generate_new_population = mocker.patch(
        "afang.strategies.optimizer.StrategyOptimizer.generate_new_population"
    )

    dummy_is_optimizer.optimize(persist=False)

    assert mocked_run_backtest.assert_called
    assert mocked_generate_initial_population.assert_called
    assert mocked_evaluate_population.assert_called
    assert mocked_calculate_crowding_distance.assert_called
    assert mocked_generate_offspring_population.assert_called
    assert mocked_non_dominated_sorting.assert_called
    assert mocked_generate_new_population.assert_called
