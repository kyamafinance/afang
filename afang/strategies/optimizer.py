import argparse
import builtins
import copy
import csv
import logging
import pathlib
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from afang.exchanges import IsExchange

logger = logging.getLogger(__name__)


class BacktestProfile:
    """Interface for a singular optimization backtest profile."""

    def __init__(self) -> None:
        """Initialize BacktestProfile class."""

        self.backtest_analysis: List[dict] = []
        self.backtest_parameters: Dict = dict()
        self.front: int = 0
        self.crowding_distance: float = 0.0
        # Number of backtest profiles that dominate this profile.
        self.domination_count: int = 0
        # Backtest profiles that this profile dominates.
        self.dominated_profiles: List[int] = list()

    def __eq__(self, other) -> bool:
        """Check for equality between two backtest profiles.

        :param other: another backtest profile.
        :return: bool.
        """

        if isinstance(other, BacktestProfile):
            return self.backtest_parameters == other.backtest_parameters
        return False

    def is_objective_positive_optimization(self, objective: str) -> bool:
        """Return if an objective is a positive optimization i.e. is it better
        if the objective value is larger?

        :param objective: objective to check.
        :return: bool
        """

        try:
            return self.backtest_analysis[0][objective]["positive_optimization"]
        except IndexError:
            raise ValueError(
                "Positive optimization cannot be fetched for an un-analyzed profile"
            )

    def get_objective_value(self, objective: str) -> Any:
        """Get the value of a given objective.

        :param objective: objective whose value is to be retrieved.
        :return: Any
        """

        try:
            return self.backtest_analysis[0][objective]["all_trades"]
        except IndexError:
            raise ValueError(
                "Objective value cannot be fetched for an un-analyzed profile"
            )

    def set_objective_value(self, objective: str, value: Any) -> None:
        """Get the value of a given objective.

        :param objective: objective whose value is to be set.
        :param value: value to use for the update.
        :return: None
        """

        try:
            self.backtest_analysis[0][objective]["all_trades"] = value
        except IndexError:
            raise ValueError("Objective value cannot be set for an un-analyzed profile")

    def reset(self) -> None:
        """Reset the backtest profile's optimization results attributes.

        :return: None
        """

        self.front = 0
        self.domination_count = 0
        self.crowding_distance = 0.0
        self.dominated_profiles.clear()


class StrategyOptimizer:
    """Interface to optimize user defined strategies."""

    def __init__(
        self, strategy: Callable, exchange: IsExchange, cli_args: argparse.Namespace
    ) -> None:
        """Initialize StrategyOptimizer class.

        :param strategy: user defined strategy instance.
        :param exchange: exchange to use in the strategy optimization.
        :param cli_args: command line arguments.
        """

        self.strategy = strategy
        self.exchange = exchange
        self.cli_args = cli_args
        self.population_backtest_params: List[Dict] = list()

        self.strategy_instance = strategy()
        default_objectives = ["average_pnl", "maximum_drawdown"]
        self.optimizer_config = self.strategy_instance.config["optimizer"]
        self.objectives = self.optimizer_config.get("objectives", default_objectives)

    def generate_initial_population(self) -> List[BacktestProfile]:
        """Generate an initial population for the optimizer.

        :return List[BacktestProfile]
        """

        population: List[BacktestProfile] = list()
        while len(population) < self.optimizer_config["population_size"]:
            backtest_profile = BacktestProfile()
            for (param, settings) in self.optimizer_config["parameters"].items():
                if settings["type"] == "int":
                    backtest_profile.backtest_parameters[param] = random.randint(
                        settings["min"], settings["max"]
                    )
                if settings["type"] == "float":
                    backtest_profile.backtest_parameters[param] = round(
                        random.uniform(settings["min"], settings["max"]),
                        settings.get("decimals", 1),
                    )

            # ensure parameters generated are logical.
            backtest_profile.backtest_parameters = (
                self.strategy_instance.define_optimization_param_constraints(
                    backtest_profile.backtest_parameters
                )
            )

            if backtest_profile not in population:
                population.append(backtest_profile)
                self.population_backtest_params.append(
                    backtest_profile.backtest_parameters
                )

        return population

    def evaluate_population(
        self, population: List[BacktestProfile]
    ) -> List[BacktestProfile]:
        """Evaluate a population to get the analysis results for each profile
        within the population.

        :param population: backtest population to evaluate.
        :return: List[BacktestProfile]
        """

        for backtest_profile in population:
            strategy = self.strategy()
            strategy.config["parameters"].update(backtest_profile.backtest_parameters)
            backtest_profile.backtest_analysis = strategy.run_backtest(
                self.exchange, self.cli_args
            )

            # penalize backtest profiles that make no trades.
            if not backtest_profile.get_objective_value("total_trades"):
                for objective in self.objectives:
                    if backtest_profile.is_objective_positive_optimization(objective):
                        backtest_profile.set_objective_value(objective, -float("inf"))
                    else:
                        backtest_profile.set_objective_value(objective, float("inf"))

        return population

    def calculate_crowding_distance(
        self, population: List[BacktestProfile]
    ) -> List[BacktestProfile]:
        """Calculate the crowding distance on a list of backtest profiles and
        return a population that reflects this calculation. Note that the
        population must already be evaluated.

        :param population: population to calculate and sort based on crowding distance.
        :return: List[BacktestProfile]
        """

        if not population:
            return population

        for objective in self.objectives:
            population = sorted(
                population,
                key=lambda x: x.get_objective_value(objective),
            )
            min_value = min(
                population,
                key=lambda x: x.get_objective_value(objective),
            ).get_objective_value(objective)
            max_value = max(
                population,
                key=lambda x: x.get_objective_value(objective),
            ).get_objective_value(objective)

            population[0].crowding_distance = float("inf")
            population[-1].crowding_distance = float("inf")

            for i in range(1, len(population) - 1):
                crowding_distance = population[i + 1].get_objective_value(
                    objective
                ) - population[i - 1].get_objective_value(objective)
                if max_value - min_value:
                    crowding_distance /= max_value - min_value
                    population[i].crowding_distance += crowding_distance

        return population

    def generate_offspring_population(
        self, population: List[BacktestProfile]
    ) -> List[BacktestProfile]:
        """Generate an offspring population based on an initial population
        using parameter crossover and parameter mutation. Note that a
        population of at least two elements is required to generate an
        offspring population.

        :param population: initial population.
        :return: List[BacktestProfile]
        """

        if len(population) < 2:
            return population

        offspring_population: List[BacktestProfile] = list()

        while len(offspring_population) != self.optimizer_config["population_size"]:
            # select two best random parents in a tournament.
            parents: List[BacktestProfile] = list()
            for i in range(2):
                random_parents = random.sample(population, k=2)
                if random_parents[0].front != random_parents[1].front:
                    best_parent = min(random_parents, key=lambda x: x.front)
                else:
                    best_parent = max(random_parents, key=lambda x: x.crowding_distance)
                parents.append(best_parent)

            new_child = BacktestProfile()
            optimization_params = self.optimizer_config["parameters"]
            new_child.backtest_parameters = copy.copy(parents[0].backtest_parameters)

            # Perform parameter crossover.
            num_crossovers = random.randint(1, len(optimization_params))
            params_to_cross = random.sample(
                list(optimization_params.keys()), k=num_crossovers
            )
            for param in params_to_cross:
                new_child.backtest_parameters[param] = copy.copy(
                    parents[1].backtest_parameters[param]
                )

            # Perform parameter mutation.
            num_mutations = random.randint(0, len(optimization_params))
            params_to_mutate = random.sample(
                list(optimization_params.keys()), k=num_mutations
            )
            for param in params_to_mutate:
                mutation_factor = random.uniform(-2, 2)
                param_type = getattr(builtins, optimization_params[param]["type"])
                new_child.backtest_parameters[param] = param_type(
                    new_child.backtest_parameters[param] * (1 + mutation_factor)
                )

                if optimization_params[param]["type"] == "float":
                    new_child.backtest_parameters[param] = round(
                        new_child.backtest_parameters[param],
                        optimization_params[param].get("decimals", 1),
                    )

                new_child.backtest_parameters[param] = max(
                    new_child.backtest_parameters[param],
                    optimization_params[param]["min"],
                )
                new_child.backtest_parameters[param] = min(
                    new_child.backtest_parameters[param],
                    optimization_params[param]["max"],
                )

            # ensure parameters generated are logical.
            new_child.backtest_parameters = (
                self.strategy_instance.define_optimization_param_constraints(
                    new_child.backtest_parameters
                )
            )

            # Add child to offspring population.
            if new_child.backtest_parameters not in self.population_backtest_params:
                offspring_population.append(new_child)
                self.population_backtest_params.append(new_child.backtest_parameters)

        return offspring_population

    @classmethod
    def is_profile_dominant(
        cls,
        profile_a: BacktestProfile,
        profile_b: BacktestProfile,
        objectives: List[str],
    ) -> bool:
        """Check if backtest profile a dominates backtest profile b.

        :param profile_a: backtest profile instance.
        :param profile_b: backtest profile instance.
        :param objectives: list of objectives to judge dominance on.
        :return: bool
        """

        a_objectively_better = True
        a_is_advantageous_to_b = False

        for objective in objectives:
            objective_val_a = profile_a.get_objective_value(objective)
            objective_val_b = profile_b.get_objective_value(objective)

            if profile_a.is_objective_positive_optimization(objective):
                if objective_val_a >= objective_val_b:
                    a_objectively_better = True if a_objectively_better else False
                else:
                    a_objectively_better = False

                if objective_val_a > objective_val_b:
                    a_is_advantageous_to_b = True
            else:
                if objective_val_a <= objective_val_b:
                    a_objectively_better = True if a_objectively_better else False
                else:
                    a_objectively_better = False

                if objective_val_a < objective_val_b:
                    a_is_advantageous_to_b = True

        if a_objectively_better and a_is_advantageous_to_b:
            return True

        return False

    def non_dominated_sorting(
        self, population: Dict[int, BacktestProfile]
    ) -> List[List[BacktestProfile]]:
        """Run non-dominated sorting on an evaluated population and return a
        list that contains lists of backtest profiles sorted according to their
        pareto fronts.

        :param population: evaluated population to sort.
        :return: List[List[BacktestProfile]]
        """

        fronts: List[List[BacktestProfile]] = list()

        for profile_id_x, profile_x in population.items():
            for profile_id_y, profile_y in population.items():
                # check if profile_x dominates or is dominated by other profiles.
                if self.is_profile_dominant(profile_x, profile_y, self.objectives):
                    profile_x.dominated_profiles.append(profile_id_y)
                if self.is_profile_dominant(profile_y, profile_x, self.objectives):
                    profile_x.domination_count += 1

            # check if profile_x is at the pareto optimal front.
            if not profile_x.domination_count:
                if not fronts:
                    fronts.append([])
                fronts[0].append(profile_x)
                profile_x.front = 0

        idx = 0

        while True:
            fronts.append([])
            for profile_x in fronts[idx]:
                for profile_id_y in profile_x.dominated_profiles:
                    population[profile_id_y].domination_count -= 1
                    if population[profile_id_y].domination_count == 0:
                        fronts[idx + 1].append(population[profile_id_y])
                        population[profile_id_y].front = idx + 1

            if len(fronts[idx + 1]) > 0:
                idx += 1
            else:
                del fronts[-1]
                break

        return fronts

    def generate_new_population(
        self, fronts: List[List[BacktestProfile]]
    ) -> List[BacktestProfile]:
        """Given a list of fronts, generate a new population of the desired
        population size ranked by the front and crowding distance.

        :param fronts: sorted fronts.
        :return: List[BacktestProfile]
        """

        new_population: List[BacktestProfile] = list()

        for front in fronts:
            if (
                len(new_population) + len(front)
                > self.optimizer_config["population_size"]
            ):
                max_profiles = self.optimizer_config["population_size"] - len(
                    new_population
                )
                if max_profiles:
                    new_population += sorted(front, key=lambda x: x.crowding_distance)[
                        -max_profiles:
                    ]
            else:
                new_population += front

        return new_population

    def persist_optimization(
        self, final_population: List[BacktestProfile], filepath: Optional[str] = None
    ) -> str:
        """Persist an optimization run in a csv file within the
        `data/optimization` directory. Note that the final population to be
        persisted must have already been evaluated.

        :param final_population: population to persist.
        :param filepath: directory filepath to store optimization results.
        :return: str
        """

        if not filepath:
            filepath = f"{pathlib.Path(__file__).parents[2]}/data/optimization"

        now = datetime.now()
        filename = f'{filepath}/optimization_{now.strftime("%m-%d-%Y-%H:%M:%S")}.csv'

        with open(filename, "w") as file:
            writer = csv.writer(file)
            for idx, profile in enumerate(final_population):
                # Write header if needed.
                if not idx:
                    header = self.objectives.copy()
                    header += list(profile.backtest_parameters.keys())
                    header += ["front", "crowding distance"]
                    writer.writerow(header)

                row = list()
                # Add backtest analysis objectives to the row.
                for objective in self.objectives:
                    if not profile.get_objective_value("total_trades"):
                        # Do not persist this profile because no trade was made.
                        break
                    row.append(profile.get_objective_value(objective))
                # Add backtest params to the row.
                for param_val in profile.backtest_parameters.values():
                    row.append(param_val)
                row.append(profile.front)
                row.append(profile.crowding_distance)
                writer.writerow(row)

        logger.info("Persisted optimization run in: %s", filename)

        return filename

    def optimize(self, persist: bool = True) -> None:
        """Optimize the user provided strategy.

        :param persist: whether to persist the optimization run.
        :return: None
        """

        logger.info(
            "Started optimization on the %s strategy",
            self.strategy_instance.strategy_name,
        )

        p_population = self.generate_initial_population()
        p_population = self.evaluate_population(p_population)
        p_population = self.calculate_crowding_distance(p_population)

        generation_count = 1
        num_generations = self.optimizer_config["num_generations"]
        while generation_count <= num_generations:
            q_population = self.generate_offspring_population(p_population)
            q_population = self.evaluate_population(q_population)
            r_population = p_population + q_population
            self.population_backtest_params.clear()

            idx = 0
            population: Dict[int, BacktestProfile] = dict()
            for backtest_profile in r_population:
                backtest_profile.reset()
                self.population_backtest_params.append(
                    backtest_profile.backtest_parameters
                )
                population[idx] = backtest_profile
                idx += 1

            fronts = self.non_dominated_sorting(population)
            for front_num in range(len(fronts)):
                fronts[front_num] = self.calculate_crowding_distance(fronts[front_num])

            p_population = self.generate_new_population(fronts)

            generation_count += 1

        if persist:
            self.persist_optimization(p_population)

        logger.info(
            "Completed optimization on the %s strategy",
            self.strategy_instance.strategy_name,
        )
