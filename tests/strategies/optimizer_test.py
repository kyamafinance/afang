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
