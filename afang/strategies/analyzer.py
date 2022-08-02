import logging
import operator
from functools import reduce
from statistics import mean
from typing import Any, Dict, List

from afang.utils.function_group import FunctionGroup

logger = logging.getLogger(__name__)

function_group = FunctionGroup()


class StrategyAnalyzer:
    """Interface to analyze any user defined strategies."""

    def __init__(self, strategy: Any) -> None:
        """Initialize StrategyAnalyzer class.

        :param strategy: user defined strategy instance.
        """

        self.strategy = strategy
        self.analysis_results: List[dict] = list()

    @function_group.add
    def compute_total_net_profit(self) -> None:
        """Calculate total net profit for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            total_net_profit = sum(
                trade.get("pnl", 0) for trade in symbol_backtest["trades"]
            )
            total_net_profit_long = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == 1
            )
            total_net_profit_short = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == -1
            )
            symbol_backtest.update(
                {
                    "total_net_profit": {
                        "all_trades": total_net_profit,
                        "long_trades": total_net_profit_long,
                        "short_trades": total_net_profit_short,
                    }
                }
            )

    @function_group.add
    def compute_gross_profit(self) -> None:
        """Calculate gross profit for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            gross_profit = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) > 0
            )
            gross_profit_long = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) > 0 and trade["direction"] == 1
            )
            gross_profit_short = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) > 0 and trade["direction"] == -1
            )
            symbol_backtest.update(
                {
                    "gross_profit": {
                        "all_trades": gross_profit,
                        "long_trades": gross_profit_long,
                        "short_trades": gross_profit_short,
                    }
                }
            )

    @function_group.add
    def compute_gross_loss(self) -> None:
        """Calculate gross loss for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            gross_loss = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) < 0
            )
            gross_loss_long = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) < 0 and trade["direction"] == 1
            )
            gross_loss_short = sum(
                trade.get("pnl", 0)
                for trade in symbol_backtest["trades"]
                if trade.get("pnl", 0) < 0 and trade["direction"] == -1
            )
            symbol_backtest.update(
                {
                    "gross_loss": {
                        "all_trades": gross_loss,
                        "long_trades": gross_loss_long,
                        "short_trades": gross_loss_short,
                    }
                }
            )

    @function_group.add
    def compute_commission(self) -> None:
        """Calculate commission for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            commission = sum(
                trade.get("commission", 0) for trade in symbol_backtest["trades"]
            )
            commission_long = sum(
                trade.get("commission", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == 1
            )
            commission_short = sum(
                trade.get("commission", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == -1
            )
            symbol_backtest.update(
                {
                    "commission": {
                        "all_trades": commission,
                        "long_trades": commission_long,
                        "short_trades": commission_short,
                    }
                }
            )

    @function_group.add
    def compute_slippage(self) -> None:
        """Calculate slippage for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            slippage = sum(
                trade.get("slippage", 0) for trade in symbol_backtest["trades"]
            )
            slippage_long = sum(
                trade.get("slippage", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == 1
            )
            slippage_short = sum(
                trade.get("slippage", 0)
                for trade in symbol_backtest["trades"]
                if trade["direction"] == -1
            )
            symbol_backtest.update(
                {
                    "slippage": {
                        "all_trades": slippage,
                        "long_trades": slippage_long,
                        "short_trades": slippage_short,
                    }
                }
            )

    @function_group.add
    def compute_profit_factor(self) -> None:
        """Calculate profit factor for all symbols. This function must be run
        after compute_gross_profit and compute_gross_loss. It relies on both
        gross profit and loss calculations.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            profit_factor = (
                symbol_backtest["gross_profit"]["all_trades"]
                / abs(symbol_backtest["gross_loss"]["all_trades"])
                if abs(symbol_backtest["gross_loss"]["all_trades"])
                else 0
            )
            profit_factor_long = (
                symbol_backtest["gross_profit"]["long_trades"]
                / abs(symbol_backtest["gross_loss"]["long_trades"])
                if abs(symbol_backtest["gross_loss"]["long_trades"])
                else 0
            )
            profit_factor_short = (
                symbol_backtest["gross_profit"]["short_trades"]
                / abs(symbol_backtest["gross_loss"]["short_trades"])
                if abs(symbol_backtest["gross_loss"]["short_trades"])
                else 0
            )
            symbol_backtest.update(
                {
                    "profit_factor": {
                        "all_trades": profit_factor,
                        "long_trades": profit_factor_long,
                        "short_trades": profit_factor_short,
                    }
                }
            )

    @function_group.add
    def compute_maximum_drawdown(self) -> None:
        """Calculate maximum drawdown for all symbols.

        :return: None
        """

        def get_max_drawdown(_returns: List) -> float:
            _max_drawdown = 0
            _temp_max_val = 0
            for i in range(1, len(_returns)):
                _temp_max_val = max(_temp_max_val, _returns[i - 1])
                try:
                    _max_drawdown = min(_max_drawdown, _returns[i] / _temp_max_val - 1)
                except ZeroDivisionError:
                    return 100.0

            return _max_drawdown * 100.0

        for symbol_backtest in self.analysis_results:
            max_drawdown = get_max_drawdown(
                list(
                    trade["final_account_balance"]
                    for trade in symbol_backtest["trades"]
                )
            )
            max_drawdown_long = get_max_drawdown(
                list(
                    trade["final_account_balance"]
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                )
            )
            max_drawdown_short = get_max_drawdown(
                list(
                    trade["final_account_balance"]
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                )
            )
            symbol_backtest.update(
                {
                    "maximum_drawdown": {
                        "all_trades": max_drawdown,
                        "long_trades": max_drawdown_long,
                        "short_trades": max_drawdown_short,
                    }
                }
            )

    @function_group.add
    def compute_total_trades(self) -> None:
        """Calculate total number of trades for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            total_trades = len(symbol_backtest["trades"])
            total_trades_long = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ]
            )
            total_trades_short = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "total_trades": {
                        "all_trades": total_trades,
                        "long_trades": total_trades_long,
                        "short_trades": total_trades_short,
                    }
                }
            )

    @function_group.add
    def compute_winning_trades(self) -> None:
        """Calculate number of winning trades for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            winning_trades = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) > 0
                ]
            )
            winning_trades_long = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) > 0 and trade["direction"] == 1
                ]
            )
            winning_trades_short = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) > 0 and trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "winning_trades": {
                        "all_trades": winning_trades,
                        "long_trades": winning_trades_long,
                        "short_trades": winning_trades_short,
                    }
                }
            )

    @function_group.add
    def compute_losing_trades(self) -> None:
        """Calculate number of losing trades for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            losing_trades = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0
                ]
            )
            losing_trades_long = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0 and trade["direction"] == 1
                ]
            )
            losing_trades_short = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0 and trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "losing_trades": {
                        "all_trades": losing_trades,
                        "long_trades": losing_trades_long,
                        "short_trades": losing_trades_short,
                    }
                }
            )

    @function_group.add
    def compute_even_trades(self) -> None:
        """Calculate number of even trades for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            even_trades = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) == 0
                ]
            )
            even_trades_long = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) == 0 and trade["direction"] == 1
                ]
            )
            even_trades_short = len(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) == 0 and trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "even_trades": {
                        "all_trades": even_trades,
                        "long_trades": even_trades_long,
                        "short_trades": even_trades_short,
                    }
                }
            )

    @function_group.add
    def compute_percent_profitable(self) -> None:
        """Calculate percentage profitability for all symbols. This function
        must be run after compute_total_trades, compute_winning_trades,
        compute_losing_trades, and compute_even_trades. It relies on the total
        number of winning, losing, and even trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            percent_profitable = (
                (
                    (
                        symbol_backtest["winning_trades"]["all_trades"]
                        + symbol_backtest["even_trades"]["all_trades"]
                    )
                    / symbol_backtest["total_trades"]["all_trades"]
                )
                * 100.0
                if symbol_backtest["total_trades"]["all_trades"]
                else 0
            )
            percent_profitable_long = (
                (
                    (
                        symbol_backtest["winning_trades"]["long_trades"]
                        + symbol_backtest["even_trades"]["long_trades"]
                    )
                    / symbol_backtest["total_trades"]["long_trades"]
                )
                * 100.0
                if symbol_backtest["total_trades"]["long_trades"]
                else 0
            )
            percent_profitable_short = (
                (
                    (
                        symbol_backtest["winning_trades"]["short_trades"]
                        + symbol_backtest["even_trades"]["short_trades"]
                    )
                    / symbol_backtest["total_trades"]["short_trades"]
                )
                * 100.0
                if symbol_backtest["total_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "percent_profitable": {
                        "all_trades": percent_profitable,
                        "long_trades": percent_profitable_long,
                        "short_trades": percent_profitable_short,
                    }
                }
            )

    @function_group.add
    def compute_average_roe(self) -> None:
        """Calculate average cost adjusted ROE for all symbols. This function
        needs to run after compute_total_trades. It relies on computing the
        total number of trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_roe = (
                sum(
                    trade.get("cost_adjusted_roe", 0)
                    for trade in symbol_backtest["trades"]
                )
                / symbol_backtest["total_trades"]["all_trades"]
                if symbol_backtest["total_trades"]["all_trades"]
                else 0
            )
            average_roe_long = (
                sum(
                    trade.get("cost_adjusted_roe", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                )
                / symbol_backtest["total_trades"]["long_trades"]
                if symbol_backtest["total_trades"]["long_trades"]
                else 0
            )
            average_roe_short = (
                sum(
                    trade.get("cost_adjusted_roe", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                )
                / symbol_backtest["total_trades"]["short_trades"]
                if symbol_backtest["total_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "average_roe": {
                        "all_trades": average_roe,
                        "long_trades": average_roe_long,
                        "short_trades": average_roe_short,
                    }
                }
            )

    @function_group.add
    def compute_average_pnl(self) -> None:
        """Calculate average PnL for all symbols. This function needs to run
        after compute_total_trades. It relies on computing the total number of
        trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_trade = (
                sum(trade.get("pnl", 0) for trade in symbol_backtest["trades"])
                / symbol_backtest["total_trades"]["all_trades"]
                if symbol_backtest["total_trades"]["all_trades"]
                else 0
            )
            average_trade_long = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                )
                / symbol_backtest["total_trades"]["long_trades"]
                if symbol_backtest["total_trades"]["long_trades"]
                else 0
            )
            average_trade_short = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                )
                / symbol_backtest["total_trades"]["short_trades"]
                if symbol_backtest["total_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "average_pnl": {
                        "all_trades": average_trade,
                        "long_trades": average_trade_long,
                        "short_trades": average_trade_short,
                    }
                }
            )

    @function_group.add
    def compute_average_winning_trade(self) -> None:
        """Calculate average winning trade PnL for all symbols. This function
        needs to run after compute_winning_trades. It relies on the computation
        of the total number of winning trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_winning_trade = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) >= 0
                )
                / symbol_backtest["winning_trades"]["all_trades"]
                if symbol_backtest["winning_trades"]["all_trades"]
                else 0
            )
            average_winning_trade_long = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) >= 0 and trade["direction"] == 1
                )
                / symbol_backtest["winning_trades"]["long_trades"]
                if symbol_backtest["winning_trades"]["long_trades"]
                else 0
            )
            average_winning_trade_short = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) >= 0 and trade["direction"] == -1
                )
                / symbol_backtest["winning_trades"]["short_trades"]
                if symbol_backtest["winning_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "average_winning_trade": {
                        "all_trades": average_winning_trade,
                        "long_trades": average_winning_trade_long,
                        "short_trades": average_winning_trade_short,
                    }
                }
            )

    @function_group.add
    def compute_average_losing_trade(self) -> None:
        """Calculate average losing trade PnL for all symbols. This function
        needs to run after compute_losing_trades. It relies on the computation
        of the total number of losing trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_losing_trade = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0
                )
                / symbol_backtest["losing_trades"]["all_trades"]
                if symbol_backtest["losing_trades"]["all_trades"]
                else 0
            )
            average_losing_trade_long = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0 and trade["direction"] == 1
                )
                / symbol_backtest["losing_trades"]["long_trades"]
                if symbol_backtest["losing_trades"]["long_trades"]
                else 0
            )
            average_losing_trade_short = (
                sum(
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade.get("pnl", 0) < 0 and trade["direction"] == -1
                )
                / symbol_backtest["losing_trades"]["short_trades"]
                if symbol_backtest["losing_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "average_losing_trade": {
                        "all_trades": average_losing_trade,
                        "long_trades": average_losing_trade_long,
                        "short_trades": average_losing_trade_short,
                    }
                }
            )

    @function_group.add
    def compute_take_profit_ratio(self) -> None:
        """Calculate take profit ratio for all symbols. This function needs to
        run after compute_average_winning_trade and
        compute_average_losing_trade. It relies on the computation of the
        average winning and losing trades in terms of PnL.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            take_profit_ratio = (
                symbol_backtest["average_winning_trade"]["all_trades"]
                / symbol_backtest["average_losing_trade"]["all_trades"]
                if symbol_backtest["average_losing_trade"]["all_trades"]
                else 0
            )
            take_profit_ratio_long = (
                symbol_backtest["average_winning_trade"]["long_trades"]
                / symbol_backtest["average_losing_trade"]["long_trades"]
                if symbol_backtest["average_losing_trade"]["long_trades"]
                else 0
            )
            take_profit_ratio_short = (
                symbol_backtest["average_winning_trade"]["short_trades"]
                / symbol_backtest["average_losing_trade"]["short_trades"]
                if symbol_backtest["average_losing_trade"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "take_profit_ratio": {
                        "all_trades": take_profit_ratio,
                        "long_trades": take_profit_ratio_long,
                        "short_trades": take_profit_ratio_short,
                    }
                }
            )

    @function_group.add
    def compute_trade_expectancy(self) -> None:
        """Calculate trade expectancy for all symbols. This function needs to
        run after compute_percent_profitable, compute_average_winning_trade and
        compute_average_losing_trade. It relies on the win rate and average
        winning and losing trade in terms of PnL.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            trade_expectancy = (
                (symbol_backtest["percent_profitable"]["all_trades"] / 100.0)
                * symbol_backtest["average_winning_trade"]["all_trades"]
            ) - (
                (1.0 - (symbol_backtest["percent_profitable"]["all_trades"] / 100.0))
                * abs(symbol_backtest["average_losing_trade"]["all_trades"])
            )
            trade_expectancy_long = (
                (symbol_backtest["percent_profitable"]["long_trades"] / 100.0)
                * symbol_backtest["average_winning_trade"]["long_trades"]
            ) - (
                (1.0 - (symbol_backtest["percent_profitable"]["long_trades"] / 100.0))
                * abs(symbol_backtest["average_losing_trade"]["long_trades"])
            )
            trade_expectancy_short = (
                (symbol_backtest["percent_profitable"]["short_trades"] / 100.0)
                * symbol_backtest["average_winning_trade"]["short_trades"]
            ) - (
                (1.0 - (symbol_backtest["percent_profitable"]["short_trades"] / 100.0))
                * abs(symbol_backtest["average_losing_trade"]["short_trades"])
            )
            symbol_backtest.update(
                {
                    "trade_expectancy": {
                        "all_trades": trade_expectancy,
                        "long_trades": trade_expectancy_long,
                        "short_trades": trade_expectancy_short,
                    }
                }
            )

    @function_group.add
    def compute_max_consecutive_winners(self) -> None:
        """Calculate maximum consecutive winners for all symbols.

        :return: None
        """

        def get_max_consecutive_winners(_pnl_series: List[float]) -> int:
            count = 0
            max_val = 0
            for pnl in _pnl_series:
                if pnl > 0:
                    count += 1
                    if count > max_val:
                        max_val = count
                else:
                    count = 0
            return max_val

        for symbol_backtest in self.analysis_results:
            max_consecutive_winners = get_max_consecutive_winners(
                [trade.get("pnl", 0) for trade in symbol_backtest["trades"]]
            )
            max_consecutive_winners_long = get_max_consecutive_winners(
                [
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ]
            )
            max_consecutive_winners_short = get_max_consecutive_winners(
                [
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "max_consecutive_winners": {
                        "all_trades": max_consecutive_winners,
                        "long_trades": max_consecutive_winners_long,
                        "short_trades": max_consecutive_winners_short,
                    }
                }
            )

    @function_group.add
    def compute_max_consecutive_losers(self) -> None:
        """Calculate maximum consecutive losers for all symbols.

        :return: None
        """

        def get_max_consecutive_losers(_pnl_series: List[float]) -> int:
            count = 0
            max_val = 0
            for pnl in _pnl_series:
                if pnl < 0:
                    count += 1
                    if count > max_val:
                        max_val = count
                else:
                    count = 0
            return max_val

        for symbol_backtest in self.analysis_results:
            max_consecutive_losers = get_max_consecutive_losers(
                [trade.get("pnl", 0) for trade in symbol_backtest["trades"]]
            )
            max_consecutive_losers_long = get_max_consecutive_losers(
                [
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ]
            )
            max_consecutive_losers_short = get_max_consecutive_losers(
                [
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "max_consecutive_losers": {
                        "all_trades": max_consecutive_losers,
                        "long_trades": max_consecutive_losers_long,
                        "short_trades": max_consecutive_losers_short,
                    }
                }
            )

    @function_group.add
    def compute_largest_winning_trade(self) -> None:
        """Calculate the largest winning trade for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            largest_winning_trade = max(
                (trade.get("pnl", 0) for trade in symbol_backtest["trades"]), default=0
            )
            largest_winning_trade_long = max(
                (
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ),
                default=0,
            )
            largest_winning_trade_short = max(
                (
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ),
                default=0,
            )
            symbol_backtest.update(
                {
                    "largest_winning_trade": {
                        "all_trades": largest_winning_trade,
                        "long_trades": largest_winning_trade_long,
                        "short_trades": largest_winning_trade_short,
                    }
                }
            )

    @function_group.add
    def compute_largest_losing_trade(self) -> None:
        """Calculate the largest losing trade for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            largest_losing_trade = min(
                (trade.get("pnl", 0) for trade in symbol_backtest["trades"]), default=0
            )
            largest_losing_trade_long = min(
                (
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ),
                default=0,
            )
            largest_losing_trade_short = min(
                (
                    trade.get("pnl", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ),
                default=0,
            )
            symbol_backtest.update(
                {
                    "largest_losing_trade": {
                        "all_trades": largest_losing_trade,
                        "long_trades": largest_losing_trade_long,
                        "short_trades": largest_losing_trade_short,
                    }
                }
            )

    @function_group.add
    def compute_average_holding_time(self) -> None:
        """Calculate average holding time per trade in candles for all symbols.
        This function must be run after compute_total_trades. It relies on the
        total number of trades.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_holding_time = (
                sum(trade.get("holding_time", 0) for trade in symbol_backtest["trades"])
                / symbol_backtest["total_trades"]["all_trades"]
                if symbol_backtest["total_trades"]["all_trades"]
                else 0
            )
            average_holding_time_long = (
                sum(
                    trade.get("holding_time", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                )
                / symbol_backtest["total_trades"]["long_trades"]
                if symbol_backtest["total_trades"]["long_trades"]
                else 0
            )
            average_holding_time_short = (
                sum(
                    trade.get("holding_time", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                )
                / symbol_backtest["total_trades"]["short_trades"]
                if symbol_backtest["total_trades"]["short_trades"]
                else 0
            )
            symbol_backtest.update(
                {
                    "average_holding_time": {
                        "all_trades": average_holding_time,
                        "long_trades": average_holding_time_long,
                        "short_trades": average_holding_time_short,
                    }
                }
            )

    @function_group.add
    def compute_maximum_holding_time(self) -> None:
        """Calculate maximum holding time per trade in candles for all symbols.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            maximum_holding_time = max(
                (trade.get("holding_time", 0) for trade in symbol_backtest["trades"]),
                default=0,
            )
            maximum_holding_time_long = max(
                (
                    trade.get("holding_time", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ),
                default=0,
            )
            maximum_holding_time_short = max(
                (
                    trade.get("holding_time", 0)
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ),
                default=0,
            )
            symbol_backtest.update(
                {
                    "maximum_holding_time": {
                        "all_trades": maximum_holding_time,
                        "long_trades": maximum_holding_time_long,
                        "short_trades": maximum_holding_time_short,
                    }
                }
            )

    @function_group.add
    def compute_monthly_pnl(self) -> None:
        """Calculate monthly PnL for all symbols.

        :return: None
        """

        def get_monthly_pnl(_trades: List[Dict]) -> Dict:
            _monthly_pnl: Dict = dict()
            for trade in _trades:
                trade_month = f"{trade['exit_time'].month}-{trade['exit_time'].year}"
                _monthly_pnl[trade_month] = (
                    _monthly_pnl.get(trade_month, 0) + trade["pnl"]
                )
            return _monthly_pnl

        for symbol_backtest in self.analysis_results:
            monthly_pnl = get_monthly_pnl(symbol_backtest["trades"])
            monthly_pnl_long = get_monthly_pnl(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == 1
                ]
            )
            monthly_pnl_short = get_monthly_pnl(
                [
                    trade
                    for trade in symbol_backtest["trades"]
                    if trade["direction"] == -1
                ]
            )
            symbol_backtest.update(
                {
                    "monthly_pnl": {
                        "all_trades": monthly_pnl,
                        "long_trades": monthly_pnl_long,
                        "short_trades": monthly_pnl_short,
                    }
                }
            )

    @function_group.add
    def compute_average_monthly_pnl(self) -> None:
        """Calculate average monthly PnL for all symbols. This function needs
        to run after compute_total_trades and compute_monthly_pnl. It relies on
        the computation of the total number of trades and the monthly PnL.

        :return: None
        """

        for symbol_backtest in self.analysis_results:
            average_monthly_pnl = (
                sum(symbol_backtest["monthly_pnl"]["all_trades"].values())
                / len(symbol_backtest["monthly_pnl"]["all_trades"])
                if len(symbol_backtest["monthly_pnl"]["all_trades"])
                else 0
            )
            average_monthly_pnl_long = (
                sum(symbol_backtest["monthly_pnl"]["long_trades"].values())
                / len(symbol_backtest["monthly_pnl"]["long_trades"])
                if len(symbol_backtest["monthly_pnl"]["long_trades"])
                else 0
            )
            average_monthly_pnl_short = (
                sum(symbol_backtest["monthly_pnl"]["short_trades"].values())
                / len(symbol_backtest["monthly_pnl"]["short_trades"])
                if len(symbol_backtest["monthly_pnl"]["short_trades"])
                else 0
            )
            symbol_backtest.update(
                {
                    "average_monthly_pnl": {
                        "all_trades": average_monthly_pnl,
                        "long_trades": average_monthly_pnl_long,
                        "short_trades": average_monthly_pnl_short,
                    }
                }
            )

    def run_analysis(self) -> List[dict]:
        """Run analysis on the user provided strategy.

        :return: List[dict]
        """

        logger.info(
            "%s %s: started analysis on the %s strategy",
            self.strategy.config["exchange"].name,
            self.strategy.config["timeframe"],
            self.strategy.strategy_name,
        )

        # Aggregate all backtested trades and sort them by exit time.
        trades = [
            list(symbol_trades.values())
            for symbol_trades in list(self.strategy.trade_positions.values())
        ]
        flattened_trades = reduce(lambda a, b: a + b, trades)
        sorted_aggregate_trades: List[dict] = sorted(
            flattened_trades, key=operator.itemgetter("exit_time")
        )

        # Calculate the initial account balance used when running the strategy.
        initial_account_balance = 0
        if sorted_aggregate_trades:
            initial_account_balance = mean(
                [
                    trade["initial_account_balance"]
                    for trade in sorted_aggregate_trades
                    if trade["trade_count"] == 1
                ]
            )

        # Update the initial and final account values of each trade in the
        # sorted aggregate trades.
        for trade in sorted_aggregate_trades:
            final_account_balance = initial_account_balance + trade["pnl"]
            trade.update(
                {
                    "initial_account_balance": initial_account_balance,
                    "final_account_balance": final_account_balance,
                }
            )
            initial_account_balance = final_account_balance

        # Populate the analysis results dict.
        symbol_trades: dict
        for (symbol, symbol_trades) in self.strategy.trade_positions.items():
            self.analysis_results.append(
                {"symbol": symbol, "trades": list(symbol_trades.values())}
            )
        self.analysis_results.insert(
            0, {"symbol": "AGGREGATE_DATA", "trades": sorted_aggregate_trades}
        )

        # Compute strategy performance analysis.
        function_group(self)

        logger.info(
            "%s %s: completed analysis on the %s strategy",
            self.strategy.config["exchange"].name,
            self.strategy.config["timeframe"],
            self.strategy.strategy_name,
        )

        return self.analysis_results
