import copy
import logging
from typing import Any, Dict, Iterable, List

import peewee

from afang.database.trades_db.trades_database import TradePosition as DBTradePosition
from afang.strategies.models import (
    AnalysisStat,
    MonthlyPnLAnalysis,
    SymbolAnalysisResult,
)
from afang.utils.function_group import FunctionGroup

logger = logging.getLogger(__name__)
function_group = FunctionGroup()


class StrategyAnalyzer:
    """Interface to analyze user defined strategies."""

    def __init__(self, strategy: Any) -> None:
        """Initialize StrategyAnalyzer class.

        :param strategy: user defined strategy instance.
        """

        self.strategy = strategy
        self.analysis_results: List[SymbolAnalysisResult] = list()

    @function_group.add
    def compute_total_net_profit(self) -> None:
        """Calculate total net profit for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            net_profit = self.safe_sum(trade.pnl for trade in symbol_analysis.trades)
            net_profit_long = self.safe_sum(
                trade.pnl for trade in symbol_analysis.trades if trade.direction == 1
            )
            net_profit_short = self.safe_sum(
                trade.pnl for trade in symbol_analysis.trades if trade.direction == -1
            )

            symbol_analysis.net_profit = AnalysisStat(
                name="Net Profit",
                all_trades=net_profit,
                long_trades=net_profit_long,
                short_trades=net_profit_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_gross_profit(self) -> None:
        """Calculate gross profit for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            gross_profit = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl > 0
            )
            gross_profit_long = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl > 0 and trade.direction == 1
            )
            gross_profit_short = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl > 0 and trade.direction == -1
            )

            symbol_analysis.gross_profit = AnalysisStat(
                name="Gross Profit",
                all_trades=gross_profit,
                long_trades=gross_profit_long,
                short_trades=gross_profit_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_gross_loss(self) -> None:
        """Calculate gross loss for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            gross_loss = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl < 0
            )
            gross_loss_long = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl < 0 and trade.direction == 1
            )
            gross_loss_short = self.safe_sum(
                trade.pnl
                for trade in symbol_analysis.trades
                if trade.pnl is not None and trade.pnl < 0 and trade.direction == -1
            )

            symbol_analysis.gross_loss = AnalysisStat(
                name="Gross Loss",
                all_trades=gross_loss,
                long_trades=gross_loss_long,
                short_trades=gross_loss_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_commission(self) -> None:
        """Calculate commission for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            commission = self.safe_sum(
                trade.commission for trade in symbol_analysis.trades
            )
            commission_long = self.safe_sum(
                trade.commission
                for trade in symbol_analysis.trades
                if trade.direction == 1
            )
            commission_short = self.safe_sum(
                trade.commission
                for trade in symbol_analysis.trades
                if trade.direction == -1
            )

            symbol_analysis.commission = AnalysisStat(
                name="Commission",
                all_trades=commission,
                long_trades=commission_long,
                short_trades=commission_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_slippage(self) -> None:
        """Calculate slippage for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            slippage = self.safe_sum(trade.slippage for trade in symbol_analysis.trades)
            slippage_long = self.safe_sum(
                trade.slippage
                for trade in symbol_analysis.trades
                if trade.direction == 1
            )
            slippage_short = self.safe_sum(
                trade.slippage
                for trade in symbol_analysis.trades
                if trade.direction == -1
            )

            symbol_analysis.slippage = AnalysisStat(
                name="Slippage",
                all_trades=slippage,
                long_trades=slippage_long,
                short_trades=slippage_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_profit_factor(self) -> None:
        """Calculate profit factor for all symbols. This function must be run
        after compute_gross_profit and compute_gross_loss. It relies on both
        gross profit and loss calculations.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            profit_factor = (
                symbol_analysis.gross_profit.all_trades
                / abs(symbol_analysis.gross_loss.all_trades)
                if abs(symbol_analysis.gross_loss.all_trades)
                else 0
            )
            profit_factor_long = (
                symbol_analysis.gross_profit.long_trades
                / abs(symbol_analysis.gross_loss.long_trades)
                if abs(symbol_analysis.gross_loss.long_trades)
                else 0
            )
            profit_factor_short = (
                symbol_analysis.gross_profit.short_trades
                / abs(symbol_analysis.gross_loss.short_trades)
                if abs(symbol_analysis.gross_loss.short_trades)
                else 0
            )

            symbol_analysis.profit_factor = AnalysisStat(
                name="Profit Factor",
                all_trades=profit_factor,
                long_trades=profit_factor_long,
                short_trades=profit_factor_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_maximum_drawdown(self) -> None:
        """Calculate maximum drawdown for all symbols.

        :return: None
        """

        def get_max_drawdown(_returns: List) -> float:
            _max_drawdown: float = 0
            _temp_max_val = 0
            for i in range(1, len(_returns)):
                _temp_max_val = max(_temp_max_val, _returns[i - 1])
                try:
                    _max_drawdown = min(_max_drawdown, _returns[i] / _temp_max_val - 1)
                except ZeroDivisionError:
                    return 100.0

            return -1 * (_max_drawdown * 100.0) if _max_drawdown else 0.0

        for symbol_analysis in self.analysis_results:
            max_drawdown = get_max_drawdown(
                list(
                    trade.final_account_balance
                    for trade in symbol_analysis.trades
                    if trade.final_account_balance
                )
            )
            max_drawdown_long = get_max_drawdown(
                list(
                    trade.final_account_balance
                    for trade in symbol_analysis.trades
                    if trade.direction == 1 and trade.final_account_balance
                )
            )
            max_drawdown_short = get_max_drawdown(
                list(
                    trade.final_account_balance
                    for trade in symbol_analysis.trades
                    if trade.direction == -1 and trade.final_account_balance
                )
            )

            symbol_analysis.maximum_drawdown = AnalysisStat(
                name="Maximum Drawdown (%)",
                all_trades=max_drawdown,
                long_trades=max_drawdown_long,
                short_trades=max_drawdown_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_total_trades(self) -> None:
        """Calculate total number of trades for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            total_trades = len({trade.sequence_id for trade in symbol_analysis.trades})
            total_trades_long = len(
                {
                    trade.sequence_id
                    for trade in symbol_analysis.trades
                    if trade.direction == 1
                }
            )
            total_trades_short = len(
                {
                    trade.sequence_id
                    for trade in symbol_analysis.trades
                    if trade.direction == -1
                }
            )

            symbol_analysis.total_trades = AnalysisStat(
                name="Total Trades",
                all_trades=total_trades,
                long_trades=total_trades_long,
                short_trades=total_trades_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_winning_trades(self) -> None:
        """Calculate number of winning trades for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            winning_trades = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl > 0
                ]
            )
            winning_trades_long = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl > 0 and trade.direction == 1
                ]
            )
            winning_trades_short = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl > 0 and trade.direction == -1
                ]
            )

            symbol_analysis.winning_trades = AnalysisStat(
                name="Winning Trades",
                all_trades=winning_trades,
                long_trades=winning_trades_long,
                short_trades=winning_trades_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_losing_trades(self) -> None:
        """Calculate number of losing trades for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            losing_trades = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0
                ]
            )
            losing_trades_long = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0 and trade.direction == 1
                ]
            )
            losing_trades_short = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0 and trade.direction == -1
                ]
            )

            symbol_analysis.losing_trades = AnalysisStat(
                name="Losing Trades",
                all_trades=losing_trades,
                long_trades=losing_trades_long,
                short_trades=losing_trades_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_even_trades(self) -> None:
        """Calculate number of even trades for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            even_trades = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl == 0
                ]
            )
            even_trades_long = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl == 0 and trade.direction == 1
                ]
            )
            even_trades_short = len(
                [
                    trade
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                    and trade.pnl == 0
                    and trade.direction == -1
                ]
            )

            symbol_analysis.even_trades = AnalysisStat(
                name="Even Trades",
                all_trades=even_trades,
                long_trades=even_trades_long,
                short_trades=even_trades_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_percent_profitable(self) -> None:
        """Calculate percentage profitability for all symbols. This function
        must be run after compute_total_trades, compute_winning_trades,
        compute_losing_trades, and compute_even_trades. It relies on the total
        number of winning, losing, and even trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            percent_profitable = (
                (
                    (
                        symbol_analysis.winning_trades.all_trades
                        + symbol_analysis.even_trades.all_trades
                    )
                    / symbol_analysis.total_trades.all_trades
                )
                * 100.0
                if symbol_analysis.total_trades.all_trades
                else 0
            )
            percent_profitable_long = (
                (
                    (
                        symbol_analysis.winning_trades.long_trades
                        + symbol_analysis.even_trades.long_trades
                    )
                    / symbol_analysis.total_trades.long_trades
                )
                * 100.0
                if symbol_analysis.total_trades.long_trades
                else 0
            )
            percent_profitable_short = (
                (
                    (
                        symbol_analysis.winning_trades.short_trades
                        + symbol_analysis.even_trades.short_trades
                    )
                    / symbol_analysis.total_trades.short_trades
                )
                * 100.0
                if symbol_analysis.total_trades.short_trades
                else 0
            )

            symbol_analysis.percent_profitable = AnalysisStat(
                name="Percent Profitable",
                all_trades=percent_profitable,
                long_trades=percent_profitable_long,
                short_trades=percent_profitable_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_average_roe(self) -> None:
        """Calculate average cost adjusted ROE for all symbols. This function
        needs to run after compute_total_trades. It relies on computing the
        total number of trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_roe = (
                self.safe_sum(
                    trade.cost_adjusted_roe for trade in symbol_analysis.trades
                )
                / symbol_analysis.total_trades.all_trades
                if symbol_analysis.total_trades.all_trades
                else 0
            )
            average_roe_long = (
                self.safe_sum(
                    trade.cost_adjusted_roe
                    for trade in symbol_analysis.trades
                    if trade.direction == 1
                )
                / symbol_analysis.total_trades.long_trades
                if symbol_analysis.total_trades.long_trades
                else 0
            )
            average_roe_short = (
                self.safe_sum(
                    trade.cost_adjusted_roe
                    for trade in symbol_analysis.trades
                    if trade.direction == -1
                )
                / symbol_analysis.total_trades.short_trades
                if symbol_analysis.total_trades.short_trades
                else 0
            )

            symbol_analysis.average_roe = AnalysisStat(
                name="Average ROE (%)",
                all_trades=average_roe,
                long_trades=average_roe_long,
                short_trades=average_roe_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_average_trade_pnl(self) -> None:
        """Calculate average PnL for all symbols. This function needs to run
        after compute_total_trades. It relies on computing the total number of
        trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_trade_pnl = (
                self.safe_sum(trade.pnl for trade in symbol_analysis.trades)
                / symbol_analysis.total_trades.all_trades
                if symbol_analysis.total_trades.all_trades
                else 0
            )
            average_trade_pnl_long = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.trades
                    if trade.direction == 1
                )
                / symbol_analysis.total_trades.long_trades
                if symbol_analysis.total_trades.long_trades
                else 0
            )
            average_trade_pnl_short = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.trades
                    if trade.direction == -1
                )
                / symbol_analysis.total_trades.short_trades
                if symbol_analysis.total_trades.short_trades
                else 0
            )

            symbol_analysis.average_trade_pnl = AnalysisStat(
                name="Average Trade PnL",
                all_trades=average_trade_pnl,
                long_trades=average_trade_pnl_long,
                short_trades=average_trade_pnl_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_average_winning_trade(self) -> None:
        """Calculate average winning trade PnL for all symbols. This function
        needs to run after compute_winning_trades. It relies on the computation
        of the total number of winning trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_winning_trade = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl >= 0
                )
                / symbol_analysis.winning_trades.all_trades
                if symbol_analysis.winning_trades.all_trades
                else 0
            )
            average_winning_trade_long = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl >= 0 and trade.direction == 1
                )
                / symbol_analysis.winning_trades.long_trades
                if symbol_analysis.winning_trades.long_trades
                else 0
            )
            average_winning_trade_short = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                    and trade.pnl >= 0
                    and trade.direction == -1
                )
                / symbol_analysis.winning_trades.short_trades
                if symbol_analysis.winning_trades.short_trades
                else 0
            )

            symbol_analysis.average_winning_trade = AnalysisStat(
                "Average Winning Trade",
                all_trades=average_winning_trade,
                long_trades=average_winning_trade_long,
                short_trades=average_winning_trade_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_average_losing_trade(self) -> None:
        """Calculate average losing trade PnL for all symbols. This function
        needs to run after compute_losing_trades. It relies on the computation
        of the total number of losing trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_losing_trade = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0
                )
                / symbol_analysis.losing_trades.all_trades
                if symbol_analysis.losing_trades.all_trades
                else 0
            )
            average_losing_trade_long = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0 and trade.direction == 1
                )
                / symbol_analysis.losing_trades.long_trades
                if symbol_analysis.losing_trades.long_trades
                else 0
            )
            average_losing_trade_short = (
                self.safe_sum(
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.pnl < 0 and trade.direction == -1
                )
                / symbol_analysis.losing_trades.short_trades
                if symbol_analysis.losing_trades.short_trades
                else 0
            )

            symbol_analysis.average_losing_trade = AnalysisStat(
                name="Average Losing Trade",
                all_trades=average_losing_trade,
                long_trades=average_losing_trade_long,
                short_trades=average_losing_trade_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_take_profit_ratio(self) -> None:
        """Calculate take profit ratio for all symbols. This function needs to
        run after compute_average_winning_trade and
        compute_average_losing_trade. It relies on the computation of the
        average winning and losing trades in terms of PnL.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            take_profit_ratio = (
                symbol_analysis.average_winning_trade.all_trades
                / symbol_analysis.average_losing_trade.all_trades
                if symbol_analysis.average_losing_trade.all_trades
                else 0
            )
            take_profit_ratio_long = (
                symbol_analysis.average_winning_trade.long_trades
                / symbol_analysis.average_losing_trade.long_trades
                if symbol_analysis.average_losing_trade.long_trades
                else 0
            )
            take_profit_ratio_short = (
                symbol_analysis.average_winning_trade.short_trades
                / symbol_analysis.average_losing_trade.short_trades
                if symbol_analysis.average_losing_trade.short_trades
                else 0
            )

            symbol_analysis.take_profit_ratio = AnalysisStat(
                name="Take Profit Ratio",
                all_trades=take_profit_ratio,
                long_trades=take_profit_ratio_long,
                short_trades=take_profit_ratio_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_trade_expectancy(self) -> None:
        """Calculate trade expectancy for all symbols. This function needs to
        run after compute_percent_profitable, compute_average_winning_trade and
        compute_average_losing_trade. It relies on the win rate and average
        winning and losing trade in terms of PnL.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            trade_expectancy = (
                (symbol_analysis.percent_profitable.all_trades / 100.0)
                * symbol_analysis.average_winning_trade.all_trades
            ) - (
                (1.0 - (symbol_analysis.percent_profitable.all_trades / 100.0))
                * abs(symbol_analysis.average_losing_trade.all_trades)
            )
            trade_expectancy_long = (
                (symbol_analysis.percent_profitable.long_trades / 100.0)
                * symbol_analysis.average_winning_trade.long_trades
            ) - (
                (1.0 - (symbol_analysis.percent_profitable.long_trades / 100.0))
                * abs(symbol_analysis.average_losing_trade.long_trades)
            )
            trade_expectancy_short = (
                (symbol_analysis.percent_profitable.short_trades / 100.0)
                * symbol_analysis.average_winning_trade.short_trades
            ) - (
                (1.0 - (symbol_analysis.percent_profitable.short_trades / 100.0))
                * abs(symbol_analysis.average_losing_trade.short_trades)
            )

            symbol_analysis.trade_expectancy = AnalysisStat(
                name="Trade Expectancy",
                all_trades=trade_expectancy,
                long_trades=trade_expectancy_long,
                short_trades=trade_expectancy_short,
                is_positive_optimization=True,
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

        for symbol_analysis in self.analysis_results:
            max_consecutive_winners = get_max_consecutive_winners(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                ]
            )
            max_consecutive_winners_long = get_max_consecutive_winners(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == 1
                ]
            )
            max_consecutive_winners_short = get_max_consecutive_winners(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == -1
                ]
            )

            symbol_analysis.max_consecutive_winners = AnalysisStat(
                name="Maximum Consecutive Winners",
                all_trades=max_consecutive_winners,
                long_trades=max_consecutive_winners_long,
                short_trades=max_consecutive_winners_short,
                is_positive_optimization=True,
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

        for symbol_analysis in self.analysis_results:
            max_consecutive_losers = get_max_consecutive_losers(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                ]
            )
            max_consecutive_losers_long = get_max_consecutive_losers(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == 1
                ]
            )
            max_consecutive_losers_short = get_max_consecutive_losers(
                [
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == -1
                ]
            )

            symbol_analysis.max_consecutive_losers = AnalysisStat(
                name="Maximum Consecutive Losers",
                all_trades=max_consecutive_losers,
                long_trades=max_consecutive_losers_long,
                short_trades=max_consecutive_losers_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_largest_winning_trade(self) -> None:
        """Calculate the largest winning trade for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            largest_winning_trade = max(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                ),
                default=0,
            )
            largest_winning_trade_long = max(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == 1
                ),
                default=0,
            )
            largest_winning_trade_short = max(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == -1
                ),
                default=0,
            )

            symbol_analysis.largest_winning_trade = AnalysisStat(
                name="Largest Winning Trade",
                all_trades=largest_winning_trade,
                long_trades=largest_winning_trade_long,
                short_trades=largest_winning_trade_short,
                is_positive_optimization=True,
            )

    @function_group.add
    def compute_largest_losing_trade(self) -> None:
        """Calculate the largest losing trade for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            largest_losing_trade = min(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None
                ),
                default=0,
            )
            largest_losing_trade_long = min(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == 1
                ),
                default=0,
            )
            largest_losing_trade_short = min(
                (
                    trade.pnl
                    for trade in symbol_analysis.sequenced_trades
                    if trade.pnl is not None and trade.direction == -1
                ),
                default=0,
            )

            symbol_analysis.largest_losing_trade = AnalysisStat(
                name="Largest Losing Trade",
                all_trades=largest_losing_trade,
                long_trades=largest_losing_trade_long,
                short_trades=largest_losing_trade_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_average_holding_time(self) -> None:
        """Calculate average holding time per trade in candles for all symbols.
        This function must be run after compute_total_trades. It relies on the
        total number of trades.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_holding_time = (
                self.safe_sum(
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time
                )
                / symbol_analysis.total_trades.all_trades
                if symbol_analysis.total_trades.all_trades
                else 0
            )
            average_holding_time_long = (
                self.safe_sum(
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time and trade.direction == 1
                )
                / symbol_analysis.total_trades.long_trades
                if symbol_analysis.total_trades.long_trades
                else 0
            )
            average_holding_time_short = (
                self.safe_sum(
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time and trade.direction == -1
                )
                / symbol_analysis.total_trades.short_trades
                if symbol_analysis.total_trades.short_trades
                else 0
            )

            symbol_analysis.average_holding_time = AnalysisStat(
                name="Average Holding Time (candles)",
                all_trades=average_holding_time,
                long_trades=average_holding_time_long,
                short_trades=average_holding_time_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_maximum_holding_time(self) -> None:
        """Calculate maximum holding time per trade in candles for all symbols.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            maximum_holding_time = max(
                (
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time
                ),
                default=0,
            )
            maximum_holding_time_long = max(
                (
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time and trade.direction == 1
                ),
                default=0,
            )
            maximum_holding_time_short = max(
                (
                    trade.holding_time
                    for trade in symbol_analysis.trades
                    if trade.holding_time and trade.direction == -1
                ),
                default=0,
            )

            symbol_analysis.maximum_holding_time = AnalysisStat(
                name="Maximum Holding Time",
                all_trades=maximum_holding_time,
                long_trades=maximum_holding_time_long,
                short_trades=maximum_holding_time_short,
                is_positive_optimization=False,
            )

    @function_group.add
    def compute_monthly_pnl(self) -> None:
        """Calculate monthly PnL for all symbols.

        :return: None
        """

        def get_monthly_pnl(_trades: List[DBTradePosition]) -> List[MonthlyPnLAnalysis]:
            monthly_pnl_stat: Dict[str, MonthlyPnLAnalysis] = dict()

            for trade in _trades:
                if not trade.exit_time or not trade.pnl:
                    continue

                trade_month = f"{trade.exit_time.month}-{trade.exit_time.year}"
                if trade_month not in monthly_pnl_stat:
                    monthly_pnl_stat[trade_month] = MonthlyPnLAnalysis(
                        month=trade_month, all_trades=0, long_trades=0, short_trades=0
                    )

                monthly_pnl_stat[trade_month].all_trades += trade.pnl
                if trade.direction == 1:
                    monthly_pnl_stat[trade_month].long_trades += trade.pnl
                elif trade.direction == -1:
                    monthly_pnl_stat[trade_month].short_trades += trade.pnl

            return list(monthly_pnl_stat.values())

        for symbol_analysis in self.analysis_results:
            monthly_pnl = get_monthly_pnl(symbol_analysis.trades)
            symbol_analysis.monthly_pnl = monthly_pnl

    @function_group.add
    def compute_average_monthly_pnl(self) -> None:
        """Calculate average monthly PnL for all symbols. This function needs
        to run after compute_total_trades and compute_monthly_pnl. It relies on
        the computation of the monthly PnL.

        :return: None
        """

        for symbol_analysis in self.analysis_results:
            average_monthly_pnl = (
                self.safe_sum(
                    monthly_pnl.all_trades
                    for monthly_pnl in symbol_analysis.monthly_pnl
                )
                / len(symbol_analysis.monthly_pnl)
                if len(symbol_analysis.monthly_pnl)
                else 0
            )
            average_monthly_pnl_long = (
                self.safe_sum(
                    monthly_pnl.long_trades
                    for monthly_pnl in symbol_analysis.monthly_pnl
                )
                / len(symbol_analysis.monthly_pnl)
                if len(symbol_analysis.monthly_pnl)
                else 0
            )
            average_monthly_pnl_short = (
                self.safe_sum(
                    monthly_pnl.short_trades
                    for monthly_pnl in symbol_analysis.monthly_pnl
                )
                / len(symbol_analysis.monthly_pnl)
                if len(symbol_analysis.monthly_pnl)
                else 0
            )

            symbol_analysis.average_monthly_pnl = AnalysisStat(
                name="Average Monthly PnL",
                all_trades=average_monthly_pnl,
                long_trades=average_monthly_pnl_long,
                short_trades=average_monthly_pnl_short,
                is_positive_optimization=True,
            )

    @classmethod
    def safe_sum(cls, iterable: Iterable) -> float:
        """Perform a safe sum on an iterable that might contain NoneType
        values.

        :return: float
        """

        return sum(filter(None, iterable))

    @classmethod
    def sequenced_trades(cls, trades: List[DBTradePosition]) -> List[DBTradePosition]:
        """Get a list of trades merged according to their sequence id.

        :param trades: list of DB trade positions.
        :return: List[dict]
        """

        sequenced_trades: Dict[str, DBTradePosition] = dict()
        for trade in trades:
            if trade.sequence_id not in sequenced_trades:
                sequenced_trades[trade.sequence_id] = copy.deepcopy(trade)
                continue

            sequenced_trade: DBTradePosition = sequenced_trades[trade.sequence_id]
            sequenced_trade.pnl += trade.pnl

        return list(sequenced_trades.values())

    def run_analysis(self) -> List[SymbolAnalysisResult]:
        """Run analysis on the user provided strategy.

        :return: List[SymbolAnalysisResult]
        """

        logger.info(
            "%s %s: started analysis on the %s strategy",
            self.strategy.config["exchange"].display_name,
            self.strategy.config["timeframe"],
            self.strategy.strategy_name,
        )

        # Connect to the trades' database.
        self.strategy.trades_database.database.connect(reuse_if_open=True)

        # Populate the analysis results list.
        symbol: str
        for symbol in self.strategy.symbols:
            try:
                symbol_positions = DBTradePosition.select().where(
                    DBTradePosition.symbol == symbol,
                )
            except peewee.PeeweeException as db_error:
                logger.error(
                    "%s %s: could not fetch symbol positions for analysis: %s",
                    self.strategy.config["exchange"].display_name,
                    symbol,
                    db_error,
                )
                continue
            self.analysis_results.append(
                SymbolAnalysisResult(
                    symbol=symbol,
                    trades=symbol_positions,
                    sequenced_trades=self.sequenced_trades(symbol_positions),
                )
            )

        # Compute strategy performance analysis.
        function_group(self)

        # Close the trades' database
        self.strategy.trades_database.database.close()

        logger.info(
            "%s %s: completed analysis on the %s strategy",
            self.strategy.config["exchange"].display_name,
            self.strategy.config["timeframe"],
            self.strategy.strategy_name,
        )

        return self.analysis_results
