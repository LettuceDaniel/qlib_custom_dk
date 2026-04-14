from qlib_research.train_valid_backtest.backtest.engine import run_backtest


def run_backtest_pair(predictions, prices, test_dates, config, benchmark_returns=None):
    """no_cost + with_cost 백테스트를 한 번에 실행.

    Args:
        predictions: dict of {date: DataFrame(instrument, score)}
        prices: DataFrame (date x instrument, RET 사용)
        test_dates: list of dates
        config: backtest config dict
        benchmark_returns: Series or None

    Returns:
        tuple: (results_df_no_cost, results_df_with_cost, daily_trades_with_cost)
        - daily_trades_no_cost는 별도 저장 불필요 (cost 미적용 시 turnover 의미 없음)
    """
    results_no_cost, daily_trades_no_cost = run_backtest(
        predictions,
        prices,
        test_dates,
        config,
        benchmark_returns,
        use_cost=False,
        return_trades=True,
    )
    results_with_cost, daily_trades_with_cost = run_backtest(
        predictions,
        prices,
        test_dates,
        config,
        benchmark_returns,
        use_cost=True,
        return_trades=True,
    )
    return results_no_cost, results_with_cost, daily_trades_with_cost