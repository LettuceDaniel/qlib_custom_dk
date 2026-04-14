import os
from datetime import datetime
import pandas as pd
import numpy as np

# Import from existing modules (shared utilities, no duplication)
from qlib_research.train_valid_backtest.evaluation.ic import compute_ic_metrics
from qlib_research.train_valid_backtest.evaluation.risk import (
    risk_analysis,
    compute_daily_pred_chg,
    compute_portfolio_turnover,
)


def build_results_and_report(
    predictions,
    results_df_no_cost,
    results_df_with_cost,
    daily_trades_with_cost,
    labels_df,
    test_dates,
    prices,
    benchmark_returns,
    backtest_config,
    output_dir,
    results_filename,
    model_name,
    seed,
):
    """공통 결과 구성 + CSV 저장 + 요약 출력.

    CSV 컬럼 스키마 (고정):
      timestamp, model_name, seed, Val_IC,
      IC, Rank IC,
      Period.Total Return(no_cost), Period.Total Return(with_cost),
      Period.Excess Return(no_cost), Period.Excess Return(with_cost),
      Ann.Total Return(no_cost), Ann.Total Return(with_cost),
      Ann.Excess Return(no_cost), Ann.Excess Return(with_cost),
      Ann_Volatility, Information_Ratio, Max_Drawdown, Win_Rate,
      Daily_Pred_Chg_Rate, Portfolio_Turnover

    Args:
        predictions: dict of {date: DataFrame(instrument, score)}
        results_df_no_cost: DataFrame with 'our_return' column
        results_df_with_cost: DataFrame with 'our_return' column
        daily_trades_with_cost: dict of {date: trade_info}
        labels_df: DataFrame(datetime, instrument, LABEL)
        test_dates: list of dates
        prices: DataFrame (date x instrument)
        benchmark_returns: Series or None
        backtest_config: dict
        output_dir: str
        results_filename: str
        model_name: str
        seed: int or str
    """
    # IC metrics
    ic_metrics = compute_ic_metrics(predictions, labels_df, test_dates)

    daily_no_cost = results_df_no_cost["our_return"]
    daily_with_cost = results_df_with_cost["our_return"]

    topk = backtest_config["backtest"].get("topk", 50)
    daily_pred_chg_rate = compute_daily_pred_chg(predictions, topk=topk)
    portfolio_turnover = compute_portfolio_turnover(results_df_with_cost, daily_trades_with_cost)

    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(daily_with_cost.index).fillna(0)
        excess_daily_no_cost = daily_no_cost - aligned_bench
        excess_daily_with_cost = daily_with_cost - aligned_bench

        risk_excess_no_cost = risk_analysis(excess_daily_no_cost)
        risk_excess_with_cost = risk_analysis(excess_daily_with_cost)
        risk_total_no_cost = risk_analysis(daily_no_cost)
        risk_total_with_cost = risk_analysis(daily_with_cost)

        ann_excess_no_cost = risk_excess_no_cost["annualized_return"]
        ann_excess_with_cost = risk_excess_with_cost["annualized_return"]
        ann_total_no_cost = risk_total_no_cost["annualized_return"]
        ann_total_with_cost = risk_total_with_cost["annualized_return"]
        ann_volatility = risk_excess_with_cost["annualized_volatility"]
        information_ratio = risk_excess_with_cost["information_ratio"]
        max_drawdown = risk_excess_with_cost["max_drawdown"]
        win_rate = risk_excess_with_cost["win_rate"]
        total_no_cost = risk_total_no_cost["total_return"]
        total_with_cost = risk_total_with_cost["total_return"]
        period_excess_no_cost = risk_excess_no_cost["total_return"]
        period_excess_with_cost = risk_excess_with_cost["total_return"]
    else:
        risk_total_no_cost = risk_analysis(daily_no_cost)
        risk_total_with_cost = risk_analysis(daily_with_cost)

        ann_excess_no_cost = 0.0
        ann_excess_with_cost = 0.0
        ann_total_no_cost = risk_total_no_cost["annualized_return"]
        ann_total_with_cost = risk_total_with_cost["annualized_return"]
        ann_volatility = risk_total_with_cost["annualized_volatility"]
        information_ratio = 0.0
        max_drawdown = risk_total_with_cost["max_drawdown"]
        win_rate = risk_total_with_cost["win_rate"]
        total_no_cost = risk_total_no_cost["total_return"]
        total_with_cost = risk_total_with_cost["total_return"]
        period_excess_no_cost = 0.0
        period_excess_with_cost = 0.0

    # Build result row with fixed schema
    results_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "seed": seed,
        "IC": ic_metrics["IC"],
        "Rank IC": ic_metrics["Rank_IC"],
        "Period.Total Return(no_cost)": total_no_cost,
        "Period.Total Return(with_cost)": total_with_cost,
        "Period.Excess Return(no_cost)": period_excess_no_cost,
        "Period.Excess Return(with_cost)": period_excess_with_cost,
        "Ann.Total Return(no_cost)": ann_total_no_cost,
        "Ann.Total Return(with_cost)": ann_total_with_cost,
        "Ann.Excess Return(no_cost)": ann_excess_no_cost,
        "Ann.Excess Return(with_cost)": ann_excess_with_cost,
        "Ann_Volatility": ann_volatility,
        "Information_Ratio": information_ratio,
        "Max_Drawdown": max_drawdown,
        "Win_Rate": win_rate,
        "Daily_Pred_Chg_Rate": daily_pred_chg_rate,
        "Portfolio_Turnover": portfolio_turnover,
    }

    results_df_row = pd.DataFrame([results_row])
    results_path = os.path.join(output_dir, results_filename)

    if os.path.exists(results_path):
        try:
            existing_df = pd.read_csv(results_path)
            combined_df = pd.concat([existing_df, results_df_row], ignore_index=True)
            combined_df.to_csv(results_path, index=False)
        except Exception as e:
            print(f"  Warning: Could not merge with existing results CSV: {e}")
            results_df_row.to_csv(results_path, index=False)
    else:
        results_df_row.to_csv(results_path, index=False)

    print(f"  Results saved to: {results_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print()
    print(f"IC: {ic_metrics['IC']:.4f}, Rank IC: {ic_metrics['Rank_IC']:.4f}")
    print()
    print(
        f"Period Total Return (no cost): {total_no_cost:.4f} ({total_no_cost * 100:.2f}%)"
    )
    print(
        f"Period Total Return (with cost): {total_with_cost:.4f} ({total_with_cost * 100:.2f}%)"
    )
    print(
        f"Period Excess Return (no cost): {period_excess_no_cost:.4f} ({period_excess_no_cost * 100:.2f}%)"
    )
    print(
        f"Period Excess Return (with cost): {period_excess_with_cost:.4f} ({period_excess_with_cost * 100:.2f}%)"
    )
    print()
    print(
        f"Ann. Total Return (no cost): {ann_total_no_cost:.4f} ({ann_total_no_cost * 100:.2f}%)"
    )
    print(
        f"Ann. Total Return (with cost): {ann_total_with_cost:.4f} ({ann_total_with_cost * 100:.2f}%)"
    )
    print(
        f"Ann. Excess Return (no cost): {ann_excess_no_cost:.4f} ({ann_excess_no_cost * 100:.2f}%)"
    )
    print(
        f"Ann. Excess Return (with cost): {ann_excess_with_cost:.4f} ({ann_excess_with_cost * 100:.2f}%)"
    )
    print()
    print(f"Information Ratio: {information_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown * 100:.2f}%)")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Daily Pred Chg Rate: {daily_pred_chg_rate * 100:.2f}%")
    print(f"Portfolio Turnover: {portfolio_turnover * 100:.2f}%")
    print(f"{'=' * 70}\n")

    return results_row