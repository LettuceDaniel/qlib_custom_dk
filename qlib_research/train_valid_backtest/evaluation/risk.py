import numpy as np
import pandas as pd


def risk_analysis(r, N=252):
    """Risk analysis for simple (arithmetic) returns.

    Args:
        r: daily return series (simple returns, not log returns)
        N: scaler for annualizing (252 for daily)

    Returns:
        dict with metrics including:
        - annualized_return: CAGR (geometric)
        - annualized_volatility: std * sqrt(N) (arithmetic)
        - information_ratio: annualized_return / annualized_volatility

        - max_drawdown: geometric drawdown
        - win_rate: percentage of positive days
        - total_return: cumulative geometric return
    """
    if len(r) == 0 or r.isna().all():
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "information_ratio": 0.0,
            # "sharpe_ratio": 0.0,  # TODO: risk-free rate 파라미터 추가 후 활성화
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    # Arithmetic stats for volatility and Sharpe
    valid_r = r.dropna()
    if len(valid_r) == 0:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "information_ratio": 0.0,
            # "sharpe_ratio": 0.0,  # TODO: risk-free rate 파라미터 추가 후 활성화
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }

    mean = valid_r.mean()
    std = valid_r.std(ddof=1)
    n_valid = len(valid_r)

    # Bug #1: Use valid_r for total_return to avoid NaN propagation
    total_return = (1 + valid_r).prod() - 1

    # Geometric annualized return (CAGR)
    # Use n_valid for proper annualization (actual trading days)
    n_years = n_valid / N
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Annualized volatility: std * sqrt(N) (unchanged - std uses arithmetic)
    annualized_volatility = std * np.sqrt(N)

    # Sharpe ratio (using geometric annualized return and arithmetic volatility)
    ratio = (
        (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
    )

    # Max drawdown: multiplicative compounding using cumulative returns (use valid_r)
    cumulative = (1 + valid_r).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()

    # Win rate: percentage of positive returns (use valid_r)
    win_rate = (valid_r > 0).sum() / len(valid_r) if len(valid_r) > 0 else 0

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "information_ratio": ratio,  # Same as Sharpe when r is excess over benchmark
        # "sharpe_ratio": 0.0,  # TODO: risk-free rate 파라미터 추가 후 활성화. 현재는 미사용.
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,  # Cumulative geometric return
    }


def compute_daily_pred_chg(predictions, topk=50):
    """
    Compute average daily prediction top-k change rate.

    This measures how unstable the model's top-k predictions are between days.
    NOT the same as actual portfolio turnover (which is limited by n_drop).

    Args:
        predictions: dict of {date: pd.Series of scores or DataFrame with instrument/score}
        topk: number of top instruments to consider (default: 50)

    Returns:
        float: average daily rate of change in top-k prediction set (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    dates = sorted(predictions.keys())
    pred_changes = []
    prev_topk = set()

    for date in dates:
        curr_preds = predictions[date]
        if isinstance(curr_preds, pd.DataFrame):
            curr_topk = set(curr_preds.nlargest(topk, "score")["instrument"].tolist())
        else:
            curr_topk = set(curr_preds.nlargest(topk).index.tolist())

        if prev_topk:
            # Number of new stocks in top-k vs previous day
            num_new = len(curr_topk - prev_topk)
            # Daily prediction change rate
            pred_change = num_new / topk
            pred_changes.append(pred_change)

        prev_topk = curr_topk

    return np.mean(pred_changes) if pred_changes else 0.0


# Backward compatibility alias
compute_turnover = compute_daily_pred_chg


def compute_portfolio_turnover(daily_results, daily_trades):
    """
    Compute actual portfolio turnover rate based on executed trades.

    This measures the actual trading activity as a fraction of portfolio value.
    Formula: average daily (total trade value including both buy and sell) / average portfolio value

    Note: trade_value in daily_trades includes BOTH buy and sell sides (gross).
    If one-way turnover is needed, divide the result by 2.

    Args:
        daily_results: DataFrame with 'total_value' column (from run_backtest)
        daily_trades: dict of {date: {"trade_value": float, ...}}

    Returns:
        float: average daily portfolio turnover rate (0.0 to 1.0+)
    """
    if daily_results is None or daily_results.empty:
        return 0.0

    trade_values = []
    portfolio_values = []

    for idx, row in daily_results.iterrows():
        date = idx
        portfolio_values.append(row.get("total_value", 0))

        if isinstance(daily_trades, dict) and date in daily_trades:
            trade_values.append(daily_trades[date].get("trade_value", 0))
        else:
            trade_values.append(0)

    if not portfolio_values or sum(portfolio_values) == 0:
        return 0.0

    # Average portfolio value over the period
    avg_portfolio_value = np.mean(portfolio_values)

    if avg_portfolio_value <= 0:
        return 0.0

    # Total trade value (both buy and sell sides)
    total_trade_value = sum(trade_values)

    # Daily average portfolio turnover
    # Turnover = avg daily trade value / avg portfolio value
    daily_avg_trade = (
        total_trade_value / len(portfolio_values) if len(portfolio_values) > 0 else 0
    )

    return daily_avg_trade / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
