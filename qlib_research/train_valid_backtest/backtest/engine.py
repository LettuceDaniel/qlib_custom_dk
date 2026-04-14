import pandas as pd
from qlib_research.train_valid_backtest.backtest.tracker import PositionTracker
from qlib_research.train_valid_backtest.backtest.strategy import TopkDropStrategy
from qlib_research.train_valid_backtest.backtest.executor import TradeExecutor

def run_backtest(predictions, prices, test_dates, config, benchmark_returns=None, use_cost=True, return_trades=False):
    """
    Run backtest with TopkDropStrategy.

    Args:
        return_trades: if True, returns (daily_results_df, trades_dict)
                       where trades_dict = {date: {"num_trades": int, "trade_value": float}}
    """
    bt = config["backtest"]
    tracker = PositionTracker(bt["initial_capital"])
    strategy = TopkDropStrategy(
        bt["topk"],
        bt["n_drop"],
        min_hold_days=bt.get("min_hold_days", 0),
    )
    executor = TradeExecutor(
        open_cost=bt["open_cost"] if use_cost else 0,
        close_cost=bt["close_cost"] if use_cost else 0,
        min_cost=bt["min_cost"] if use_cost else 0,
    )
    risk_degree = bt["risk_degree"]
    initial_capital = bt["initial_capital"]

    prev_preds = None
    daily_results = []
    daily_trades = {}  # {date: {"num_trades": int, "trade_value": float, "num_stocks": int}}

    for date in test_dates:
        date = pd.Timestamp(date)
        # Ensure robust date indexing
        if date in prices.index:
            today_prices = prices.loc[date]
        else:
            # Try to match by normalized date if necessary, or fallback
            date_normalized = date.normalize()
            if date_normalized in prices.index:
                today_prices = prices.loc[date_normalized]
            else:
                today_prices = pd.Series(0.0, index=prices.columns)

        stop_loss = bt.get("stop_loss", 0.0)
        tracker.apply_returns(today_prices, stop_loss=stop_loss)

        today_preds = predictions.get(date.normalize())

        if today_preds is not None and not today_preds.empty:
            if prev_preds is not None:
                to_sell, to_buy = strategy.generate_trades(
                    prev_preds, set(tracker.position.keys())
                )

                # Track trades for portfolio turnover calculation
                num_sell = len(to_sell)
                num_buy = len(to_buy)
                total_trade_value = 0.0

                # Update tracker.cash immediately after each sell
                for s in to_sell:
                    if s in tracker.position:
                        pos = tracker.position[s]
                        trade_val = pos["amount"] * pos["price"]
                        total_trade_value += trade_val
                        proceeds = executor.execute_sell(tracker.position, s)
                        tracker.cash += proceeds

                # Correct allocation with initial budget and atomic cash updates
                initial_cash = tracker.cash
                budget = initial_cash * risk_degree

                remaining = list(to_buy)
                per_stock_budget = budget / len(remaining) if remaining else 0

                while remaining:
                    stock_id = remaining.pop(0)

                    ret = today_prices.get(stock_id, 0.0)
                    open_cost = executor.open_cost
                    value_per = (
                        per_stock_budget / (1 + open_cost)
                        if (1 + open_cost) > 0
                        else per_stock_budget
                    )

                    total_needed = executor.execute_buy(
                        tracker.position, stock_id, value_per, ret, tracker.cash
                    )

                    if total_needed > 0:
                        # Track buy value (total_needed includes cost)
                        total_trade_value += total_needed
                        tracker.cash -= total_needed
                        budget -= total_needed
                        strategy.record_buys([stock_id])

                # Record daily trade info
                daily_trades[date] = {
                    "num_trades": num_sell + num_buy,
                    "trade_value": total_trade_value,
                    "num_stocks": num_sell + num_buy,
                }

            # Always update prev_preds when today_preds is available
            prev_preds = today_preds

        # Increment hold days counter for all held stocks at end of day
        strategy.increment_days()

        result = tracker.snapshot(date)
        prev_total = (
            daily_results[-1]["total_value"] if daily_results else bt["initial_capital"]
        )
        result["our_return"] = (
            (result["total_value"] - prev_total) / prev_total if prev_total > 0 else 0
        )

        if benchmark_returns is not None:
            if benchmark_returns.index.tz:
                date_key = (
                    date.tz_localize("UTC").tz_convert(benchmark_returns.index.tz)
                    if date.tz is None
                    else date.tz_convert(benchmark_returns.index.tz)
                )
            else:
                date_key = date
            result["benchmark_return"] = benchmark_returns.get(date_key.floor("D"), 0)

        daily_results.append(result)

        # Ensure daily_trades has entry for every date (even if no trades)
        if date not in daily_trades:
            daily_trades[date] = {
                "num_trades": 0,
                "trade_value": 0.0,
                "num_stocks": 0,
            }

    results_df = pd.DataFrame(daily_results).set_index("datetime")

    if return_trades:
        return results_df, daily_trades
    return results_df
