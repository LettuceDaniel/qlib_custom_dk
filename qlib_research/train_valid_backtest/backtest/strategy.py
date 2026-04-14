import numpy as np
import pandas as pd

class TopkDropStrategy:
    def __init__(self, topk, n_drop, min_hold_days=0):
        self.topk = topk
        self.n_drop = n_drop
        self.min_hold_days = min_hold_days
        self.hold_days = {}

    def record_buys(self, to_buy):
        """Record newly bought stocks with hold_days=0."""
        for stock in to_buy:
            self.hold_days[stock] = 0

    def increment_days(self):
        """Increment hold_days counter for all held stocks."""
        for stock in self.hold_days:
            self.hold_days[stock] += 1

    def generate_trades(self, predictions, current_holdings):
        # Sync hold_days with current_holdings (remove stocks no longer held)
        self.hold_days = {s: self.hold_days.get(s, 0) for s in current_holdings}

        # Single nlargest call to avoid tie-breaking inconsistency between top_k and to_buy
        top_k_df = predictions.nlargest(self.topk, "score")
        top_k = set(top_k_df["instrument"].tolist())

        hold_scores = predictions[predictions["instrument"].isin(current_holdings)]
        missing = set(current_holdings) - set(hold_scores["instrument"])
        if missing:
            missing_df = pd.DataFrame(
                {"instrument": list(missing), "score": [-np.inf] * len(missing)}
            )
            hold_scores = pd.concat([hold_scores, missing_df], ignore_index=True)

        # Instruments held but NOT in top_k → candidates to sell (lowest score first)
        scored = list(zip(hold_scores["instrument"], hold_scores["score"]))
        scored.sort(key=lambda x: x[1])
        to_sell_all = [s for s, _ in scored if s not in top_k]

        # Apply min_hold_days constraint: don't sell stocks below threshold
        if self.min_hold_days > 0:
            to_sell_all = [
                s for s in to_sell_all
                if self.hold_days.get(s, 0) >= self.min_hold_days
            ]

        # Instruments in top_k but NOT currently held → candidates to buy (highest score first)
        hold_set = set(current_holdings)
        to_buy_all = [row["instrument"] for _, row in top_k_df.iterrows() if row["instrument"] not in hold_set]

        # Build initial position: if not yet at topk, buy without dropping
        current_size = len(current_holdings)

        if current_size < self.topk:
            # Not yet at topk: fill positions without dropping
            n_buy = min(len(to_buy_all), self.topk - current_size)
            to_sell = []
            to_buy = to_buy_all[:n_buy]
        else:
            # At topk: apply n_drop for replacement
            n_trades = min(len(to_sell_all), len(to_buy_all), self.n_drop)
            to_sell = to_sell_all[:n_trades]
            to_buy = to_buy_all[:n_trades]

        return to_sell, to_buy