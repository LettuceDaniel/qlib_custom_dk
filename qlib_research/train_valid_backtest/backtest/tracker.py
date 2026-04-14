import numpy as np

class PositionTracker:
    def __init__(self, initial_capital):
        self.cash = float(initial_capital)
        self.position = {}

    def apply_returns(self, prices, stop_loss=0.0):
        for stock_id in list(self.position.keys()):
            # Get return, default to NaN if not found
            ret = prices.get(stock_id, np.nan)

            # If return is NaN (trading halt or missing data), we keep the position as is (ret=0)
            if np.isnan(ret):
                continue

            # If return is -1 or less (delisting/total loss), the position value becomes 0
            if ret <= -1.0:
                del self.position[stock_id]
                continue

            # Apply return to the accumulated price
            self.position[stock_id]["price"] *= (1 + ret)

            # Stop-loss check: liquidate if price dropped below entry * (1 - stop_loss)
            if stop_loss > 0 and "entry_price" in self.position[stock_id]:
                entry_price = self.position[stock_id]["entry_price"]
                current_price = self.position[stock_id]["price"]
                if current_price <= entry_price * (1 - stop_loss):
                    pos = self.position[stock_id]
                    recovered_value = pos["amount"] * current_price
                    self.cash += recovered_value
                    del self.position[stock_id]

    @property
    def position_value(self):
        return sum(p["amount"] * p["price"] for p in self.position.values())

    @property
    def total_value(self):
        return self.cash + self.position_value

    def snapshot(self, date):
        pos_val = self.position_value
        total_val = self.cash + pos_val
        return {
            "datetime": date,
            "cash": self.cash,
            "position_value": pos_val,
            "total_value": total_val,
        }
