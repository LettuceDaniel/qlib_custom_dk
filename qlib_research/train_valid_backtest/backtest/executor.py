import numpy as np

class TradeExecutor:
    def __init__(self, open_cost=0.0, close_cost=0.0, min_cost=0.0):
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost

    def execute_sell(self, position, stock_id):
        """매도 실행: 포지션 삭제 및 대금 반환"""
        pos = position[stock_id]
        trade_val = pos["amount"] * pos["price"]
        cost = max(trade_val * self.close_cost, self.min_cost)
        proceeds = trade_val - cost
        del position[stock_id]
        return proceeds

    def execute_buy(self, position, stock_id, value, ret, cash):
        """매수 실행: 비용 계산, 잔고 확인 후 포지션 생성"""
        if value <= 0 or np.isnan(ret) or ret <= -1.0:
            return 0.0

        cost = max(value * self.open_cost, self.min_cost)
        total_needed = value + cost

        # 잔고 부족 시 매매 실패 처리
        if total_needed > cash:
            return 0.0

        # 실제 포지션 업데이트 (execute_sell과 대칭)
        price = 1.0 * (1 + ret)
        buy_amount = value / price
        position[stock_id] = {
            "amount": buy_amount,
            "price": price,
            "entry_price": price,
        }
        return total_needed
