import cvxpy as cp
import numpy as np
from datetime import datetime
from utils.finance_utils import get_adj_close_from_stocks, get_portfolio_return

def qp(
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds: list,
    target_return: float = None,
):
    """
    QP solver for Markowitz optimization.
    """
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    if adj_close_df.empty:
        raise ValueError(f"Stocks {stocks} not found in range {start_date} to {end_date}")

    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 250.8875
    mean_returns = np.array(log_returns.mean() * 250.8875).flatten()

    n = len(stocks)
    w = cp.Variable(n)

    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(portfolio_variance)

    constraints = [cp.sum(w) == 1]

    if target_return is not None:
        constraints.append(mean_returns @ w >= target_return)

    for i, (lower, upper) in enumerate(bounds):
        constraints.append(w[i] >= lower)
        constraints.append(w[i] <= upper)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    return {
        "weights": w.value,
        "expected_return": mean_returns @ w.value,
        "volatility": np.sqrt(w.value.T @ cov_matrix.values @ w.value),
        "status": prob.status,
    }

if __name__ == "__main__":
    print(qp(["AAPL", "MSFT", "GOOG"], datetime(2020, 1, 1), datetime(2023, 1, 1), [(0, 1), (0, 1), (0,1)]))
