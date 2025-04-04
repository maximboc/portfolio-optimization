from ortools.sat.python import cp_model
import numpy as np
from utils.finance_utils import get_adj_close_from_stocks
from datetime import datetime

def cpsat(    
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds : list,
    risk_rate : int):
    """
    Optimize a portfolio using CP-SAT to find the optimal weights
    
    Parameters:
    - stocks: List of stock symbols
    - start_date: Start date for historical data
    - end_date: End date for historical data
    - bounds: List of (min, max) weight bounds for each stock
    - risk_rate: Risk tolerance parameter (0-1, higher means more aggressive)
    """
    # Step 1: Get historical data
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    
    # Step 2: Calculate returns and covariance
    returns = adj_close_df.pct_change().dropna()
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Step 3: Set up CP-SAT model to generate valid weight combinations
    precision = 1  # 5% increments (0.05)
    
    # Create the model
    model = cp_model.CpModel()
    
    # Create integer variables for weights (as percentages)
    weight_vars = []
    for i, (lower_bound, upper_bound) in enumerate(bounds):
        # Convert bounds to integers (percentage points)
        lower = int(lower_bound * 100)
        upper = int(upper_bound * 100)
        
        # Create integer variable
        w = model.new_int_var(lower, upper, f"weight_{i}")
        weight_vars.append(w)
    
    # Add constraint: weights sum to 100%
    model.add(sum(weight_vars) == 100)
    
    # Add constraint: weights are multiples of precision
    # Fix: Use add_modulo_equality instead of % operator
    for w in weight_vars:
        model.add_modulo_equality(0, w, precision)
    
    # We'll use a callback to collect solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, weight_vars):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.weight_vars = weight_vars
            self.solutions = []
            
        def on_solution_callback(self):
            weights = [self.Value(v) / 100.0 for v in self.weight_vars]
            self.solutions.append(weights)
    
    # Set up the solver
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 4
    
    # Create and register the solution collector
    solution_collector = SolutionCollector(weight_vars)
    
    # Solve the model
    status = solver.solve(model, solution_collector)
    
    if not solution_collector.solutions:
        return "No feasible portfolios found"
    
    # Step 4: Evaluate each feasible portfolio
    possible_portfolios = []
    for weights in solution_collector.solutions:
        # Calculate expected return and risk
        expected_return = sum(weights[i] * expected_returns[i] for i in range(len(stocks)))
        risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Score based on risk_rate
        # Higher risk_rate means we prefer higher returns (more aggressive)
        # Lower risk_rate means we prefer lower risk (more conservative)
        score = risk_rate * expected_return - (1 - risk_rate) * risk
        
        possible_portfolios.append({
            "weights": {stocks[i]: weights[i] for i in range(len(stocks))},
            "expected_return": expected_return,
            "risk": risk,
            "score": score
        })
    
    # Find the portfolio with the highest score
    best_portfolio = max(possible_portfolios, key=lambda p: p["score"])

    optimized_weights = np.array(list(best_portfolio["weights"].values()))
    return optimized_weights