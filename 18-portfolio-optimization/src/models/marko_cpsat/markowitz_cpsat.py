from ortools.sat.python import cp_model
import numpy as np
from datetime import datetime
from utils.finance_utils import get_adj_close_from_stocks, get_maximum_risk, get_risk_free_rate
import math

def cpsat(stocks: list,
                 start_date: datetime,
                 end_date: datetime,
                 bounds: list,
                 risk_rate: float,
                 risk_free_rate: float = get_risk_free_rate()):
    """
    Optimize a portfolio using CP-SAT by maximizing a surrogate Sharpe ratio.
    
    The surrogate objective is:
    
         score = risk_rate * (Portfolio Expected Return - Risk-free Rate) 
                 - (1 - risk_rate) * (Portfolio Risk)
    
    where:
         Portfolio Expected Return = sum_i (w_i * expected_return_i)
         Portfolio Risk ≈ sum_i (w_i * std_i)
    
    Each weight w_i is an integer representing a percentage (with sum = 100).
    
    We scale floating-point coefficients to integers (using e.g. 10^4).
    """
    
    # Step 1: Acquire historical adjusted-close data for the stocks
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    
    # Step 2: Calculate daily returns and then compute expected returns and standard deviations per asset.
    returns = adj_close_df.pct_change().dropna()
    expected_returns_series = returns.mean()   # Mean return per asset
    std_series = returns.std()                   # Volatility per asset
    
    # Scaling factors to convert floats to integers
    scale_obj = 10**4   # For expected returns
    scale_risk = 10**4  # For risk (volatility)
    
    # Compute scaled coefficients for expected returns (each asset)
    coeffs = []
    for i in range(len(stocks)):
        coeff = expected_returns_series.iloc[i]
        scaled_coeff = int(round(coeff * scale_obj))
        coeffs.append(scaled_coeff)
        print(f"Coefficient for {stocks[i]} (raw): {coeff}, scaled: {scaled_coeff}")
    
    # Compute scaled risk coefficients from asset volatilities
    risk_coeffs = []
    for i in range(len(stocks)):
        risk_val = std_series.iloc[i]
        scaled_risk = int(round(risk_val * scale_risk))
        risk_coeffs.append(scaled_risk)
        print(f"Risk coefficient for {stocks[i]} (raw): {risk_val}, scaled: {scaled_risk}")
    
    # Scale risk-free rate for the same scale as expected returns.
    # Since weights sum to 100, the risk-free "contribution" will be risk_free_rate * 100.
    risk_free_rate_scaled = int(round(risk_free_rate * scale_obj))
    
    # Convert risk_rate into an integer weight using the same scaling.
    # This gives us a trade-off coefficient: risk_rate_int corresponds to the return part,
    # and (scale_obj - risk_rate_int) corresponds to the risk penalty.
    risk_rate = risk_rate / get_maximum_risk(stocks, start_date, end_date)
    risk_rate_int = int(round(risk_rate * scale_obj))
    non_risk_rate_int = scale_obj - risk_rate_int  # equivalent to (1 - risk_rate)*scale_obj
    
    # Setup CP-SAT model.
    model = cp_model.CpModel()
    
    # Define decision variables: each stock's weight (in integer percentage terms)
    weight_vars = []
    for i, (lower_bound, upper_bound) in enumerate(bounds):
        lower = int(lower_bound * 100)
        upper = int(upper_bound * 100)
        w = model.NewIntVar(lower, upper, f"weight_{i}")
        weight_vars.append(w)
    
    # Constraint: Weights must sum to 100%
    model.Add(sum(weight_vars) == 100)
    
    # Define the portfolio's expected return (scaled) as:
    #     portfolio_return = sum_i (w_i * scaled_expected_return_i)
    portfolio_return = sum(weight_vars[i] * coeffs[i] for i in range(len(stocks)))
    
    # Compute portfolio excess return by subtracting the risk-free rate contribution.
    # Since the weights add to 100, the total risk-free contribution is risk_free_rate_scaled * 100.
    portfolio_excess_return = portfolio_return - (risk_free_rate_scaled * 100)
    
    # Define portfolio risk as the weighted sum of individual risk coefficients.
    portfolio_risk = sum(weight_vars[i] * risk_coeffs[i] for i in range(len(stocks)))
    
    # Our surrogate Sharpe objective (linear combination):
    # Maximize: risk_rate_int * (portfolio_excess_return) - non_risk_rate_int * (portfolio_risk)
    objective = risk_rate_int * portfolio_excess_return - non_risk_rate_int * portfolio_risk
    model.Maximize(objective)
    
    # Solve the model.
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 10
    
    status = solver.Solve(model)
    if status not in [cp_model.OPTIMAL]:
        return {"error": "Aucun portefeuille faisable trouvé"}
    
    # Recover weights (convert back from percentage)
    weights = [solver.Value(w) / 100.0 for w in weight_vars]
    
    # Compute the actual portfolio expected return and risk (using original, unscaled values)
    final_expected_return = sum(weights[i] * expected_returns_series.iloc[i] for i in range(len(stocks)))
    final_risk = sum(weights[i] * std_series.iloc[i] for i in range(len(stocks)))
    
    # Calculate the actual Sharpe ratio if risk is nonzero.
    if final_risk > 0:
        sharpe_ratio = (final_expected_return - risk_free_rate) / final_risk
    else:
        sharpe_ratio = float('-inf')
    
    best_portfolio = {
        "weights": weights,
        "expected_return": final_expected_return,
        "risk": final_risk,
        "sharpe_ratio": sharpe_ratio
    }
    
    return best_portfolio
