from datetime import datetime
import numpy as np
from ortools.sat.python import cp_model
from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_risk_free_rate, get_maximum_risk

def cpsat(
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds: list,
    risk_rate: float
):
    """
    Function implementing the Markowitz Model using CP-SAT solver
    
    Args:
        stocks: List of stock symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        bounds: List of tuples with (min_weight, max_weight) for each stock
        risk_rate: Risk-free rate
        
    Returns:
        optimized_weights: Array of optimized portfolio weights
    """
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    
    if adj_close_df.empty:
        print(
            f"ERROR : Stocks {stocks} not found in given range \n \
            with start date : {start_date} \n and End date : {end_date} "
        )
        return []
    
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    
    # Covariance matrix (on trading days)
    cov_matrix = log_returns.cov() * 250.8875
    
    # Expected returns (annualized)
    expected_returns = log_returns.mean() * 252
    
    # Define the scale for integer variables (precision level)
    SCALE = 100  # Use 100 for 1% precision
    
    # Create the CP-SAT model
    model = cp_model.CpModel()
    
    # Create integer variables for weights (scaled by SCALE)
    weight_vars = []
    for i in range(len(stocks)):
        min_w = max(0, int(bounds[i][0] * SCALE))
        max_w = min(SCALE, int(bounds[i][1] * SCALE))
        weight_vars.append(model.new_int_var(min_w, max_w, f"weight_{i}"))
    
    # Constraint: Sum of weights must equal SCALE (i.e., 100%)
    model.add(sum(weight_vars) == SCALE)

    # Calculate return expression (expected portfolio return)
    return_expr = sum(expected_returns[i] * weight_vars[i] for i in range(len(stocks)))
    
    # Create an objective that balances return and risk
    # For risk consideration, use linear constraints to approximate variance
    
    # Find stock pairs with highest positive correlation
    corr_matrix = log_returns.corr()
    corr_pairs = []
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
    
    # Sort by correlation (highest positive first)
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # For highly correlated pairs, limit their combined weight
    top_pos_corr = [p for p in corr_pairs if p[2] > 0.5][:5]  # Top 5 positive correlations > 0.5
    for i, j, corr_val in top_pos_corr:
        # Stronger correlation = tighter limit
        max_combined = int(SCALE * (1.5 - corr_val))  # At corr=1, limit is 50%, at corr=0.5, limit is 100%
        model.add(weight_vars[i] + weight_vars[j] <= max_combined)
    
    # Encourage investment in negatively correlated pairs
    neg_corr_pairs = [p for p in corr_pairs if p[2] < -0.3][:5]  # Top 5 negative correlations < -0.3
    neg_corr_rewards = []
    
    for idx, (i, j, corr_val) in enumerate(neg_corr_pairs):
        # Create variables that represent whether both stocks have significant weight
        threshold = int(0.05 * SCALE)  # 5% threshold
        
        # Binary variables indicating if weight exceeds threshold
        has_i = model.new_bool_var(f"has_{i}")
        has_j = model.new_bool_var(f"has_{j}")
        
        # Enforce relationship with weights
        model.add(weight_vars[i] >= threshold).only_enforce_if(has_i)
        model.add(weight_vars[i] < threshold).only_enforce_if(has_i.not_())
        model.add(weight_vars[j] >= threshold).only_enforce_if(has_j)
        model.add(weight_vars[j] < threshold).only_enforce_if(has_j.not_())
        
        # Reward if both stocks are included
        both_included = model.new_bool_var(f"both_{i}_{j}")
        model.add_bool_and([has_i, has_j]).only_enforce_if(both_included)
        model.add_bool_or([has_i.not_(), has_j.not_()]).only_enforce_if(both_included.not_())
        
        # Calculate reward (stronger negative correlation = higher reward)
        reward = int(abs(corr_val) * 1000)
        neg_corr_rewards.append(reward * both_included)
    
    # Address risk based on the risk_rate parameter
    # Higher risk_rate = more emphasis on return, lower risk_rate = more emphasis on risk
    risk_weight = max(0.1, min(0.9, risk_rate / (max(expected_returns) * 2)))
    
    # Calculate individual stock risks (standard deviations)
    stock_risks = np.sqrt(np.diag(cov_matrix))
    
    # Add constraints for overall portfolio risk
    if risk_weight < 0.5:  # For lower risk preference
        # Sort stocks by risk
        risk_sorted = sorted(range(len(stocks)), key=lambda i: stock_risks[i], reverse=True)
        top_risky = risk_sorted[:min(3, len(stocks))]  # Top 3 riskiest stocks
        
        # Limit allocation to riskiest stocks based on risk_weight
        max_risky_alloc = int(SCALE * (0.3 + risk_weight))  # Between 30% and 80%
        model.add(sum(weight_vars[i] for i in top_risky) <= max_risky_alloc)
    
    # Create the final objective function
    # Scale return based on risk preference
    scaled_return = int(return_expr * 100 * risk_weight)
    
    # Add diversification rewards (for negative correlations)
    diversification_reward = sum(neg_corr_rewards)
    
    # Final objective: maximize return + diversification
    model.maximize(scaled_return + diversification_reward)
    
    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    
    # Set solver parameters
    solver.parameters.max_time_in_seconds = 30  # Adjust as needed
    solver.parameters.num_search_workers = 8  # Adjust based on available cores
    
    # Solve the model
    status = solver.solve(model)
    
    # Extract solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        optimized_weights = np.array([solver.value(weight_vars[i]) / SCALE for i in range(len(stocks))])
        return optimized_weights
    else:
        print(f"Solver status: {solver.status_name(status)}")
        # Fallback to equal weights
        return np.array([1/len(stocks)] * len(stocks))
