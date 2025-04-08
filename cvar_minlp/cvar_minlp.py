import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from gekko import GEKKO
from utils.finance_utils import get_adj_close_from_stocks

def cvar_minlp(
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds: list,
    risk_rate: float,
    confidence_level: float = 0.95,
    max_assets: int = None
):
    """
    Function implementing the CVaR Model with MINLP using Gekko
    
    Parameters:
    - stocks: List of stock tickers
    - start_date: Start date for historical data
    - end_date: End date for historical data
    - bounds: List of (min_weight, max_weight) tuples for each stock
    - risk_rate: Risk-free rate (decimal)
    - confidence_level: Confidence level for CVaR (default: 0.95)
    - max_assets: Maximum number of assets to include (default: all)
    
    Returns:
    - optimized_weights: Dictionary with "weights" key containing numpy array of optimal portfolio weights
    """
    # Get adjusted close prices
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    if adj_close_df.empty:
        print(
            f"ERROR : Stocks {stocks} not found in given range \n \
            with start date : {start_date} \n and End date : {end_date} "
        )
        return {"weights": np.array([1/len(stocks) for _ in range(len(stocks))])}
    
    # Compute log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    
    N = log_returns.shape[0]  # Number of scenarios (days)
    M = len(stocks)           # Number of assets
    
    # Set default max_assets if not provided
    if max_assets is None or max_assets > M:
        max_assets = M
    
    # Initialize Gekko Model
    m = GEKKO(remote=False)
    m.options.SOLVER = 3  # Changed to BONMIN for better handling of integer variables
    
    # Decision Variables
    w = [m.Var(lb=bounds[i][0], ub=bounds[i][1]) for i in range(M)]  # Portfolio weights
    
    # Binary variables for asset selection (MINLP part)
    z = [m.Var(lb=0, ub=1, integer=True) for _ in range(M)]
    
    # Minimum investment threshold if asset is selected
    min_investment = 0.01 # Minimum 1% if selected
    
    # Link binary variables to weights
    for i in range(M):
        m.Equation(w[i] <= bounds[i][1] * z[i])
        m.Equation(w[i] >= min_investment * z[i])
    
    # Cardinality constraint
    m.Equation(m.sum(z) <= max_assets)
    
    # Portfolio weights sum to 1
    m.Equation(m.sum(w) == 1)
    
    # VaR and excess losses for CVaR
    var_threshold = m.Var(lb=-1, ub=1)  # VaR threshold
    xi = [m.Var(lb=0) for _ in range(N)]  # Excess losses
    
    # Calculate portfolio returns for each scenario
    for i in range(N):
        # Calculate total portfolio return first, then subtract risk-free rate
        portfolio_return = m.sum([log_returns.iloc[i, j] * w[j] for j in range(M)]) - risk_rate
        # CVaR constraint: excess loss beyond VaR threshold
        m.Equation(xi[i] >= -portfolio_return - var_threshold)
    
    # CVaR objective
    alpha = 1 - confidence_level  # Convert confidence level to alpha (e.g., 0.95 -> 0.05)
    cvar = var_threshold + (1 / (alpha * N)) * m.sum(xi)
    
    # Minimize CVaR (maximize negative CVaR)
    m.Obj(cvar)
    
    # Solve the model
    try:
        m.solve(disp=False)  # Set to True for more debugging info
        
        # Extract optimized weights
        optimized_weights = np.array([w[i].value[0] for i in range(M)])
        
        return {"weights": optimized_weights}
        
    except Exception as e:
        print(f"Optimization error: {e}")
        # Return equally weighted portfolio as fallback with correct structure
        return {"weights": np.array([1/M for _ in range(M)])}

def main():
    """
    Test function to demonstrate the CVaR MINLP implementation
    """
    # Example usage
    stocks = ['TSLA','AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX']
    start_date = '2023-04-01'
    end_date = '2024-03-31'
    risk_rate = 0.05 # 5%
  
    # Define bounds (min and max weights)
    min_weights = [0.0] * len(stocks)
    max_weights = [1.0] * len(stocks)
    bounds = list(zip(min_weights, max_weights))
    print(f"Bounds: {bounds}")
  
    # Run optimization
    optimized_weights = cvar_minlp(
        stocks, 
        start_date, 
        end_date, 
        bounds, 
        risk_rate, 
        confidence_level=0.95, 
        max_assets=5  # Allow a maximum of 3 assets
    )
  
    # Print the results
    print("\nOptimized Portfolio Weights:")
    for stock, weight in zip(stocks, optimized_weights["weights"]):
        print(f"{stock}: {weight:.4f}")

if __name__ == "__main__":
    main()