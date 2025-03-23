import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
from utils.markovitz_utils import *

stocks = ['SPY','BND','GLD','QQQ','VTI']
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Create empty DataFrame for adjusted close prices
adj_close_df = pd.DataFrame()

# Download data for each stock individually
for s in stocks:
    data = yf.download(s, start=start_date, end=end_date, auto_adjust=False)
    adj_close_df[s] = data['Adj Close']
    
# Calculate log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()

# Calculate covariance matrix (annualized)
cov_matrix = log_returns.cov() * 252

# Get risk-free rate from FRED
fred = Fred(api_key="e9048dc2c26dae67bc75a443cd644ce3")
ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_rate.iloc[-1]

# Define optimization constraints and bounds
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(stocks))]

# Set initial weights equally distributed
initial_weights = np.array([1/len(stocks)] * len(stocks))

# Optimize portfolio for maximum Sharpe ratio
optimized_results = minimize(negative_sharpe_ratio, initial_weights, 
                            args=(log_returns, cov_matrix, risk_free_rate),
                            method='SLSQP', constraints=constraints, bounds=bounds)

# Get the optimized weights
optimized_weights = optimized_results.x
