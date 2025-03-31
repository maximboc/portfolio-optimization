import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_risk_free_rate, get_maximum_risk
import streamlit as st

def compute_var(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_rate: int,
):
    """
    Function to compute the Value at Risk (VaR) of a portfolio
    """
    portfolio_return = np.sum(log_returns.mean() * weights) * 250.8875
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(250.8875)

    # VaR calculation
    var = portfolio_return - risk_rate * portfolio_std_dev

    return var

def compute_cvar(weights: np.ndarray,
                 log_returns: pd.DataFrame,
                 confidence_level=0.95):
    """
    Computes Conditional Value at Risk (CVaR) using historical simulation
    """
    # Compute portfolio returns for each day
    portfolio_returns = log_returns @ weights
    
    # Compute the historical VaR at the given confidence level
    var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    
    # Compute CVaR as the average of returns beyond VaR
    cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
    
    return cvar


def minlp(
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds : list,
    risk_rate : int
):
    """
    Function implementing the CVaR MinLP Model
    """
    
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    if adj_close_df.empty:
        print(
            f"ERROR : Stocks {stocks} not found in given range \n \
                with start date : {start_date} \n and End date : {end_date} "
        )
        return []

    constraints = None

    


    optimized_weights = None

    return optimized_weights
