import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks
import streamlit as st


def slsqp(
    stocks: list,
    start_date: datetime,
    end_date: datetime,
    bounds : list,
    risk_rate : int
):
    """
    Function implementing the Markowitz Model - SLSQP
    """
    fitness = []
    
    adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
    if adj_close_df.empty:
        print(
            f"ERROR : Stocks {stocks} not found in given range \n \
                with start date : {start_date} \n and End date : {end_date} "
        )
        return []

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    # Covariance matrice (on trading days)
    cov_matrix = log_returns.cov() * 250.8875

    initial_weights = np.array([1 / len(stocks)] * len(stocks))

    optimized_results = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(log_returns, cov_matrix, risk_rate),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        callback=lambda x: fitness.append(sharpe_ratio(x, log_returns, cov_matrix, risk_rate)),
    )

    optimized_weights = optimized_results.x

    return {
        "weights": optimized_weights,
        "fitness": fitness,
        "best_fitness": fitness[-1],
        "time": optimized_results.nit,
    }
