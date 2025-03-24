import yfinance as yf
import pandas as pd
from fredapi import Fred


def get_adj_close_from_stocks(stocks, start_date, end_date):
    """
        Extract Adjusted Close from mentioned stocks on specific dates
        Adj Close => Closing price adjusted 
                    for splits and dividend distributions
    """
    adj_close_df = pd.DataFrame()
    
    for s in stocks:
        data = yf.download(s, start=start_date, end=end_date, auto_adjust=False)
        adj_close_df[s] = data['Adj Close']
    
    return adj_close_df


def get_risk_free_rate():
    """
        Returns the Risk Free Rate based on Federal
        Risk Free Rate => theoretical rate of return received on zero-risk assets
    """
    fred = Fred(api_key="e9048dc2c26dae67bc75a443cd644ce3")
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    return risk_free_rate

