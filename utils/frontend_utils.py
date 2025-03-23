from yahooquery import search
import streamlit as st
import yfinance as yf

def get_ticker_from_name(name):
    try:
        result = search(name)
        if 'quotes' in result and len(result['quotes']) > 0:
            return result['quotes'][0]['symbol']
    except Exception as e:
        print(f"Error retrieving ticker for {name}: {e}")
    return name 

# Cache the search results to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def search_stocks(query):
    if not query or len(query) < 2:
        return []
    try:
        results = search(query)
        return results.get("quotes", [])
    except Exception as e:
        st.error(f"Error searching stocks: {e}")
        return []

def get_portfolio_return(tickers, weights, start_date, end_date):
    tickers = [get_ticker_from_name(ticker.strip()) if not ticker.isalpha() else ticker for ticker in tickers]
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_return = (1 + portfolio_returns).prod() - 1
    return cumulative_return * 100
