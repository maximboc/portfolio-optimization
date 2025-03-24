from yahooquery import search
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt


models_dict = {
    "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)": "SLSQP",
    "Markowitz - GA (Genetic Algorithm)": "GA",
    "Markowitz - CP-SAT (Constraint Programming SAT Solver)": "CP_SAT",
    "CVaR - MINLP (Mixed-Integer Nonlinear Programming)": "MINLP",
}


# Cache the search
@st.cache_data(ttl=3600)
def search_stocks(query):
    if not query or len(query) < 2:
        return []
    try:
        results = search(query)
        return results.get("quotes", [])
    except Exception as e:
        st.error(f"ERROR - search_stocks : {e}")
        return []


def get_ticker_from_name(name):
    try:
        result = search(name)
        if "quotes" in result and len(result["quotes"]) > 0:
            return result["quotes"][0]["symbol"]
    except Exception as e:
        print(f"Error retrieving ticker for {name}: {e}")
    return name


def get_portfolio_return(tickers, weights, start_date, end_date):
    tickers = [
        get_ticker_from_name(ticker.strip()) if not ticker.isalpha() else ticker
        for ticker in tickers
    ]
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)[
        "Adj Close"
    ]
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)

    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    # first one => to plot over time
    # second => final return over the time period
    return cumulative_returns.iloc[-1] * 100, cumulative_returns


def plot_returns(opti, og):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(opti.index, opti * 100, label="Optimized Portfolio", color="blue")
    
    ax.plot(og.index, og * 100, label="Original Portfolio", color="red")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Portfolio Cumulative Returns Over Time")
    ax.legend()
    ax.grid()
    
    # Show plot in Streamlit
    st.pyplot(fig)

def create_bounds(min_weights : list, max_weights : list):
    bounds = []
    for i in range (len(max_weights)):
        if (min_weights[i] > max_weights[i] or max_weights[i] == 0.0):
            bounds.append((0.0, 0.5))
        else:
            bounds.append((min_weights[i], max_weights[i]))
    return bounds
