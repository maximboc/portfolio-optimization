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
    """
        Search Tickers from an input
    """
    if not query or len(query) < 2:
        return []
    try:
        results = search(query)
        return results.get("quotes", [])
    except Exception as e:
        st.error(f"ERROR - search_stocks : {e}")
        return []


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
