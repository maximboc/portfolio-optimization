from yahooquery import search
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

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
    df = pd.DataFrame({
        "Optimized Portfolio": opti * 100,
        "Original Portfolio": og * 100
    }, index=opti.index)
    
    df.index.name = "Date"
    
    st.line_chart(df, use_container_width=True)
