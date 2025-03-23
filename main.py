import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from yahooquery import search
from utils.frontend_utils import get_ticker_from_name, get_portfolio_return, search_stocks


st.title("Portfolio Optimization Calculator")

st.sidebar.header("Portfolio Allocation")

num_assets = st.sidebar.number_input("Number of assets in portfolio", min_value=1, max_value=20, value=5, step=1)

# Initialize session state for storing selected tickers and their details
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = [{"symbol": "", "name": ""} for _ in range(num_assets)]

# Adjust the number of selected tickers if num_assets changes
if len(st.session_state.selected_tickers) != num_assets:
    if len(st.session_state.selected_tickers) < num_assets:
        # Add more empty selections
        st.session_state.selected_tickers.extend([{"symbol": "", "name": ""} for _ in range(num_assets - len(st.session_state.selected_tickers))])
    else:
        # Trim excess selections
        st.session_state.selected_tickers = st.session_state.selected_tickers[:num_assets]

tickers = []
allocations = []
min_weights = []
max_weights = []

for i in range(num_assets):
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Search input and autocomplete
    search_query = col1.text_input(f"Search Company {i+1}", "", key=f"search_{i}")
    
    if search_query:
        search_results = search_stocks(search_query)
        if search_results:
            options = [f"{res['symbol']} - {res.get('shortname', res.get('longname', 'Unknown'))}" for res in search_results]
            selected_option = col1.selectbox(
                f"Select stock for position {i+1}", 
                options=[""] + options, 
                index=0, 
                key=f"select_{i}"
            )
            
            if selected_option:
                symbol = selected_option.split(" - ")[0]
                name = selected_option.split(" - ")[1] if " - " in selected_option else ""
                st.session_state.selected_tickers[i] = {"symbol": symbol, "name": name}
        else:
            col1.info("No matching stocks found. Try a different search term.")
    
    # Display currently selected ticker
    current_ticker = st.session_state.selected_tickers[i]
    if current_ticker["symbol"]:
        col1.info(f"Selected: {current_ticker['symbol']} - {current_ticker['name']}")
        tickers.append(current_ticker["symbol"])
    else:
        tickers.append("")
    
    # Allocation inputs
    allocation = col2.number_input(f"Allocation {i+1} (%)", min_value=0.0, max_value=100.0, step=5.0, key=f"alloc_{i}")
    min_weight = col3.number_input(f"Min Weight {i+1} (%)", min_value=0.0, max_value=100.0, step=5.0, key=f"min_{i}")
    max_weight = col4.number_input(f"Max Weight {i+1} (%)", min_value=0.0, max_value=100.0, step=5.0, key=f"max_{i}")
    
    allocations.append(allocation / 100)
    min_weights.append(min_weight / 100)
    max_weights.append(max_weight / 100)

# Check if allocations sum to 100%
total_allocation = sum(allocations) * 100
if total_allocation != 100:
    st.sidebar.warning(f"Current allocation: {total_allocation:.1f}%. Allocations should sum to 100%.")

start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))

# Portfolio summary
st.subheader("Portfolio Summary")
portfolio_df = pd.DataFrame({
    "Ticker": [t["symbol"] for t in st.session_state.selected_tickers],
    "Company": [t["name"] for t in st.session_state.selected_tickers],
    "Allocation (%)": [a * 100 for a in allocations],
    "Min Weight (%)": [m * 100 for m in min_weights],
    "Max Weight (%)": [m * 100 for m in max_weights]
})
st.dataframe(portfolio_df)

if st.sidebar.button("Calculate Return"):
    valid_tickers = [t for t in tickers if t]
    valid_allocations = [allocations[i] for i, t in enumerate(tickers) if t]
    
    if not valid_tickers:
        st.error("Please select at least one stock.")
    elif abs(sum(valid_allocations) - 1.0) > 0.01:
        st.error("Allocations must sum to 100%.")
    else:
        try:
            with st.spinner("Calculating portfolio returns..."):
                total_return = get_portfolio_return(valid_tickers, valid_allocations, start_date, end_date)
                st.success(f"The portfolio return over the selected period is {total_return:.2f}%")
                
                # Optional: Display individual stock returns
                try:
                    stock_data = yf.download(valid_tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
                    stock_returns = ((stock_data.iloc[-1] / stock_data.iloc[0]) - 1) * 100
                    
                    returns_df = pd.DataFrame({
                        "Ticker": valid_tickers,
                        "Return (%)": [stock_returns.get(t, 0) for t in valid_tickers],
                        "Contribution (%)": [stock_returns.get(t, 0) * valid_allocations[i] / 100 for i, t in enumerate(valid_tickers)]
                    })
                    
                    st.subheader("Individual Stock Returns")
                    st.dataframe(returns_df)
                except Exception as e:
                    st.warning(f"Could not calculate individual stock returns: {e}")
                    
        except Exception as e:
            st.error(f"Error calculating returns: {e}")
