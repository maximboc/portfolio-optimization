import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.finance_utils import create_bounds, get_adj_close_from_stocks, get_maximum_risk, get_portfolio_return, get_risk_free_rate
from .model_selector import select_model
from utils.frontend_utils import (
    plot_returns,
    search_stocks,
    models_dict,
)


def print_portfolio(
    name: str,
    tickers: list,
    allocations: list,
    min_weights: list = [],
    max_weights: list = [],
    bounds=[],
):
    st.subheader(name)
    if bounds:
        min_weights, max_weights = zip(*bounds)
    # Create DataFrame
    portfolio_df = pd.DataFrame(
        {
            "Ticker": [t["symbol"] for t in tickers],
            "Company": [t["name"] for t in tickers],
            "Allocation (%)": [a * 100 for a in allocations],
            "Min Weight (%)": [m * 100 for m in min_weights],
            "Max Weight (%)": [m * 100 for m in max_weights],
        }
    )
    st.dataframe(portfolio_df)


def display_stocks(num_assets: int):
    # Initialize session state for storing selected tickers and their details
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = [
            {"symbol": "", "name": ""} for _ in range(num_assets)
        ]

    # Adjust the number of selected tickers if num_assets changes
    if len(st.session_state.selected_tickers) != num_assets:
        if len(st.session_state.selected_tickers) < num_assets:
            # Add more empty selections
            st.session_state.selected_tickers.extend(
                [
                    {"symbol": "", "name": ""}
                    for _ in range(num_assets - len(st.session_state.selected_tickers))
                ]
            )
        else:
            # Trim excess selections
            st.session_state.selected_tickers = st.session_state.selected_tickers[
                :num_assets
            ]

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
                options = [
                    f"{res['symbol']} - {res.get('shortname', res.get('longname', 'Unknown'))}"
                    for res in search_results
                ]
                selected_option = col1.selectbox(
                    f"Select stock for position {i+1}",
                    options=[""] + options,
                    index=0,
                    key=f"select_{i}",
                )

                if selected_option:
                    symbol = selected_option.split(" - ")[0]
                    name = (
                        selected_option.split(" - ")[1]
                        if " - " in selected_option
                        else ""
                    )
                    st.session_state.selected_tickers[i] = {
                        "symbol": symbol,
                        "name": name,
                    }
            else:
                col1.info("No matching stocks found. Try a different search term.")

        # Display currently selected ticker
        current_ticker = st.session_state.selected_tickers[i]
        if current_ticker["symbol"]:
            col1.info(
                f"Selected: {current_ticker['symbol']} - {current_ticker['name']}"
            )
            tickers.append(current_ticker["symbol"])
        else:
            tickers.append("")

        # Allocation inputs
        allocation = col2.number_input(
            f"Allocation {i+1} (%)",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=f"alloc_{i}",
        )
        min_weight = col3.number_input(
            f"Min Weight {i+1} (%)",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=f"min_{i}",
        )
        max_weight = col4.number_input(
            f"Max Weight {i+1} (%)",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key=f"max_{i}",
        )

        allocations.append(allocation / 100)
        min_weights.append(min_weight / 100)
        max_weights.append(max_weight / 100)

    # Check if allocations sum to 100%
    total_allocation = sum(allocations) * 100
    if total_allocation != 100:
        st.sidebar.warning(
            f"Current allocation: {total_allocation:.1f}%. Allocations should sum to 100%."
        )

    today = datetime.date.today()
    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1), max_value=today)
    end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1), max_value=today)
    if (start_date > end_date):
        st.sidebar.warning("Start date should be before End Date.")

    print_portfolio(
        name="Portfolio Summary",
        tickers=st.session_state.selected_tickers,
        allocations=allocations,
        min_weights=min_weights,
        max_weights=max_weights,
    )

    return tickers, allocations, start_date, end_date, min_weights, max_weights


def plot_multi_model_returns(models_returns, original_returns=None):
    """
    Plot returns for multiple models plus original allocation if provided
    
    Args:
        models_returns: Dictionary with model names as keys and returns time series as values
        original_returns: Optional time series of original allocation returns
    """
    # Create a combined DataFrame for all models
    combined_df = pd.DataFrame()
    
    # Add each model's returns
    for model_name, returns in models_returns.items():
        combined_df[model_name] = returns
    
    # Add original allocation if provided
    if original_returns is not None:
        combined_df["Original Allocation"] = original_returns
    
    st.subheader("Portfolio Performance Comparison")
    st.line_chart(combined_df)
    st.caption("Date vs Cumulative Return (%)")

def compare_model_metrics(models_data):
    """
    Create a comparison table of model metrics
    
    Args:
        models_data: Dictionary with model names as keys and metric dictionaries as values
    """
    metrics_df = pd.DataFrame(columns=["Model", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Volatility (%)", "Risk (%)"])
    
    for model_name, metrics in models_data.items():
        new_row = pd.DataFrame({
            "Model": [model_name],
            "Total Return (%)": [metrics.get("return", 0)],
            "Sharpe Ratio": [metrics.get("sharpe", 0)],
            "Max Drawdown (%)": [metrics.get("max_drawdown", 0)],
            "Volatility (%)": [metrics.get("volatility", 0)],
            "Risk (%)": [metrics.get("risk", 0)]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df)
    
    # Create bar chart comparison for total return
    st.subheader("Total Return Comparison")
    chart_data = metrics_df[["Model", "Total Return (%)"]].set_index("Model")
    st.bar_chart(chart_data)


def calculate_model_metrics(returns_series, risk):
    """
    Calculate various performance metrics from a returns time series
    
    Args:
        returns_series: Time series of portfolio returns
        risk: Risk level used for the model
        
    Returns:
        Dictionary of metrics
    """
    # Calculate daily returns
    daily_returns = returns_series.pct_change().dropna()
    
    # Total return
    total_return = (returns_series.iloc[-1] / returns_series.iloc[0] - 1) * 100
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming risk-free rate from the function)
    risk_free = get_risk_free_rate()
    sharpe = (total_return/100 - risk_free) / (volatility/100)
    
    # Maximum drawdown
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    return {
        "return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "risk": risk * 100
    }


def init_multi_model_display(num_assets: int):
    # First get all the stock data and allocations
    tickers, allocations, start_date, end_date, min_weights, max_weights = display_stocks(num_assets)
    
    # Get risk-free rate and max risk
    risk_free_rate = get_risk_free_rate()
    cur_tickers = [t for t in tickers if t]
    
    if not cur_tickers:
        st.error("Please select at least one stock.")
        return
    
    max_risk = get_maximum_risk(cur_tickers, start_date, end_date)
    
    # Allow selection of risk level
    cur_risk = st.sidebar.number_input(
        "Choose risk rate", 
        min_value=risk_free_rate * 100, 
        max_value=max_risk * 100, 
        value=risk_free_rate * 100, 
        step=0.1
    )
    cur_risk /= 100
    
    st.sidebar.write(
        f"For your portfolio, the minimum risk is {risk_free_rate:.2%} and the maximum risk is {max_risk:.2%}."
    )
    
    st.sidebar.subheader("Select Models to Compare")
    selected_models = {}
    for model_name in models_dict.keys():
        selected_models[model_name] = st.sidebar.checkbox(f"Include {model_name}", model_name == "Sharpe Ratio")
    
    selected_model_names = [name for name, selected in selected_models.items() if selected]
    
    if not selected_model_names:
        st.sidebar.warning("Please select at least one model for comparison")
        return
    
    if st.sidebar.button("Calculate and Compare Models"):
        final_tickers = [t for t in tickers if t]
        final_allocations = [allocations[i] for i, t in enumerate(tickers) if t]
        
        if not final_tickers:
            st.error("Please select at least one stock.")
            if start_date > end_date:
                st.error("Start date should be before End Date.")
                
        elif abs(sum(final_allocations) - 1.0) > 0.01:
            st.error("Allocations must sum to 100%.")
            if start_date > end_date:
                st.error("Start date should be before End Date.")
        elif start_date > end_date:
            st.error("Start date should be before End Date.")
        else:
            bounds = create_bounds(min_weights, max_weights)
            
            og_return, og_returns_time = get_portfolio_return(
                final_tickers, final_allocations, start_date, end_date
            )
            
            st.subheader("Model Performance Comparison")
            
            model_returns = {}
            model_metrics = {}
            model_allocations = {}
            
            progress_bar = st.progress(0)
            progress_step = 1 / len(selected_model_names)
            progress_value = 0
            
            for i, model_name in enumerate(selected_model_names):
                with st.spinner(f"Calculating {model_name} model..."):
                    model = models_dict[model_name]
                    
                    my_model_dict = select_model(
                        model, final_tickers, start_date, end_date, bounds, cur_risk
                    )
                    print(my_model_dict)
                    # Calculate the returns
                    opti_return, opti_returns_time = get_portfolio_return(
                        final_tickers, my_model_dict["weights"], start_date, end_date
                    )
                    
                    # Store results
                    model_returns[model_name] = opti_returns_time
                    model_metrics[model_name] = calculate_model_metrics(opti_returns_time, cur_risk)
                    model_allocations[model_name] = my_model_dict["weights"]
                    
                    # Update progress
                    progress_value += progress_step
                    progress_bar.progress(min(progress_value, 1.0))
            
            # Clear progress bar
            progress_bar.empty()
            
            # Plot returns for all models
            st.subheader("📈 Cumulative Returns Comparison")
            plot_multi_model_returns(model_returns, og_returns_time)
            
            # Show metrics comparison
            compare_model_metrics(model_metrics)
            
            # Display optimal allocations for each model
            st.subheader("Optimal Allocations by Model")
            
            # Create tabs for each model's allocation
            tabs = st.tabs(selected_model_names)
            for i, (model_name, tab) in enumerate(zip(selected_model_names, tabs)):
                with tab:
                    print_portfolio(
                        name=f"{model_name} Optimal Allocation",
                        tickers=st.session_state.selected_tickers,
                        allocations=model_allocations[model_name],
                        bounds=bounds,
                    )
            
            # Display original allocation as a separate tab
            original_tab = st.expander("Original Allocation")
            with original_tab:
                st.write(f"Original allocation return: {og_return:.2f}%")
                print_portfolio(
                    name="Original Portfolio",
                    tickers=st.session_state.selected_tickers,
                    allocations=final_allocations,
                    min_weights=min_weights,
                    max_weights=max_weights,
                )
            
            # Stock performance
            st.subheader("📈 Individual Stock Returns Over the Selected Period")
            stock_prices = get_adj_close_from_stocks(final_tickers, start_date, end_date)
            stock_returns = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100
            st.dataframe(stock_returns.to_frame(name="Return (%)"))
            st.line_chart(stock_prices / stock_prices.iloc[0] * 100, use_container_width=True)
                
                
    st.sidebar.markdown(
        "**⚠️ Past performances<br>cannot predict the future.**", 
        unsafe_allow_html=True
    )

