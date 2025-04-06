import streamlit as st
import pandas as pd
import datetime

from utils.finance_utils import (
    create_bounds,
    get_adj_close_from_stocks,
    get_maximum_risk,
    get_portfolio_return,
    get_risk_free_rate,
)
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
    start_date = st.sidebar.date_input(
        "Start Date", datetime.date(2023, 1, 1), max_value=today
    )
    end_date = st.sidebar.date_input(
        "End Date", datetime.date(2024, 1, 1), max_value=today
    )
    if start_date > end_date:
        st.sidebar.warning("Start date should be before End Date.")

    print_portfolio(
        name="Portfolio Summary",
        tickers=st.session_state.selected_tickers,
        allocations=allocations,
        min_weights=min_weights,
        max_weights=max_weights,
    )

    return tickers, allocations, start_date, end_date, min_weights, max_weights


def init_display(num_assets: int, model: str):

    tickers, allocations, start_date, end_date, min_weights, max_weights = (
        display_stocks(num_assets)
    )

    model = models_dict[model]

    risk_free_rate = get_risk_free_rate()
    cur_tickers = [t for t in tickers if t]

    max_risk = get_maximum_risk(cur_tickers, start_date, end_date)

    cur_risk = st.sidebar.number_input(
        "Choose risk rate",
        min_value=risk_free_rate * 100,
        max_value=max_risk * 100,
        value=risk_free_rate * 100,
        step=0.1,
    )
    cur_risk /= 100

    st.sidebar.write(
        f"For your portfolio, the minimum risk is {risk_free_rate:.2%} and the maximum risk is {max_risk:.2%}."
    )

    if st.sidebar.button("Calculate Return"):
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
            try:
                with st.spinner("Calculating portfolio returns..."):
                    bounds = create_bounds(min_weights, max_weights)
                    opti_weights = select_model(
                        model, final_tickers, start_date, end_date, bounds, cur_risk
                    )
                    print(final_tickers)
                    
                    opti_return, opti_returns_time = get_portfolio_return(
                        final_tickers, opti_weights, start_date, end_date
                    )

                    og_return, og_returns_time = get_portfolio_return(
                        final_tickers, final_allocations, start_date, end_date
                    )
                    print_portfolio(
                        name="Optimized Porftolio",
                        tickers=st.session_state.selected_tickers,
                        allocations=opti_weights,
                        bounds=bounds,
                    )

                    plot_returns(opti_returns_time, og_returns_time)
                    st.success(
                        f"Your return : {og_return:.2f}% \n Optimized Portfolio return : {opti_return:.2f}%"
                    )

                    st.subheader("📈 Individual Stock Returns Over the Selected Period")

                    stock_prices = get_adj_close_from_stocks(
                        final_tickers, start_date, end_date
                    )

                    stock_returns = (
                        stock_prices.iloc[-1] / stock_prices.iloc[0] - 1
                    ) * 100

                    st.dataframe(stock_returns.to_frame(name="Return (%)"))

                    st.line_chart(
                        stock_prices / stock_prices.iloc[0] * 100,
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error calculating returns: {e}")
    st.sidebar.markdown(
        "**⚠️ Past performances<br>cannot predict the future.**", unsafe_allow_html=True
    )
