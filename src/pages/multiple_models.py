import streamlit as st
from front.frontend_multiple import init_multi_model_display


def page_multi():
    # st.set_page_config(page_title="Portfolio Optimization Calculator", page_icon="📈")

    st.title("Portfolio Optimization Calculator")

    st.sidebar.header("Portfolio Allocation")

    num_assets = st.sidebar.number_input(
        "Number of assets in portfolio", min_value=1, max_value=20, value=5, step=1
    )

    st.markdown(
        """
    ## Multi-Model Portfolio Optimization

    This tool allows you to compare different optimization strategies for your investment portfolio.
    Select stocks, set allocations, and compare how different optimization models would adjust your 
    portfolio to maximize returns for a given risk level.
    """
    )

    init_multi_model_display(num_assets)
