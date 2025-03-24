import streamlit as st
from front.frontend import init_display

st.title("Portfolio Optimization Calculator")

st.sidebar.header("Portfolio Allocation")

num_assets = st.sidebar.number_input(
    "Number of assets in portfolio", min_value=1, max_value=20, value=5, step=1
)

model = st.sidebar.selectbox(
    "Optimization Model",
    (
        "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)",
        "Markowitz - GA (Genetic Algorithm)",
        "Markowitz - CP-SAT (Constraint Programming SAT Solver)",
        "CVaR - MINLP (Mixed-Integer Nonlinear Programming)"
    ),
)

st.subheader(model)

init_display(num_assets, model)
