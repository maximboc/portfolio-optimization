import streamlit as st
from front.frontend import init_display

st.title("Portfolio Optimization Calculator")
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}  /* Hides the page navigation */
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("Portfolio Allocation")

# Initialize session state values if they don't exist
if 'num_assets' not in st.session_state:
    st.session_state.num_assets = 5
    
if 'model' not in st.session_state:
    st.session_state.model = "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)"

# Generate a unique ID for the portfolio
if 'portfolio_id' not in st.session_state:
    st.session_state.portfolio_id = "portfolio_" + str(hash(str(st.session_state)))

# Fix the number input widget
num_assets = st.sidebar.number_input(
    "Number of assets in portfolio", 
    min_value=1, 
    max_value=20, 
    value=st.session_state.num_assets, 
    step=1,
    key="num_assets"  # This automatically updates session_state.num_assets
)

model_options = (
    "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)",
    "Markowitz - GA (Genetic Algorithm)",
    "Markowitz - CP-SAT (Constraint Programming SAT Solver)",
    "CVaR - MINLP (Mixed-Integer Nonlinear Programming)",
)

model = st.sidebar.selectbox(
    "Optimization Model",
    model_options,
    index=model_options.index(st.session_state.model),
    key="model"  # This automatically updates session_state.model
)

st.subheader(st.session_state.model)

# You might need to modify init_display to store additional form values
init_display(st.session_state.num_assets, st.session_state.model)

st.markdown(
    """
    <style>
        .chat-button-container {
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 1000;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Create a container with the fixed position
with st.container():
    st.markdown('<div class="chat-button-container">', unsafe_allow_html=True)
    if st.button("🤖 Chat with AI"):
        st.switch_page("pages/llm.py")
    st.markdown("</div>", unsafe_allow_html=True)
