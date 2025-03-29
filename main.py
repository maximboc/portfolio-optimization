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

# Define a callback function to update session state
def update_num_assets():
    # This function now explicitly updates the session state
    # when the number input widget changes
    pass  # The actual update happens automatically through the key

# Use the number input widget with the callback
num_assets = st.sidebar.number_input(
    "Number of assets in portfolio", 
    min_value=1, 
    max_value=20, 
    value=st.session_state.num_assets,  # Explicitly set the value from session state
    step=1,
    key="num_assets_input",  # Use a different key to avoid conflicts
    on_change=update_num_assets
)

# Keep num_assets in sync with the widget
st.session_state.num_assets = num_assets

model_options = (
    "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)",
    "Markowitz - GA (Genetic Algorithm)",
    "Markowitz - CP-SAT (Constraint Programming SAT Solver)",
    "CVaR - MINLP (Mixed-Integer Nonlinear Programming)",
)

model = st.sidebar.selectbox(
    "Optimization Model",
    model_options,
    index=model_options.index(st.session_state.model),  # Set correct initial selection
    key="model_input"  # Use a different key
)

# Keep model in sync with the widget
st.session_state.model = model

st.subheader(st.session_state.model)

# Pass the session state values to init_display
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
    if st.sidebar.button("🤖 Chat with AI"):
        st.switch_page("pages/llm.py")
    st.markdown("</div>", unsafe_allow_html=True)