import streamlit as st
from pages.multiple_models import *
from pages.single_model import *
from pages.llm import *

def hide_streamlit_style():
    """Used to hide the Streamlit menu and footer."""
    hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_style, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Portfolio Optimization & Financial ChatBot", page_icon="📈"
    )
    hide_streamlit_style()

    # Sidebar for page selection
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        (
            "Multi-Model Portfolio Optimization",
            "Aziz - The Financial ChatBro",
        ),
    )

    if page == "Multi-Model Portfolio Optimization":
        page_multi()

    elif page == "Aziz - The Financial ChatBro":
        page_llm()


if __name__ == "__main__":
    main()
