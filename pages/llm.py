import os
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime

from dotenv import load_dotenv


def page_llm():
    # st.set_page_config(page_title="Aziz - The Financial ChatBro", page_icon="💸")

    # Load environment variables
    load_dotenv()
    # model_id="mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Ensure we don't reset existing session state variables
    # that might have been set in the main page
    if "num_assets" not in st.session_state:
        st.session_state.num_assets = 5

    if "model" not in st.session_state:
        st.session_state.model = (
            "Markowitz - SLSQP (Sequential Least Squares Quadratic Programming)"
        )

    # Initialize session state for portfolio values if they don't exist
    if "allocations" not in st.session_state:
        st.session_state.allocations = [0.0] * st.session_state.num_assets

    if "min_weights" not in st.session_state:
        st.session_state.min_weights = [0.0] * st.session_state.num_assets

    if "max_weights" not in st.session_state:
        st.session_state.max_weights = [0.0] * st.session_state.num_assets

    if "cur_risk" not in st.session_state:
        st.session_state.cur_risk = 0.0

    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.date(2023, 1, 1)

    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.date(2024, 1, 1)

    def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
        """
        Returns a language model for HuggingFace inference.

        Parameters:
        - model_id (str): The ID of the HuggingFace model repository.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        - temperature (float): The temperature for sampling from the model.

        Returns:
        - llm (HuggingFaceEndpoint): The language model for HuggingFace inference.
        """
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            # token=os.getenv("HF_TOKEN"), => automatically passed
        )
        return llm

    # Configure the Streamlit app
    st.title("Aziz - The Financial ChatBro")
    st.markdown(
        f"*This is a simple chatbot that uses the HuggingFace transformers library to generate responses to your text input. It uses the {model_id}.*"
    )

    # Initialize session state for avatars
    if "avatars" not in st.session_state:
        st.session_state.avatars = {"user": None, "assistant": None}

    # Initialize session state for user text input
    if "user_text" not in st.session_state:
        st.session_state.user_text = None

    # Initialize session state for model parameters
    if "max_response_length" not in st.session_state:
        st.session_state.max_response_length = 2560

    if "system_message" not in st.session_state:
        st.session_state.system_message = "Aziz, The Financial ChatBro 💸, a friendly AI conversing with a human user."

    if "starter_message" not in st.session_state:
        st.session_state.starter_message = "Hello, there! How can I help you today?"

    # Sidebar for settings
    with st.sidebar:
        st.header("System Settings")

        # Model Settings
        st.session_state.max_response_length = 200000

        # Avatar Selection

        st.session_state.avatars["assistant"] = "🤖"
        st.session_state.avatars["user"] = "🗣️"
        # Reset Chat History
        reset_history = st.button("Reset Chat History")

    # Initialize or reset chat history
    if "chat_history" not in st.session_state or reset_history:
        st.session_state.chat_history = [
            {"role": "assistant", "content": st.session_state.starter_message}
        ]

    def get_response(
        system_message,
        chat_history,
        user_text,
        eos_token_id=["User"],
        max_new_tokens=2560,
        get_llm_hf_kws={},
    ):
        """
        Generates a response from the chatbot model.

        Args:
            system_message (str): The system message for the conversation.
            chat_history (list): The list of previous chat messages.
            user_text (str): The user's input text.
            model_id (str, optional): The ID of the HuggingFace model to use.
            eos_token_id (list, optional): The list of end-of-sentence token IDs.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
            get_llm_hf_kws (dict, optional): Additional keyword arguments for the get_llm_hf function.

        Returns:
            tuple: A tuple containing the generated response and the updated chat history.
        """
        # Set up the model
        hf = get_llm_hf_inference(max_new_tokens=max_new_tokens, temperature=0.1)

        # Create the prompt template
        prompt = PromptTemplate.from_template(
            (
                "[INST] {system_message}"
                "\nCurrent Conversation:\n{chat_history}\n\n"
                "\nUser: {user_text}.\n [/INST]"
                "\nAI (Aziz):"
            )
        )
        # Make the chain and bind the prompt
        chat = (
            prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key="content")
        )

        # Generate the response
        response = chat.invoke(
            input=dict(
                system_message=system_message,
                user_text=user_text,
                chat_history=chat_history,
            )
        )
        response = response.split("AI:")[-1]

        # Update the chat history
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": response})
        return response, chat_history

    # Chat interface
    chat_interface = st.container(border=True)
    with chat_interface:
        output_container = st.container()
        st.session_state.user_text = st.chat_input(placeholder="Enter your text here.")

    # Display chat messages
    with output_container:
        # For every message in the history
        for message in st.session_state.chat_history:
            # Skip the system message
            if message["role"] == "system":
                continue

            # Display the chat message using the correct avatar
            with st.chat_message(
                message["role"], avatar=st.session_state["avatars"][message["role"]]
            ):
                st.markdown(message["content"])

        # When the user enter new text:
        if st.session_state.user_text:

            # Display the user's new message immediately
            with st.chat_message("user", avatar=st.session_state.avatars["user"]):
                st.markdown(st.session_state.user_text)

            # Display a spinner status bar while waiting for the response
            with st.chat_message(
                "assistant", avatar=st.session_state.avatars["assistant"]
            ):

                with st.spinner("Thinking..."):
                    # Call the Inference API with the system_prompt, user text, and history
                    response, st.session_state.chat_history = get_response(
                        system_message=st.session_state.system_message,
                        user_text=st.session_state.user_text,
                        chat_history=st.session_state.chat_history,
                        max_new_tokens=st.session_state.max_response_length,
                    )
                    st.markdown(response)
