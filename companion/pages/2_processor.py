import streamlit as st
from typing import Set

from backend import run_create_processor
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

st.title("Processors Generator Page 🎉")
st.sidebar.title("Processor Generator Page 🎉")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

comments = st.text_input("Comments", placeholder="Enter your comment here...")
submitted = st.button("Submit")

if submitted and len(comments) > 0:
    with st.spinner("Generating response..."):
        generated_response = run_create_processor(
            comments=comments,
        )

        st.session_state.chat_history.insert(0, (comments, generated_response))
        st.session_state.user_prompt_history.insert(0, comments)
        st.session_state.chat_answers_history.insert(0, generated_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(
            user_query,
            is_user=True,
        )
        message(generated_response)
