import streamlit as st
from typing import Set

from backend import run_pydantic_json_convert
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

st.title("json-2-pydantic")
st.sidebar.title("json-2-pydantic")


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

json_obj = st.text_input("Json", placeholder="Enter your json here...")
comments = st.text_input("Comments", placeholder="Enter your comment here...")
submitted = st.button("Submit")

if submitted and len(json_obj) > 0:
    with st.spinner("Generating response..."):
        generated_response = run_pydantic_json_convert(
            json_obj=json_obj, comments=comments
        )

        sources = set(
            [doc.metadata["url"] for doc in generated_response["source_documents"]]
        )
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state.chat_history.insert(
            0, (json_obj, generated_response["result"])
        )
        st.session_state.user_prompt_history.insert(0, json_obj)
        st.session_state.chat_answers_history.insert(0, formatted_response)

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
