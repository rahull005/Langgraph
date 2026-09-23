import streamlit as st
from Langgraph_backend_db_6 import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ***************************************** Utility Functions **************************

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["messages_history"] = []

    add_thread(thread_id)

    st.rerun()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# **************************************** Session Setup ******************************

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(st.session_state["thread_id"])


# **************************************** Sidebar UI *********************************

st.sidebar.title("Langgraph_chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")


for thread_id in st.session_state["chat_threads"]:

    if st.sidebar.button(str(thread_id)):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_msgs = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_msgs.append({
                "role": role,
                "content": msg.content
            })

        st.session_state["messages_history"] = temp_msgs

        st.rerun()


# **************************************** Main UI ************************************

for messages in st.session_state["messages_history"]:

    with st.chat_message(messages["role"]):
        st.text(messages["content"])


user_input = st.chat_input("Type here")


if user_input:

    st.session_state["messages_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.text(user_input)


    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }


    with st.chat_message("assistant"):

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                if isinstance(message_chunk, AIMessage):

                    if message_chunk.content:
                        yield message_chunk.content


        ai_message = st.write_stream(
            ai_only_stream()
        )


    st.session_state["messages_history"].append({
        "role": "assistant",
        "content": ai_message
    })