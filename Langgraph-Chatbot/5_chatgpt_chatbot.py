import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(user_input):

    words = user_input.strip().split()

    if len(words) <= 6:
        return user_input.strip()

    return " ".join(words[:6]) + "..."


def add_thread(thread_id, title="New Chat"):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    st.session_state["messages_history"] = []

    add_thread(thread_id, "New Chat")


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================
# Session State
# ============================================================

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = {}


add_thread(
    st.session_state["thread_id"],
    "New Chat"
)


# ============================================================
# Display Current Conversation
# ============================================================

for message in st.session_state["messages_history"]:

    with st.chat_message(message["role"]):
        st.text(message["content"])


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("+ New Chat"):

    reset_chat()

    st.rerun()


st.sidebar.header("My Conversations")


for thread_id, title in st.session_state["chat_threads"].items():

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}"
    ):

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


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input("Type here...")


if user_input:

    # Was this the first message?
    is_new_chat = (
        len(st.session_state["messages_history"]) == 0
    )


    # --------------------------------------------------------
    # Generate chat title
    # --------------------------------------------------------

    if is_new_chat:

        title = generate_chat_title(user_input)

        thread_id = st.session_state["thread_id"]

        st.session_state["chat_threads"][thread_id] = title


    # --------------------------------------------------------
    # Save User Message
    # --------------------------------------------------------

    st.session_state["messages_history"].append({
        "role": "user",
        "content": user_input
    })


    with st.chat_message("user"):

        st.text(user_input)


    # --------------------------------------------------------
    # LangGraph Config
    # --------------------------------------------------------

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # --------------------------------------------------------
    # Generate AI Response
    # --------------------------------------------------------

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

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    yield message_chunk.content


        ai_message = st.write_stream(
            ai_only_stream()
        )


    # --------------------------------------------------------
    # Save Assistant Message
    # --------------------------------------------------------

    st.session_state["messages_history"].append({
        "role": "assistant",
        "content": ai_message
    })