import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage


if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

for message in st.session_state["messages_history"]:  #dict(message_history : [{}])
    with st.chat_message(message['role']):
        st.text(message['content'])

"""

    [
        {
            "role": "user",
            "content": "What is LangGraph?"
        },
        {
            "role": "assistant",
            "content": "LangGraph is a framework for building stateful agent workflows."
        }
    ]

"""


user_input = st.chat_input("type here")
if user_input:
    st.session_state['messages_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                {"configurable":{"thread_id":"thread_1"}},
                stream_mode="messages"
            )
        )
        st.session_state['messages_history'].append({'role': 'assistant', 'content': ai_message})
