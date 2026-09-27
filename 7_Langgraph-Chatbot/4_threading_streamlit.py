import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage,AIMessage
import uuid


#***************************************** Utility Functions **************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id



def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['messages_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])



# **************************************** Session Setup ******************************
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []


for messages in st.session_state["messages_history"]:
    with st.chat_message(messages['role']):   #this just creates a text-box with role logo
        st.text(messages['content'])


if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()


if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************


st.sidebar.title("Langgraph_chatbot")
if st.sidebar.button("new chat"):
    reset_chat()
st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_msgs = []
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_msgs.append({'role':role,'content':msg.content})
        st.session_state['messages_history'] = temp_msgs



# **************************************** Main UI ************************************

user_input = st.chat_input("type here")
if user_input:
    st.session_state["messages_history"].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {"configurable":{"thread_id":st.session_state["thread_id"]}}
    
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['messages_history'].append({'role': 'assistant', 'content': ai_message})