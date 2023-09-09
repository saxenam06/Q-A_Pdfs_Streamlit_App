from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def main():
    st.set_page_config(page_title= "Your Own GPT")
    st.header("Your Own GPT")

    load_dotenv()
    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
        SystemMessage(content="You are a helpful Assistant")
    ]
    
    with st.sidebar:
        user_input = st.text_input("Your Message:", key = "user_input")

    if user_input:
        st.session_state.messages.append(HumanMessage(content = user_input))
        ai_response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=ai_response.content))

    messages = st.session_state.get('messages',[])
    
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i)+'user') 
        else:
            message(msg.content, is_user=False, key=str(i)+ 'ai')
  

if __name__ == '__main__':
    main()
    