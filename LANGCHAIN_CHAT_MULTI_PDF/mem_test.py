from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE  

def main():
    load_dotenv()
    llm = ChatOpenAI()
    conversation = ConversationChain(
        llm=llm, 
        memory= ConversationEntityMemory(llm=llm), 
        prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        verbose=True)
    print("Hello I am ChatGPT CLI")

    while True:
        user_input = input("> ")
        ai_response = conversation.predict(input = user_input)
        print("\nAI message :\n", ai_response)
  

if __name__ == '__main__':
    main()
    