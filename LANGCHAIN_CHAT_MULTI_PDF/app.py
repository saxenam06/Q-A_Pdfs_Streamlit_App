from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template
def get_pdf_text(pdfs_docs):
    text = ""
    for pdf in pdfs_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def handle_userinput(user_question):
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        

    
def main():
    load_dotenv()
    st.set_page_config(page_title= "Chat with Multiple PDFS")
    st.write(css, unsafe_allow_html=True)


    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    st.header("Chat with Multiple PDFS :books:")
    user_question = st.text_input("Ask a question about your documemts:")
    if user_question:
        handle_userinput(user_question)    
   
    with st.sidebar:
        st.subheader("Your documents")
        pdfs_docs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdfs_docs)
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                embeddings = OpenAIEmbeddings()
                #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
                knowledge_base = FAISS.from_texts(text_chunks, embeddings)
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                llm = ChatOpenAI()
                st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm = llm,
                    retriever=knowledge_base.as_retriever(),
                    memory = memory
                    )
         
  

if __name__ == '__main__':
    main()
    