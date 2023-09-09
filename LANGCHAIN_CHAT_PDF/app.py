from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
  #  print(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title= "Ask Your PDF")
    st.header("Ask Your PDF 😊")

    pdf = st.file_uploader("Upload your PDF", type="pdf") 

    if pdf is not None: 
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunkspip 
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
 
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            seld_chunk = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = seld_chunk, question = user_question)           
            st.write(response) 

   
if __name__ == '__main__':
    main()
    