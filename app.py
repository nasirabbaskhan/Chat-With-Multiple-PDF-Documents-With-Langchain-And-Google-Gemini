# imports for llm
from langchain_google_genai import ChatGoogleGenerativeAI 

# imports for RAG
import PyPDF2  as pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
# importing for prompt
from langchain.prompts import PromptTemplate
# importing for chaining
from langchain.chains.question_answering import load_qa_chain
# import for streamlit app
import streamlit as st
# importing for env variables loader
from dotenv import load_dotenv

import os

load_dotenv()

#load the GOOGLE_API_KEY
google_api_key =  os.getenv("GOOGLE_API_KEY")


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


    

#------------------------------------------------------------------------------------
# load the pdf function
def get_pdf_text(uploaded_file):
    text=""
    reader = pdf.PdfReader(uploaded_file)
    text=""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text+=page.extract_text()
    return text

            
 
# function for text_splitter into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# function for vactor store
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # it will save on my local folder because it can be used when we have needed
    
    
def get_conversational_chain():
    prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure the to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", do not provide the wrong answer.
    Context: \n {context}?\n
    Question:\n{question}\n
    
    Answer: 
    
    """   
    # Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
    llm =  ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  
        temperature=0.2,          
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    
    response = chain(
        {"input_documents":docs, "question":user_question}
        ,return_only_outputs=True
        )
    
    print(response)
    st.subheader("The response is:")
    st.write(response["output_text"]) 

    
#--------------------------------------------------------------------------------------


# streamlit app
def main():
    
    st.set_page_config(page_title="chat with multiple PDF")
    st.header("Chat with multiple PDF using Gemini")

    user_question = st.text_input("Ask the question from the PDF files:")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menue:")
        pdf = st.file_uploader("Upload your PDF files and Click on the submit & process button")
        submit = st.button("submit & process")
        if submit:
            with st.spinner("Processing..."):
                raw_text= get_pdf_text(pdf)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                    
       
if __name__ == "__main__":
    main()