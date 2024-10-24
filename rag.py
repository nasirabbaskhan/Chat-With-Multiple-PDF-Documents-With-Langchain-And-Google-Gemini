# importing LLM
from langchain_google_genai import ChatGoogleGenerativeAI
#importing for rag
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# importing for prompt
from langchain.prompts import PromptTemplate
# importing for chaining
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# importing for streamlit
import streamlit as st
# importing for env variables loader
from dotenv import load_dotenv
import os
load_dotenv()


# load google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# 1:RAG
pdf_loader = PyPDFLoader("GenerativeAI.pdf")
pdf_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pdf_docs)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = FAISS.from_documents(documents, embedding=embedding)

retriever = vector_store.as_retriever()




# 2:LLM
llm:ChatGoogleGenerativeAI =  ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,          
)

# response = llm.invoke("what is generative AI")
# print(response)


# 3:prompt
prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure the to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", do not provide the wrong answer
    Context: \n {context}?\n
    Question:\n{question}\n
    
    Answer: 
    
    """ 

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# 4:creating chains of llm, prompt and retriever

# prompt_llm_chain = llm|prompt
prompt_llm_chain = create_stuff_documents_chain(llm, prompt)
# retriever_chain = create_retrieval_chain(retriever, prompt_llm_chain)


# 5:invoking the result
# response = retriever_chain.invoke({"input":"what we learn in Quarter 1"})
# print(response)
# print(response['answer'])

def get_response_from_llm(user_input):
    similarity_data = vector_store.similarity_search(user_input)
    # response = prompt_llm_chain.run({"context":similarity_data, "question":user_input})
    context = "\n".join([doc.page_content for doc in similarity_data])
    
    # Step 3: Use the LLM with the prompt and context to generate the response
    response = prompt_llm_chain({"context": context, "input": user_input})
    
    return context.text
    

# streamlit app
st.set_page_config(page_title="rag demo")
st.header("Rag based application")

user_input= st.text_input("Ask about PIAIC Slabus")

submit = st.button("Ask the question")

if submit and user_input:
    st.subheader("Response is:")
    response = get_response_from_llm(user_input)
    st.write(response)