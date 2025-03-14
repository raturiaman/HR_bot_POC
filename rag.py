import streamlit as st
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from pinecone import Pinecone as PineconeClient

# ---------- Settings (API Keys from Streamlit Secrets) ---------
api_key_openai = st.secrets["OPENAI_API_KEY"]
api_key_pinecone = st.secrets["PINECONE_API_KEY"]
directory = st.secrets["directory"]
index_name = "hr-policies-index"

# ----------- Environment Setup -----------
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

# ----------- Document Processing Functions -----------
def read_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def get_embeddings():
    return OpenAIEmbeddings()

# ----------- Memory Initialization -----------
def get_memory():
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    return st.session_state["memory"]

# ----------- Prompt Templates -----------
condense_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question without changing its content. If the question is not related to previous context, answer directly.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(condense_template)

qa_template = """
You are a helpful QA assistant specialized in HR Policies. Answer questions based on the given context. If you don't know, clearly say you don't know. Don't fabricate answers.

Context:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

# ----------- Retrieval Chain Creation -----------
def create_chain(vectorstore, memory):
    return ConversationalRetrievalChain.from_llm(
        OpenAI(),
        vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )

# ----------- Main Function -----------
def ask_model():
    docs = read_docs(directory)
    chunks = chunk_docs(docs)
    embeddings = OpenAIEmbeddings()

    # Initialize Pinecone client
    pc = PineconeClient(api_key=api_key_pinecone)

    # LangChain Pinecone wrapper (Correct Usage)
    vectorstore = LangChainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    memory = get_memory()
    chain = create_chain(vectorstore, memory)

    return chain

# ----------- Perform Retrieval -----------
def perform_query(chain, query):
    result = chain({"question": query})
    return result
