import streamlit as st
import os

# Updated imports for LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# ---------- Settings (API Keys and Configurations from Streamlit Secrets) ---------
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
directory = st.secrets.get("directory", os.getenv("PDF_DIRECTORY", "./pdfs"))

if not api_key_openai:
    raise ValueError("Missing OpenAI API key. Check secrets.toml or environment variables.")

# ----------- Environment Setup -----------
os.environ["OPENAI_API_KEY"] = api_key_openai

# ----------- Document Processing Functions -----------
def read_docs(directory):
    """Load all PDFs in the given directory."""
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def get_embeddings():
    """
    Return OpenAI embeddings for text encoding.
    Using the "text-similarity-ada-001" model which outputs 1024-dimensional embeddings.
    """
    return OpenAIEmbeddings(model="text-similarity-ada-001")

# ----------- Memory Initialization -----------
def get_memory():
    """
    Use conversation buffer with window memory (last 5 messages).
    This memory is stored in st.session_state to persist across user interactions.
    """
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
    return st.session_state["memory"]

# ----------- Prompt Templates -----------
condense_template = """
Given the conversation below and a follow-up question, rephrase the follow-up question
to be standalone if it references previous context. If it is unrelated, just use it as-is.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""
condense_question_prompt = PromptTemplate.from_template(condense_template)

qa_template = """
You are a helpful QA assistant focused on the 'Human Rights Policy' below. If the policy context
does not cover the user's question, respond EXACTLY with this text (replace {question} with
the user’s question, but do not modify anything else):

"I didn't find anything in the Human Rights Policy about '{question}'.
Please consult the Employee Handbook or contact HR/Finance for more details."

Context:
{context}

Question: {question}
Helpful Answer:
"""
qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

# ----------- Retrieval Chain Creation -----------
def create_chain(vectorstore, memory):
    """
    Build a ConversationalRetrievalChain:
      - llm: OpenAI
      - retriever: from FAISS vectorstore
      - memory: conversation buffer
      - prompts: condense follow-up and final QA
    """
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain

# ----------- Main Function -----------
def ask_model():
    """
    1) Load & chunk the PDF(s).
    2) Create embeddings & build a FAISS vectorstore.
    3) Create memory & retrieval chain.
    4) Return chain for question-answer usage.
    """
    # Load and chunk docs
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # Create embeddings
    embeddings = get_embeddings()

    # Build a FAISS vectorstore from the document chunks
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Create memory & chain
    memory = get_memory()
    chain = create_chain(vectorstore, memory)
    return chain

# ----------- Perform Retrieval -----------
def perform_query(chain, query):
    """
    Provide the user query to the chain and get the structured result.
    Access the final answer as result["answer"].
    """
    result = chain({"question": query})
    return result