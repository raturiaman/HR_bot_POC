import streamlit as st
import os

# UPDATED imports for the new version of LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from pinecone import Pinecone, Index

############################################
#          PINECONE SETTINGS              #
############################################

# Example: Putting environment in secrets.toml
#
#  [secrets]
#  OPENAI_API_KEY = "sk-..."
#  PINECONE_API_KEY = "..."
#  PINECONE_ENVIRONMENT = "us-east-1"
#  directory = "./pdfs"
#  index_name = "hr-policies-index-new"

# ---------- Settings (API Keys from Streamlit Secrets) ---------
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
pinecone_env = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1")
directory = st.secrets.get("directory", os.getenv("PDF_DIRECTORY", "./pdfs"))
index_name = st.secrets.get("index_name", "hr-policies-index-new")

# Ensure API keys are set
if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

############################################
#             ENVIRONMENT SETUP           #
############################################
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

############################################
#      DOCUMENT PROCESSING FUNCTIONS       #
############################################
def read_docs(directory):
    """Load all PDFs in the given directory."""
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings():
    """Return OpenAI embeddings for text encoding (dimension=1536)."""
    return OpenAIEmbeddings()

############################################
#         MEMORY INITIALIZATION           #
############################################
def get_memory():
    """Use conversation buffer memory with a window of 5 messages."""
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
    return st.session_state["memory"]

############################################
#           PROMPT TEMPLATES              #
############################################

# 1) Condense Follow-Up Questions
condense_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question
so it is standalone if it references previous context. If it is unrelated, just return the
question as-is.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(condense_template)

# 2) Strict QA Prompt with fallback
qa_template = """
You are a helpful QA assistant specialized in the 'Human Rights Policy'.
Use ONLY the context below to answer the question. The context is from the 'Human Rights Policy'.

If the question cannot be answered using the context or is out of scope,
respond EXACTLY with:

"No relevant information found in the Human Rights Policy for your query. Please consult your HR department for more details."

Context:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

############################################
#       RETRIEVAL CHAIN CREATION          #
############################################
def create_chain(vectorstore, memory):
    """Build a ConversationalRetrievalChain using an OpenAI LLM."""
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain

############################################
#           MAIN LOGIC FUNCTIONS          #
############################################
def ask_model():
    """
    1. Load & chunk the PDF(s) from 'directory'.
    2. Create embeddings.
    3. Connect to Pinecone with the correct environment.
    4. Create or update the index with from_documents.
    5. Return a retrieval chain.
    """
    # 1) Load & chunk
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # 2) Embeddings
    embeddings = get_embeddings()

    # 3) Pinecone client with environment
    pc = Pinecone(api_key=api_key_pinecone, environment=pinecone_env)

    # 4) Retrieve the correct Pinecone index
    index = pc.Index(index_name)  # Correctly fetches the index instance

    # 5) Build the LangChain Pinecone vectorstore
    vectorstore = LangChainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # 6) Create chain with memory
    memory = get_memory()
    chain = create_chain(vectorstore, memory)
    return chain

############################################
#        PERFORM RETRIEVAL FUNCTION       #
############################################
def perform_query(chain, query):
    """Call the chain with the user's query; returns the chain's result."""
    result = chain({"question": query})
    return result