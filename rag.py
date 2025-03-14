import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec  # Corrected import
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

############################################
#          PINECONE SETTINGS              #
############################################
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
pinecone_env = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1")
directory = st.secrets.get("directory", os.getenv("PDF_DIRECTORY", "./pdfs"))
index_name = st.secrets.get("index_name", "hr-policies-index")

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

############################################
#          INITIALIZE PINECONE            #
############################################
# Create a Pinecone instance
pc = Pinecone(api_key=api_key_pinecone)

# Ensure index exists
existing_indexes = [index_info.name for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,  # Updated to match Llama embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=pinecone_env  # Ensure the correct region is used
        )
    )

# Correctly fetch the Pinecone index
index = pc.Index(index_name)  # ✅ Correct way to get the index instance

############################################
#        DOCUMENT PROCESSING FUNCTIONS    #
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
    """Return OpenAI embeddings for text encoding (dimension=1024 for llama-text-embed-v2)."""
    return OpenAIEmbeddings(model="llama-text-embed-v2")

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
#         PROMPT TEMPLATES                #
############################################
qa_template = """
You are a helpful QA assistant specialized in the 'Human Rights Policy'.
Use ONLY the context below to answer the question. The context is from the 'Human Rights Policy'.

If the question cannot be answered using the context or is out of scope,
respond EXACTLY with:

"No relevant information found in the Human Rights Policy for your query. Please consult your HR department for more details."

Context:
{context}

Question: {question}
Helpful Answer:
"""
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
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain

############################################
#         MAIN LOGIC FUNCTION             #
############################################
def ask_model():
    """
    1. Load & chunk the PDF(s) from 'directory'.
    2. Create embeddings.
    3. Connect to Pinecone with the correct environment.
    4. Create or update the index with from_documents.
    5. Return a retrieval chain.
    """
    # Load & chunk
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # Embeddings
    embeddings = get_embeddings()

    # Build LangChain Pinecone vectorstore (CORRECTED)
    vectorstore = LangChainPinecone(
        index_name=index_name,  # ✅ Correct: Passing the index name
        embedding=embeddings
    )

    # Create chain with memory
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
