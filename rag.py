import streamlit as st
import os
import time

# Updated imports for LangChain components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# Import the new official Pinecone package
from pinecone import Pinecone, ServerlessSpec

# ---------- Retrieve API keys and settings ----------
# Try both lowercase and uppercase keys
api_key_openai = st.secrets.get("api_key_openai") or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
api_key_pinecone = st.secrets.get("api_key_pinecone") or st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
directory = st.secrets.get("directory", os.getenv("directory", "./pdfs"))
index_name = st.secrets.get("index_name", os.getenv("index_name", "hr-policies-index"))
# Default to Option A for controller host; update your secret if Option B is needed.
pinecone_controller_host = st.secrets.get("pinecone_controller_host", "https://controller.us-east-1.pinecone.io")
pinecone_region = "us-east-1"  # Adjust if needed

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check your secrets or environment variables.")

# ----------- Environment Setup -----------
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

# Initialize the Pinecone client using the controller endpoint
pc = Pinecone(
    api_key=api_key_pinecone,
    host=pinecone_controller_host,
    spec=ServerlessSpec(cloud='aws', region=pinecone_region)
)

# Attempt to list indexes
try:
    existing_indexes = pc.list_indexes().names()
    print("Existing indexes:", existing_indexes)
except Exception as e:
    raise Exception(f"Error listing indexes. Check your controller host '{pinecone_controller_host}' and API key. {e}")

if index_name not in existing_indexes:
    raise Exception(f"Index '{index_name}' not found. Please create it in your Pinecone dashboard.")

# Retrieve your existing index (query operations will use the proper endpoint configured in Pinecone)
index = pc.Index(index_name)

# ----------- Document Processing Functions -----------
def read_docs(directory):
    """Load all PDFs in the given directory."""
    loader = PyPDFDirectoryLoader(directory)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from {directory}.")
    if docs:
        preview = docs[0].page_content.strip()
        print("Preview of first document (first 500 chars):")
        print(preview[:500])
    else:
        raise ValueError(f"No documents found in directory: {directory}")
    return docs

def chunk_docs(documents, chunk_size=400, chunk_overlap=20):
    """
    Split documents into chunks.
    Using a smaller chunk size in case the document text is short.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s) from the documents.")
    # Filter out empty chunks
    filtered_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"Filtered to {len(filtered_chunks)} non-empty chunk(s).")
    if not filtered_chunks:
        raise ValueError("No non-empty document chunks produced. Check your PDFs and chunking parameters.")
    return filtered_chunks

def get_embeddings():
    """
    Return OpenAI embeddings for text encoding.
    Using the "text-embedding-3-large" model which outputs 3072-dimensional embeddings.
    """
    return OpenAIEmbeddings(model="text-embedding-3-large")

# ----------- Memory Initialization -----------
def get_memory():
    """
    Use a conversation buffer with window memory (last 5 messages).
    Stored in st.session_state to persist across interactions.
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
      - retriever: from Pinecone vectorstore
      - memory: conversation buffer
      - prompts: for condensing follow-up questions and final QA.
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
    2) Create embeddings & build a Pinecone vectorstore.
    3) Create memory & retrieval chain.
    4) Return chain for question-answer usage.
    """
    # Load and chunk docs
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # Create embeddings
    embeddings = get_embeddings()

    # Prepare texts and metadata from chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Build a Pinecone vectorstore using the retrieved index.
    # "text_key" is set to "text" (adjust if needed) and namespace is None.
    vectorstore = LangChainPinecone(index=index, embedding=embeddings, text_key="text", namespace=None)
    vectorstore.add_texts(texts, metadatas)

    # Create memory & retrieval chain
    memory = get_memory()
    chain = create_chain(vectorstore, memory)
    return chain

# ----------- Perform Retrieval -----------
def perform_query(chain, query):
    """
    Provide the user query to the chain and return the structured result.
    """
    result = chain({"question": query})
    return result