import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
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
api_key_openai = st.secrets["OPENAI_API_KEY"]
api_key_pinecone = st.secrets["PINECONE_API_KEY"]
directory = st.secrets["directory"]
index_name = st.secrets["index_name"]

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml.")

############################################
#          INITIALIZE PINECONE            #
############################################
pc = Pinecone(api_key=api_key_pinecone)

# Ensure the index exists; if not, create it
existing_indexes = [index_info.name for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches the embedding model's output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Get the Pinecone index instance
index = pc.Index(index_name)

############################################
#        DOCUMENT PROCESSING FUNCTIONS    #
############################################
def read_docs(directory):
    """
    Load all PDF documents from the specified directory.
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in directory '{directory}'.")
    return documents

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings():
    """
    Return OpenAI embeddings for text encoding.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")

############################################
#         MEMORY INITIALIZATION           #
############################################
def get_memory():
    """
    Use conversation buffer memory with a window of 5 messages.
    """
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
    """
    Build a ConversationalRetrievalChain using an OpenAI LLM.
    """
    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(api_key=api_key_openai),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        chain_type_kwargs={"prompt": qa_prompt},
        verbose=True
    )
    return chain

############################################
#         MAIN LOGIC FUNCTION             #
############################################
def ask_model():
    """
    1. Load and chunk the PDF(s) from the given directory.
    2. Create embeddings for the document chunks.
    3. Connect to Pinecone and update the index.
    4. Create and return a retrieval chain.
    """
    try:
        # Load documents and split into chunks
        docs = read_docs(directory)
        chunks = chunk_docs(docs)
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Build the LangChain Pinecone vectorstore from document chunks
        vectorstore = LangChainPinecone.from_documents(
            documents=chunks,
            embedding=embeddings,
            index=index
        )
        
        # Retrieve conversation memory
        memory = get_memory()
        
        # Create and return the retrieval chain
        chain = create_chain(vectorstore, memory)
        return chain
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the model: {e}")

############################################
#        PERFORM RETRIEVAL FUNCTION       #
############################################
def perform_query(chain, query):
    """
    Call the retrieval chain with the user's query and return the chain's result.
    """
    try:
        result = chain({"question": query})
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to perform query: {e}")