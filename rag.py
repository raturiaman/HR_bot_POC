import streamlit as st
import os

########################
# LangChain & Community
########################
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone

from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

########################
# Official Pinecone
########################
from pinecone import Pinecone, ServerlessSpec

########################
# API Keys & Settings
########################
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
pinecone_env = st.secrets.get("PINECONE_ENVIRONMENT", os.getenv("PINECONE_ENV", ""))
directory = st.secrets.get("directory", "./pdfs")
index_name = st.secrets.get("index_name", "hr-policies-index")

# Ensure API keys are set
if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone
if pinecone_env:
    os.environ["PINECONE_ENV"] = pinecone_env

########################
# Document & Embedding
########################

def read_docs(pdf_directory):
    """
    Loads all PDFs in the given directory using PyPDFDirectoryLoader.
    """
    loader = PyPDFDirectoryLoader(pdf_directory)
    return loader.load()

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    """
    Splits the loaded docs into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def get_embeddings():
    return OpenAIEmbeddings()  # text-embedding-ada-002 by default

########################
# Memory
########################

def get_memory():
    """
    Creates or retrieves a conversation buffer memory from st.session_state.
    This stores the last 5 messages for ChatGPT-like context.
    """
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
    return st.session_state["memory"]

########################
# Prompt Templates
########################

condense_template = """
Given the conversation below and a follow-up question, rephrase the follow-up question
to be a standalone question if it references prior context. If it's unrelated, answer directly.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""
condense_question_prompt = PromptTemplate.from_template(condense_template)


qa_template = """
You are a helpful QA assistant specialized in the 'Human Rights Policy'. 
Use ONLY the context below to answer the question. The context is from the 'Human Rights Policy'.

If the question cannot be answered using the context (for example, it's about politics or budgets),
respond EXACTLY with:

"No relevant information found in the Human Rights Policy for your query. Please consult your HR department for more details."

Context:
{context}

Question: {question}
Helpful Answer:
"""
qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

########################
# Create Retrieval Chain
########################
def create_chain(vectorstore, memory):
    """
    Build a ConversationalRetrievalChain with:
    - OpenAI LLM
    - The Pinecone-based retriever
    - Conversation memory
    - Condense follow-up prompt
    - Strict QA fallback prompt
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

########################
# Manual Upsert to Pinecone
########################

def upsert_chunks_to_pinecone(chunks, embeddings, pc, index_name):
    """
    1) Create the Pinecone index if it doesn't exist
    2) Embed and upsert chunked text into the index
    3) This yields a ready-to-use index with doc embeddings
    """
    # Check or create index
    existing = pc.list_indexes().names()
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' ...")
        # match embedding dimension (1536 for ada-002)
        dimension = 1536  
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west4")  # or your region
        )
        # Wait for index creation
        import time
        while index_name not in pc.list_indexes().names():
            print(f"Waiting for {index_name} to be ready...")
            time.sleep(3)

    # Connect to the newly created or existing index
    pinecone_index = pc.Index(index_name)

    # Prepare data to upsert
    # chunk.page_content is your text, chunk.metadata is optional
    import uuid
    vectors_to_upsert = []
    for chunk in chunks:
        # embed the text
        text_embedding = embeddings.embed_query(chunk.page_content)
        # generate a unique ID for each chunk
        chunk_id = str(uuid.uuid4())
        # store metadata if you want
        meta = chunk.metadata if chunk.metadata else {}
        # add to batch
        vectors_to_upsert.append(
            (chunk_id, text_embedding, meta)
        )

    # Upsert in Pinecone
    print(f"Upserting {len(vectors_to_upsert)} chunks into Pinecone index '{index_name}'...")
    pinecone_index.upsert(vectors_to_upsert)

########################
# Main Function
########################

def ask_model():
    """
    1) Load & chunk PDF(s)
    2) Embed & upsert them to Pinecone manually
    3) Connect to the existing Pinecone index with from_existing_index
    4) Create chain from the vectorstore & memory
    """
    # 1) Load docs & chunk them
    docs = read_docs(directory)
    chunks = chunk_docs(docs)
    embeddings = get_embeddings()

    # 2) Initialize new Pinecone client
    pc = Pinecone(api_key=api_key_pinecone)
    # Upsert chunked data to Pinecone index
    upsert_chunks_to_pinecone(chunks, embeddings, pc, index_name)

    # 3) Connect to the existing index from LangChain
    vectorstore = LangChainPinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # 4) Create chain with memory
    memory = get_memory()
    chain = create_chain(vectorstore, memory)
    return chain

def perform_query(chain, query):
    """
    Provide the user query to the chain and return the structured result.
    """
    result = chain({"question": query})
    return result