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

# ---------- Settings (API Keys and Configurations from Streamlit Secrets) ---------
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
directory = st.secrets.get("directory", os.getenv("PDF_DIRECTORY", "./pdfs"))
index_name = st.secrets.get("index_name", "hr-policies-index")
pinecone_host = st.secrets.get("pinecone_host", "https://hr-policies-index-gh700zo.svc.aped-4627-b74a.pinecone.io")
# For your configuration, we are using AWS us-east-1
pinecone_region = "us-east-1"

# Ensure API keys are set
if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

# ----------- Environment Setup -----------
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

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
# 1) Condense Follow-Up Questions
condense_template = """
Given the conversation below and a follow-up question, rephrase the follow-up question
to be standalone if it references previous context. If it is unrelated, just use it as-is.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""
condense_question_prompt = PromptTemplate.from_template(condense_template)

# 2) QA Prompt with dynamic fallback referencing the user's question
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
    2) Create embeddings & build a Pinecone vectorstore.
    3) Create memory & retrieval chain.
    4) Return chain for question-answer usage.
    """
    # Load and chunk docs
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # Create embeddings
    embeddings = get_embeddings()

    # Create Pinecone client instance using the new API
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(
        api_key=api_key_pinecone,
        host=pinecone_host,
        spec=ServerlessSpec(cloud='aws', region=pinecone_region)
    )

    # Retrieve the index instance from the Pinecone client
    index = pc.Index(index_name)

    # Manually build the vectorstore by providing the index instance.
    # The LangChain wrapper expects a client of type pinecone.Index, so we bypass the
    # internal creation using from_documents.
    # Here we set text_key to "text" and leave namespace as None.
    vectorstore = LangChainPinecone(index=index, embedding=embeddings, text_key="text", namespace=None)

    # Prepare texts and metadata from chunks and add them to the vectorstore.
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    vectorstore.add_texts(texts, metadatas)

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