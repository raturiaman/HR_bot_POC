import streamlit as st
import os

# Document loaders and text splitting
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vector store (community packages)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone

# Memory and prompt templates
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# Official pinecone package
from pinecone import Pinecone

# ---------- Settings (API Keys) ---------
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
directory = st.secrets.get("directory", os.getenv("PDF_DIRECTORY", "./pdfs"))
index_name = st.secrets.get("index_name", "hr-policies-index")

# Ensure API keys are set
if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

# ----------- Environment Setup -----------
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

# ----------- Document Processing -----------
def read_docs(pdf_directory):
    """
    Loads all PDFs from the specified directory using PyPDFDirectoryLoader.
    """
    loader = PyPDFDirectoryLoader(pdf_directory)
    return loader.load()

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    """
    Splits the loaded documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings():
    """
    Provides the OpenAI embeddings (text-embedding-ada-002 by default).
    """
    return OpenAIEmbeddings()

# ----------- Memory -----------
def get_memory():
    """
    Creates or retrieves a conversation buffer memory from st.session_state.
    This stores the last 5 messages in conversation for ChatGPT-like interaction.
    """
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
    return st.session_state["memory"]

# ----------- Prompt Templates -----------

# 1) Condensing Follow-Up Questions
condense_template = """
Given the conversation below and a follow-up question, rephrase the follow-up question
to be a standalone question if it references prior context. If it's unrelated, answer directly.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""

condense_question_prompt = PromptTemplate.from_template(condense_template)


# 2) Strict QA Prompt with fallback
qa_template = """
You are a helpful QA assistant specialized in the 'Human Rights Policy'. 
Use ONLY the context below to answer the question. The context is from the 'Human Rights Policy'.

If the question cannot be answered using the context (for example, it is about politics 
or any out-of-scope topic), respond EXACTLY with:

"No relevant information found in the Human Rights Policy for your query. Please consult your HR department for more details."

Context:
{context}

Question: {question}
Helpful Answer:
"""

qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)

# ----------- Retrieval Chain Creation -----------
def create_chain(vectorstore, memory):
    """
    Build a ConversationalRetrievalChain with:
      - llm=OpenAI()
      - the Pinecone-based retriever
      - conversation memory
      - custom condense prompt
      - custom QA prompt with fallback
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

# ----------- Main / Model Setup -----------
def ask_model():
    """
    1) Load PDF docs from directory
    2) Chunk them
    3) Create an embeddings object
    4) Create or connect a Pinecone vectorstore
    5) Build the retrieval chain
    6) Return the chain
    """
    # 1) Load and chunk the PDF documents
    docs = read_docs(directory)
    chunks = chunk_docs(docs)

    # 2) Embeddings
    embeddings = get_embeddings()

    # 3) Initialize official Pinecone client
    pc = Pinecone(api_key=api_key_pinecone)

    # 4) Build vectorstore from docs each time
    #    or use from_existing_index(...) if you prefer
    vectorstore = LangChainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    # 5) Memory for conversation
    memory = get_memory()

    # 6) Create chain
    chain = create_chain(vectorstore, memory)
    return chain

# ----------- Perform Retrieval -----------
def perform_query(chain, query):
    """
    Provide the user query to the chain and return the result.
    Access the final answer via result["answer"].
    """
    result = chain({"question": query})
    return result