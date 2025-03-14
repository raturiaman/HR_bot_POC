import streamlit as st
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from pinecone import Pinecone

# --------- Load keys from secrets or environment
api_key_openai = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
index_name = st.secrets.get("index_name", "hr-policies-index")

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

# Set environment variables
os.environ["OPENAI_API_KEY"] = api_key_openai
os.environ["PINECONE_API_KEY"] = api_key_pinecone

# --------- Step 1: Load PDF
def load_pdf(pdf_path: str):
    # You can point this to a directory with only one PDF
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    return docs

# --------- Step 2: Split into Chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

# --------- Step 3: Create or Connect to Pinecone Index
def build_vectorstore(chunks):
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key_pinecone)

    embeddings = OpenAIEmbeddings()  # OpenAI Embeddings
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore

# --------- Step 4: Create Conversational Chain
def create_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    # Prompt for refining follow-up questions
    condense_template = """
        Given the following conversation and a follow-up question, rephrase the question to be standalone if it
        references previous context. If it's unrelated, answer directly:
        Chat History:
        {chat_history}
        Follow-Up Input: {question}
        Standalone question:
    """
    condense_prompt = PromptTemplate.from_template(condense_template)

    # QA Prompt for final answer
    qa_template = """
    You are an HR-policy assistant. Use the context below (excerpts from 'Human Rights Policy.pdf') to answer the question.
    If you don't know the answer, just say you don't know; do not fabricate.

    Context:
    {context}
    ---
    Question: {question}
    Helpful Answer:
    """
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain

# --------- Step 5: Build the App
def main():
    st.title("Human Rights Policy Chatbot")

    # 1. Load + chunk the single PDF
    docs = load_pdf("./")  # Or wherever 'Human Rights Policy.pdf' is
    chunks = split_docs(docs)

    # 2. Build / connect to Pinecone vectorstore
    vectorstore = build_vectorstore(chunks)

    # 3. Create chain
    chain = create_chain(vectorstore)

    # 4. Streamlit: ask user for questions
    user_question = st.text_input("Ask about the Human Rights Policy:")
    if user_question:
        result = chain({"question": user_question})
        st.write(result["answer"])

if __name__ == "__main__":
    main()