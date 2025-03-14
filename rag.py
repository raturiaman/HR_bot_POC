import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Use OpenAI embeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

############################################
#          PINECONE SETTINGS              #
############################################
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_pinecone = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
directory = os.getenv("PDF_DIRECTORY", "./pdfs")
index_name = os.getenv("PINECONE_INDEX_NAME", "hr-policies-index")

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check environment variables.")

############################################
#          INITIALIZE PINECONE            #
############################################
pc = Pinecone(api_key=api_key_pinecone)

# Ensure index exists
existing_indexes = [index_info.name for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Match the dimension of OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=pinecone_env
        )
    )

# Fetch the Pinecone index
index = pc.Index(index_name)

############################################
#        DOCUMENT PROCESSING FUNCTIONS    #
############################################
def read_docs(directory):
    """Load all PDFs in the given directory."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in directory '{directory}'.")
    return documents

def chunk_docs(documents, chunk_size=800, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def get_embeddings():
    """Return OpenAI embeddings for text encoding."""
    return OpenAIEmbeddings(model="text-embedding-ada-002")  # Use OpenAI embeddings

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
        chain_type_kwargs={"prompt": qa_prompt},
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
    try:
        # Load & chunk
        docs = read_docs(directory)
        chunks = chunk_docs(docs)

        # Embeddings
        embeddings = get_embeddings()

        # Build LangChain Pinecone vectorstore
        vectorstore = LangChainPinecone.from_documents(
            documents=chunks,
            embedding=embeddings,
            index=index
        )

        # Create chain with memory
        memory = get_memory()
        chain = create_chain(vectorstore, memory)
        return chain
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the model: {e}")

############################################
#        PERFORM RETRIEVAL FUNCTION       #
############################################
def perform_query(chain, query):
    """Call the chain with the user's query; returns the chain's result."""
    try:
        result = chain({"question": query})
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to perform query: {e}")