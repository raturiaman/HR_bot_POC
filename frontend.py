import streamlit as st
from pinecone import Pinecone
import rag  # Importing the RAG module

############################################
#             SETTINGS                    #
############################################
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index"  # Updated to match new configuration

if not api_key_openai or not api_key_pinecone:
    st.error("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

############################################
#       HELPER FUNCTIONS                   #
############################################
def generate_response(input_data_query):
    """
    Calls the RAG module to process the query and return the response.
    """
    try:
        chain = rag.ask_model()
        output = rag.perform_query(chain, input_data_query)
        return output["answer"]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request. Please try again later."

def display_messages():
    """
    Displays the conversation in the Streamlit UI.
    """
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(message["content"])

############################################
#            STREAMLIT UI                  #
############################################
st.title("Human Rights Policy Chatbot")
st.write("Ask questions about the Human Rights Policy.")

# --- Language Dropdown ---
languages = [
    "English",
    "Hindi",
    "Punjabi",
    "Urdu",
    "Mandarin",
    "Japanese",
    "German"
]
selected_language = st.selectbox("Select a language for your question:", languages)

# Some example prompts for quick testing
initial_prompts = {
    "policy_scope": "What is covered in the Human Rights Policy?",
    "report_violation": "How can employees report a violation of this Human Rights Policy?",
    "training_details": "Does the policy talk about training employees on human rights?",
    "last_update": "When was the Human Rights Policy last updated?"
}

# Display initial prompts as buttons
st.write("### Quick Prompts")
col1, col2 = st.columns(2)
with col1:
    if st.button(initial_prompts["policy_scope"]):
        st.session_state["messages"].append({"role": "user", "content": initial_prompts["policy_scope"]})
        answer = generate_response(initial_prompts["policy_scope"])
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    if st.button(initial_prompts["report_violation"]):
        st.session_state["messages"].append({"role": "user", "content": initial_prompts["report_violation"]})
        answer = generate_response(initial_prompts["report_violation"])
        st.session_state["messages"].append({"role": "assistant", "content": answer})
with col2:
    if st.button(initial_prompts["training_details"]):
        st.session_state["messages"].append({"role": "user", "content": initial_prompts["training_details"]})
        answer = generate_response(initial_prompts["training_details"])
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    if st.button(initial_prompts["last_update"]):
        st.session_state["messages"].append({"role": "user", "content": initial_prompts["last_update"]})
        answer = generate_response(initial_prompts["last_update"])
        st.session_state["messages"].append({"role": "assistant", "content": answer})

# Display chat history
display_messages()

############################################
#        CHAT INPUT FUNCTION               #
############################################
def user_query():
    """
    Handles user input from Streamlit's chat UI and displays the conversation.
    """
    prompt = st.chat_input(f"Ask your question in {selected_language}...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        answer = generate_response(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["messages"] = st.session_state["messages"][-100:]  # Limit messages for memory management
        display_messages()

# Listen for user-typed questions
user_query()

# Add a "Clear Chat" button
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.experimental_rerun()