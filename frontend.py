import streamlit as st
from pinecone import Pinecone  # Pinecone client import
import rag  # Importing the RAG module

############################################
#             SETTINGS                    #
############################################
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index-new"  # Updated index name

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

messages = st.empty()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

############################################
#       HELPER FUNCTIONS                   #
############################################
def generate_response(input_data_query):
    """
    Calls the RAG module to process the query and return the response.
    """
    chain = rag.ask_model()
    output = rag.perform_query(chain, input_data_query)
    return output

def display_messages():
    """
    Displays the conversation in the Streamlit UI.
    """
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        st.chat_message(role).write(message["content"])

############################################
#            STREAMLIT UI                  #
############################################
st.title("Human Rights Policy Chatbot")

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

placeholder = st.empty()

with placeholder.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        firstQ = st.form_submit_button(label=initial_prompts["policy_scope"])
        secondQ = st.form_submit_button(label=initial_prompts["report_violation"])
    with col2:
        thirdQ = st.form_submit_button(label=initial_prompts["training_details"])
        fourthQ = st.form_submit_button(label=initial_prompts["last_update"])

def handle_initial_prompt(prompt_key):
    """Handles form button clicks."""
    st.session_state["messages"].append({"role": "user", "content": initial_prompts[prompt_key]})
    ans = generate_response(initial_prompts[prompt_key])
    st.session_state["messages"].append({"role": "assistant", "content": ans["answer"]})
    display_messages()
    placeholder.empty()

if firstQ:
    handle_initial_prompt("policy_scope")
if secondQ:
    handle_initial_prompt("report_violation")
if thirdQ:
    handle_initial_prompt("training_details")
if fourthQ:
    handle_initial_prompt("last_update")

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
        st.session_state["messages"].append({"role": "assistant", "content": answer["answer"]})
        st.session_state["messages"] = st.session_state["messages"][-100:]  # Limit messages for memory management
        display_messages()
        placeholder.empty()

# Listen for user-typed questions
user_query()