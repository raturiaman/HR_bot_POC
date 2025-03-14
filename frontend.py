import streamlit as st
from pinecone import Pinecone  # Pinecone client import
import rag  # The RAG module you created

# ----------------- Setting -------------------------
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index"

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key.")

messages = st.empty()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

########################
#  Helper Functions    #
########################
def generate_response(input_data_query):
    chain = rag.ask_model()
    output = rag.perform_query(chain, input_data_query)
    return output

def display_messages():
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        st.chat_message(role).write(message["content"])

########################
# Streamlit UI Layout  #
########################
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

# --- Chat Input for user’s own question ---
def user_query():
    prompt = st.chat_input(f"Ask your question in {selected_language}...")
    if prompt:
        # Here 'prompt' is typed by the user in the selected language.
        # Right now, we pass it directly to the chain. If you want to 
        # handle multi-lingual retrieval, you could add translation here.
        st.session_state["messages"].append({"role": "user", "content": prompt})
        answer = generate_response(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": answer["answer"]})
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

user_query()