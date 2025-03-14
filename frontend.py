import streamlit as st
from pinecone import Pinecone
import rag
from googletrans import Translator  # For translation

############################################
#             SETTINGS                    #
############################################
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index"

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

messages = st.empty()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

############################################
#       HELPER FUNCTIONS                   #
############################################
def translate_text(text, dest_language):
    """Translates text to the selected language."""
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

def generate_response(input_data_query, language="en"):
    """
    Calls the RAG module to process the query and return the response.
    Optionally translates the query and response.
    """
    try:
        if language != "en":
            input_data_query = translate_text(input_data_query, language)
        chain = rag.ask_model()
        output = rag.perform_query(chain, input_data_query)
        if language != "en":
            output = translate_text(output, "en")  # Translate response back to English
        return output
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return {"answer": "Sorry, I couldn't process your request. Please try again later."}

def display_messages():
    """Displays the conversation in the Streamlit UI."""
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for message in st.session_state["messages"]:
            role = "user" if message["role"] == "user" else "assistant"
            st.chat_message(role).write(message["content"])

def handle_initial_prompt(prompt_key):
    """Handles form button clicks."""
    st.session_state["messages"].append({"role": "user", "content": initial_prompts[prompt_key]})
    ans = generate_response(initial_prompts[prompt_key])
    st.session_state["messages"].append({"role": "assistant", "content": ans["answer"]})
    display_messages()

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

# Add a "Clear Chat" button
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    display_messages()

# Improve initial prompts layout
with placeholder.container():
    st.write("### Quick Prompts")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(initial_prompts["policy_scope"]):
            handle_initial_prompt("policy_scope")
        if st.button(initial_prompts["report_violation"]):
            handle_initial_prompt("report_violation")
    with col2:
        if st.button(initial_prompts["training_details"]):
            handle_initial_prompt("training_details")
        if st.button(initial_prompts["last_update"]):
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
        answer = generate_response(prompt, selected_language.lower())
        st.session_state["messages"].append({"role": "assistant", "content": answer["answer"]})
        st.session_state["messages"] = st.session_state["messages"][-100:]  # Limit messages
        display_messages()

# Listen for user-typed questions
user_query()