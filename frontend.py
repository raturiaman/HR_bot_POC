"""
Frontend for our Human Rights Policy Chatbot.
Uses the updated approach in rag.py where we upsert to Pinecone manually,
then connect via from_existing_index.
"""

import streamlit as st
import rag  # Our updated RAG module

# If you want multilingual translations:
from deep_translator import GoogleTranslator, single_detection

########################
# Configuration
########################
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = st.secrets.get("index_name", "hr-policies-index-1536")

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

########################
# Session State for Chat
########################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

########################
# Language Options
########################
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Mandarin": "zh-cn",
    "Japanese": "ja",
    "German": "de"
}

########################
# Streamlit UI
########################
st.title("Human Rights Policy Chatbot")
selected_language = st.selectbox("Select output language:", list(languages.keys()))

########################
# Helper Functions
########################
def display_messages():
    """Render the conversation from st.session_state."""
    for msg in st.session_state["messages"]:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(msg["content"])


def generate_response(query_text: str) -> dict:
    """Call the updated rag module's chain and retrieve an answer."""
    # Check if chain is built in session_state
    if "chain" not in st.session_state:
        st.session_state["chain"] = rag.ask_model()
    chain = st.session_state["chain"]

    # Perform query
    output = rag.perform_query(chain, query_text)
    return output

########################
# Testing Buttons
########################
sample_prompts = {
    "policy_scope":  "What is covered in the Human Rights Policy?",
    "report_violation": "How can employees report a violation of this Human Rights Policy?",
    "training_details": "Does the policy talk about training employees on human rights?",
    "last_update": "When was the Human Rights Policy last updated?"
}

placeholder = st.empty()

with placeholder.form("sample_questions"):
    col1, col2 = st.columns(2)
    with col1:
        btn_scope = st.form_submit_button(label=sample_prompts["policy_scope"])
        btn_report = st.form_submit_button(label=sample_prompts["report_violation"])
    with col2:
        btn_training = st.form_submit_button(label=sample_prompts["training_details"])
        btn_update = st.form_submit_button(label=sample_prompts["last_update"])


def handle_prompt_click(prompt_key):
    # Adds user prompt to chat, gets chain answer, translates if needed
    st.session_state["messages"].append({"role": "user", "content": sample_prompts[prompt_key]})
    raw_result = generate_response(sample_prompts[prompt_key])
    english_answer = raw_result["answer"]

    # Translate from English to selected language
    final_answer = GoogleTranslator(
        source='en',
        target=languages[selected_language]
    ).translate(english_answer)

    st.session_state["messages"].append({"role": "assistant", "content": final_answer})
    display_messages()
    placeholder.empty()

if btn_scope:
    handle_prompt_click("policy_scope")
if btn_report:
    handle_prompt_click("report_violation")
if btn_training:
    handle_prompt_click("training_details")
if btn_update:
    handle_prompt_click("last_update")


########################
# Main Chat Input
########################
def user_query():
    prompt = st.chat_input("Ask your question about the HR Policy...")
    if prompt:
        # Detect user input language
        try:
            input_lang = single_detection(prompt, api_key=None)
        except:
            input_lang = "auto"

        # Translate user question to English (if not already)
        english_question = GoogleTranslator(
            source=input_lang,
            target='en'
        ).translate(prompt)

        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Query chain
        chain_result = generate_response(english_question)
        english_answer = chain_result["answer"]

        # Translate final answer to selected language
        final_answer = GoogleTranslator(
            source='en',
            target=languages[selected_language]
        ).translate(english_answer)

        # Add assistant answer to chat
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

user_query()