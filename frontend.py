"""
This is a retrieval-augmented QA (RAG) model for "Human Rights Policy".
It uses Pinecone for vector search, OpenAI for LLM/embeddings, and 
deep-translator for multilingual answers (if desired).
"""

import streamlit as st
from pinecone import Pinecone
import rag  # Our backend logic

# Optional: For multilingual translation
from deep_translator import GoogleTranslator, single_detection

# ----------------- Setting -------------------------
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index"

if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

# Store messages for the conversation display
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Human Rights Policy Chatbot (Strict Policy Answers)")

# 1) Language codes for the output language (optional)
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Mandarin": "zh-cn",
    "Japanese": "ja",
    "German": "de"
}
selected_language = st.selectbox("Select the output language:", list(languages.keys()))

#######################
#    Helper Methods   #
#######################
def display_messages():
    """
    Renders the conversation from st.session_state["messages"] in the Streamlit UI.
    """
    for msg in st.session_state["messages"]:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(msg["content"])

def generate_response(query_text: str) -> dict:
    """
    1) Build or reuse the chain from rag.py
    2) Provide the query to the chain
    3) Return the result
    """
    chain = rag.ask_model()  # Rebuild or reuse
    output = rag.perform_query(chain, query_text)
    return output  # e.g. {"answer": "some text"}


#######################
# Quick Testing Buttons
#######################
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
    """
    Push the sample prompt to conversation, get the chain's answer,
    then translate the final answer to the user-selected language.
    """
    st.session_state["messages"].append({"role": "user", "content": sample_prompts[prompt_key]})
    raw_result = generate_response(sample_prompts[prompt_key])
    english_answer = raw_result["answer"]

    # Translate answer from English to selected language
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


#######################
#   Chat Input
#######################
def user_query():
    """
    1) Takes user input from st.chat_input
    2) Attempt to auto-detect the typed language
    3) Translate to English, pass to chain
    4) Translate final answer back to selected language
    5) Display everything in st.session_state["messages"]
    """
    prompt = st.chat_input("Ask any question about the Human Rights Policy here...")
    if prompt:
        # Detect user input language (optional)
        try:
            input_lang = single_detection(prompt, api_key=None)
        except:
            input_lang = "auto"

        # Translate user question to English
        english_question = GoogleTranslator(
            source=input_lang,
            target='en'
        ).translate(prompt)

        # Store user question
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Query chain
        chain_result = generate_response(english_question)
        english_answer = chain_result["answer"]

        # Translate chain's answer to selected output language
        final_answer = GoogleTranslator(
            source='en',
            target=languages[selected_language]
        ).translate(english_answer)

        # Store the final answer
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})

        # Limit messages
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

user_query()