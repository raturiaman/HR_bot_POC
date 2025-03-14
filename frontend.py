"""
This is a RAG model which uses BeyondKey's Human Rights Policy,
now supporting multilingual answers using deep-translator.

Created By:
Create Date:
Last Updated Date:
"""

import streamlit as st
from pinecone import Pinecone
import rag  # The RAG module you created

# (NEW) Use deep-translator instead of googletrans
from deep_translator import GoogleTranslator, single_detection

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

# 1) Language codes for the output language
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Mandarin": "zh-cn",
    "Japanese": "ja",
    "German": "de"
}

st.title("Human Rights Policy Chatbot (Multilingual Answers with deep-translator)")

# 2) Let user pick output language
selected_language = st.selectbox(
    "Select output language for answers:", 
    list(languages.keys())
)

########################
#  Helper Functions    #
########################
def generate_response(input_data_query: str):
    """
    Calls the RAG module: builds chain, gets result from chain for the user query.
    """
    chain = rag.ask_model()
    output = rag.perform_query(chain, input_data_query)
    return output

def display_messages():
    """
    Renders the conversation from st.session_state["messages"] in Streamlit.
    """
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        st.chat_message(role).write(message["content"])

########################
#   Quick test prompts #
########################
initial_prompts = {
    "policy_scope":  "What is covered in the Human Rights Policy?",
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
    # We'll assume these are in English, so no pre-translation needed.
    st.session_state["messages"].append({"role": "user", "content": initial_prompts[prompt_key]})
    
    # Query chain with the English prompt
    ans = generate_response(initial_prompts[prompt_key])
    
    # Translate from English to the selected language
    final_answer = GoogleTranslator(
        source='en', 
        target=languages[selected_language]
    ).translate(ans["answer"])
    
    # Display
    st.session_state["messages"].append({"role": "assistant", "content": final_answer})
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

########################
#  Chat Input Box      #
########################
def user_query():
    """
    1) Attempt to detect user input language or let them type in any language
    2) Translate user query to English for the chain
    3) Query the chain
    4) Translate the chain's English answer to selected output language
    5) Display in st.session_state["messages"]
    """
    prompt = st.chat_input("Ask your question in ANY language. Answers will appear in selected language.")
    if prompt:
        # Try to detect user's input language (works best for well-formed text)
        try:
            # single_detection returns the language code, e.g. 'hi', 'ja'
            input_lang = single_detection(prompt, api_key=None)
        except:
            # If detection fails, fallback to 'auto'
            input_lang = 'auto'
        
        # Translate the user question to English for the chain
        english_question = GoogleTranslator(
            source=input_lang, 
            target='en'
        ).translate(prompt)
        
        # Show the user question as typed (no translation in messages)
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Query chain with the English version
        raw_answer = generate_response(english_question)
        english_answer = raw_answer["answer"]
        
        # Translate chain's English answer to selected language
        final_answer = GoogleTranslator(
            source='en', 
            target=languages[selected_language]
        ).translate(english_answer)
        
        # Store final answer in the chat
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        
        # Keep only the last 100 messages
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

user_query()