"""
This is a RAG model which uses BeyondKey's Human Rights Policy.
Created By:
Create Date:
Last Updated Date:

Updates:
Update on 23 Feb 2024 by Jugal:
    * Code merged in Streamlit app.
    * Some fixes and updates
"""

import streamlit as st
from pinecone import Pinecone
import rag  # The RAG module you created

# (NEW) Translator import
from googletrans import Translator

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
#      New Code        #
########################
# (NEW) Initialize translator once
translator = Translator()

# Language dropdown options (for output language)
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Mandarin": "zh-cn",
    "Japanese": "ja",
    "German": "de"
}

st.title("Human Rights Policy Chatbot (Multilingual Answers)")

# (NEW) Let user pick output language
selected_language = st.selectbox("Select output language for answers:", list(languages.keys()))

########################
#  Helper Functions    #
########################
def generate_response(input_data_query: str):
    """
    This function calls the RAG (Retrieval-Augmented Generation) module functions
    to build the chain and get the result from the chain for the user query.
    """
    chain = rag.ask_model()
    output = rag.perform_query(chain, input_data_query)
    return output

def display_messages():
    """
    Renders the conversation from st.session_state["messages"] in the Streamlit UI.
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
    # No translation needed for these pre-defined prompts (already in English).
    st.session_state["messages"].append({"role": "user", "content": initial_prompts[prompt_key]})
    
    # Query chain
    ans = generate_response(initial_prompts[prompt_key])
    
    # (NEW) Translate the answer from English to the selected language
    translated_answer = translator.translate(ans["answer"], dest=languages[selected_language]).text
    
    st.session_state["messages"].append({"role": "assistant", "content": translated_answer})
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
    Handles user input from Streamlit's chat UI and displays the conversation.
    1) Detect or assume user input language
    2) Translate user query to English
    3) Query the chain
    4) Translate chain's answer from English to chosen language
    5) Display everything in st.session_state["messages"]
    """
    prompt = st.chat_input("Ask your question in ANY language. Answers will appear in selected language.")
    if prompt:
        # (NEW) 1) Detect or assume user input language
        detect_obj = translator.detect(prompt)
        input_lang = detect_obj.lang  # e.g., 'en', 'hi', 'pa'
        
        # (NEW) 2) Translate user query to English (the chain expects English)
        # If user typed in English, translator won't change it
        english_question = translator.translate(prompt, src=input_lang, dest="en").text
        
        # Log the user input in the conversation
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # (NEW) 3) Query the chain with the English version
        raw_answer = generate_response(english_question)
        english_answer = raw_answer["answer"]
        
        # (NEW) 4) Translate chain's English answer to the selected language
        final_answer = translator.translate(
            english_answer, 
            src="en", 
            dest=languages[selected_language]
        ).text
        
        # Store the final answer in the chat
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        
        # Keep only the last 100 messages
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

user_query()