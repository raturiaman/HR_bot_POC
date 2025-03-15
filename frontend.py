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
from pinecone import Pinecone  # Pinecone client import
import rag  # Importing the RAG module

# ----------------- Setting -------------------------
api_key_openai = st.secrets.get("OPENAI_API_KEY", "")
api_key_pinecone = st.secrets.get("PINECONE_API_KEY", "")
directory = st.secrets.get("directory", "./pdfs")
index_name = "hr-policies-index"  # This is the default in rag.py

# Ensure API keys are set
if not api_key_openai or not api_key_pinecone:
    raise ValueError("Missing OpenAI or Pinecone API key. Check secrets.toml or environment variables.")

messages = st.empty()

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def generate_response(input_data_query):
    """
    This function calls the RAG (Retrieval-Augmented Generation) module functions
    to build the chain and get the result from the chain for the user query.
    """
    # Ask for or reuse a chain in session
    chain = rag.ask_model()
    question = input_data_query
    output = rag.perform_query(chain, question)
    return output  # Return the structured response from the chain

st.title("Human Rights Policy Chatbot")

def display_messages():
    """
    Renders the conversation from st.session_state["messages"] in the Streamlit UI.
    """
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        st.chat_message(role).write(message["content"])

# Some example prompts for quick testing
initial_prompts = {
    "policy_scope":  "What is covered in the Human Rights Policy?",
    "report_violation": "How can employees report a violation of this Human Rights Policy?",
    "training_details": "Does the policy talk about training employees on human rights?",
    "last_update": "When was the Human Rights Policy last updated?"
}

placeholder = st.empty()

# Quick test form
with placeholder.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        firstQ = st.form_submit_button(label=initial_prompts["policy_scope"])
        secondQ = st.form_submit_button(label=initial_prompts["report_violation"])
    with col2:
        thirdQ = st.form_submit_button(label=initial_prompts["training_details"])
        fourthQ = st.form_submit_button(label=initial_prompts["last_update"])

# Handle form button clicks
if firstQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["policy_scope"]})
    ans = generate_response(initial_prompts["policy_scope"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if secondQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["report_violation"]})
    ans = generate_response(initial_prompts["report_violation"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if thirdQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["training_details"]})
    ans = generate_response(initial_prompts["training_details"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if fourthQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["last_update"]})
    ans = generate_response(initial_prompts["last_update"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

def user_query():
    """Handles user input from Streamlit's chat UI and displays the conversation."""
    prompt = st.chat_input("Ask about the Human Rights Policy...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        answer = generate_response(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": answer['answer']})
        # Limit messages to the last 100 to manage memory usage
        st.session_state["messages"] = st.session_state["messages"][-100:]
        display_messages()
        placeholder.empty()

# Listen for user-typed questions
user_query()