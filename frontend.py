"""
This is a RAG model which uses BeyondKey HR policies.
Created By: 
Create Date: 
Last Updated Date:

Updates:
Update on 23 Feb 2024 by Jugal:
    * Code merged in Streamlit app.
    * Some fixes and updates
"""

import streamlit as st
from pinecone import Pinecone  # Correct Pinecone import
import rag  # Importing the RAG module

# ----------------- Setting -------------------------
api_key_openai = st.secrets["OPENAI_API_KEY"]
api_key_pinecone = st.secrets["PINECONE_API_KEY"]
directory = st.secrets["directory"]
index_name = st.secrets["index_name"]

# ----------------- Setting -------------------------

messages = st.empty()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def generate_response(input_data_query):
    chain = rag.ask_model()
    question = input_data_query
    output = rag.perform_query(chain, question)
    return output  # Return the response

st.title("HR Policy Chatbot")

def display_messages():
    for message in st.session_state["messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        st.chat_message(role).write(message["content"])

initial_prompts = {
    "leave_policy": "Explain Leave Policy of the company",
    "loan_policy": "Explain Loan Policy of the company",
    "employee_assets": "What are the assets provided by the company for an employee?",
    "working_hours": "What are the minimum working hours in a week?"
}

placeholder = st.empty()

with placeholder.form(key='my_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        firstQ = st.form_submit_button(label=initial_prompts["leave_policy"])
        thirdQ = st.form_submit_button(label=initial_prompts["employee_assets"])
    with col2:
        secondQ = st.form_submit_button(label=initial_prompts["loan_policy"])
        fourthQ = st.form_submit_button(label=initial_prompts["working_hours"])

if firstQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["leave_policy"]})
    ans = generate_response(initial_prompts["leave_policy"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if secondQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["loan_policy"]})
    ans = generate_response(initial_prompts["loan_policy"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if thirdQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["employee_assets"]})
    ans = generate_response(initial_prompts["employee_assets"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

if fourthQ:
    st.session_state["messages"].append({"role": "user", "content": initial_prompts["working_hours"]})
    ans = generate_response(initial_prompts["working_hours"])
    st.session_state["messages"].append({"role": "assistant", "content": ans['answer']})
    display_messages()
    placeholder.empty()

def user_query():
    """Handles user input and displays the conversation."""
    prompt = st.chat_input("Ask a question about HR policies...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        answer = generate_response(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": answer['answer']})
        st.session_state["messages"] = st.session_state["messages"][-100:]  # Limit messages
        display_messages()
        placeholder.empty()

user_query()