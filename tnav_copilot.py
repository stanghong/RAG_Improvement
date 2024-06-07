

import streamlit as st
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv('.env')

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o"

# Initialize OpenAI client
client = OpenAI(api_key=openai.api_key)

# Streamlit UI components
st.title("Tnav Copilot")

# Chat history: If not exists, initialize it
if 'history' not in st.session_state:
    st.session_state.history = []

# Ensure unique keys for user input to avoid DuplicateWidgetID error
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

# Generate unique key for text input
input_key = f'input_{st.session_state.input_key}'
user_input = st.text_input("Ask Tnav a question or type 'exit' to stop:", key=input_key)

if st.button("Send", key='send_button'):
    if user_input.lower() == "exit":
        st.stop()
    elif user_input:
        # API call to OpenAI with user input
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": 
                 "You are a reservoir engineer and simulation expert in the oil and gas industry."
                 "you will help write tnav simulation code"
                  " You can help engineers build, read, and debug tnav eclipse style simulation code and provide explainable simulation code. If you do not know, just say 'I do not know'."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
        )

        if completion:
            response = completion.choices[0].message.content
            # Append user input and response to history
            st.session_state.history.append({"user": user_input, "assistant": response})
            
            # Display the response in a chat box using Markdown
            st.markdown(f"**Assistant:**\n{response}", unsafe_allow_html=True)
        
        # Increment the input key for the next input
        st.session_state['input_key'] += 1
