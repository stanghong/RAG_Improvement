import streamlit as st
import subprocess

# Function to interact with the Ollama model with streaming
def query_ollama_model_streaming(prompt):
    # Command to run the Ollama model
    command = ["ollama", "run", "deepseek-r1:latest", prompt]
    
    # Start the subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream the output
    for line in process.stdout:
        yield line

# Streamlit app
def main():
    st.title("Local Chatbot with Deepseek r1")
    
    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Button to start a new thread (clear chat history)
    if st.button("Start New Thread"):
        st.session_state.chat_history = []
        st.rerun()  # Refresh the app to clear the screen

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

    # Input text box for user prompt
    user_input = st.chat_input("Enter your prompt:")
    
    if user_input:
        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user input immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_input)
        
        # Create a placeholder for the thinking process
        thinking_placeholder = st.empty()
        
        # Stream the response
        full_response = ""
        for chunk in query_ollama_model_streaming(user_input):
            full_response += chunk
            # Update the thinking process box
            thinking_placeholder.info(f"**Thinking Process:**\n\n{full_response}")
        
        # Display the final response in a separate box
        with st.chat_message("assistant", avatar="🤖"):
            st.success(f"**Final Response:**\n\n{full_response}")
        
        # Add the final bot response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()