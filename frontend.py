import streamlit as st
import requests
import time
st.set_page_config(page_title="RAG Buddy", page_icon="🏛️")
st.title("🏛️ UP Government AI Assistant")
st.caption("Powered by LangGraph, Qdrant, and Gemini")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about UP Government policies, portals, or departments."}]

# Display chat history on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What is the Pragati Portal?"):
    
    # 1. Display user message exactly as typed
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display the AI response (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Connect to your FastAPI Backend
        url = f"http://127.0.0.1:8000/chat/stream?message={prompt}"
        
        try:
            # The 'stream=True' catches the chunks yielded by your LangGraph!
            with requests.post(url, stream=True) as response:
                if response.status_code == 429:
                    st.warning("⚠️ Rate limit exceeded! Please wait a minute before asking again.")
                elif response.status_code != 200:
                    st.error(f"🚨 Server error: {response.status_code}")
                else:
                    try:
                        # 1. Download the full response as fast as possible to free up the server!
                        full_text = response.text
                        
                        # 2. Create a local typewriter generator for the beautiful effect
                        def typewriter_effect(text):
                            for char in text:
                                yield char
                                time.sleep(0.005) # Super smooth speed
                        
                        # 3. Streamlit handles the blinking cursor and rendering perfectly
                        full_response = st.write_stream(typewriter_effect(full_text))
                        
                    except Exception as e:
                        st.error(f"📡 Connection interrupted: {e}")
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.ConnectionError:
            st.error("🚨 FastApi server is not running! Please start Uvicorn first.")
