# app/main.py
import streamlit as st
from pathlib import Path
import time
from typing import List, Dict
import os, sys
from urllib.parse import urlencode
from pinecone import Pinecone, ServerlessSpec

#################
# Please comment this line while working on local machine
# import sys
# sys.modules["sqlite3"] = __import__("pysqlite3")
####################

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config

# Set page config as the first Streamlit command
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
)

from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore
from core.llm import LLMManager

def check_environment():
    """Check if all required environment variables are set."""
    missing_vars = []
    
    if not config.OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not config.PINECONE_API_KEY:
        missing_vars.append("PINECONE_API_KEY")
    if not config.PINECONE_ENVIRONMENT:
        missing_vars.append("PINECONE_ENVIRONMENT")
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}\n"
        error_msg += "Please ensure these variables are set in your .env file or environment."
        raise ValueError(error_msg)

def display_sources(sources: List[Dict]):
    """Display sources with proper formatting and links."""
    if not sources:
        return
    
    with st.expander("📚 Source References", expanded=False):
        for i, source in enumerate(sources, 1):
            metadata = source.get('metadata', {})
            url = metadata.get('url', '')
            
            st.markdown(f"### Reference {i}")
            if url:
                st.markdown(f"[🔗 {metadata.get('source', 'Source')}]({url})")
            else:
                st.markdown(f"**{metadata.get('source', 'Source')}**")
            
            # Show preview text
            preview_text = source['text'][:1000] + "..." if len(source['text']) > 1000 else source['text']
            st.caption(preview_text)
            st.divider()

# Update the initialize_components function
# @st.cache_resource
# def initialize_components():
#     try:
#         # Check environment variables first
#         check_environment()
#         pc = Pinecone(
#             api_key=config.PINECONE_API_KEY,
#             environment=config.PINECONE_ENVIRONMENT
#         )

#         components = {
#             'embedding_manager': EmbeddingManager(),
#             'vector_store': VectorStore(),
#             'llm_manager': LLMManager()
#         }
        
#         # Verify Pinecone index exists and is accessible
#         index = pc.Index(config.PINECONE_INDEX_NAME)
        
#         return components
#     except Exception as e:
#         st.error(f"Initialization Error: {str(e)}")
#         st.info("Please check your .env file and ensure all required API keys are set correctly.")
#         return None

@st.cache_resource
def initialize_components():
    try:
        # Check environment variables first
        check_environment()
        
        # Initialize Pinecone with better error handling
        try:
            pc = Pinecone(
                api_key=config.PINECONE_API_KEY,
                environment=config.PINECONE_ENVIRONMENT
            )
            # Verify Pinecone index exists and is accessible
            index = pc.Index(config.PINECONE_INDEX_NAME)
        except Exception as e:
            st.error(f"Pinecone initialization error: {str(e)}")
            return None
            
        # Initialize components one by one with better error handling
        try:
            embedding_manager = EmbeddingManager()
        except Exception as e:
            st.error(f"Embedding Manager Error: {str(e)}")
            return None
            
        try:
            vector_store = VectorStore()
        except Exception as e:
            st.error(f"Vector Store Error: {str(e)}")
            return None
            
        try:
            llm_manager = LLMManager()
        except Exception as e:
            st.error(f"LLM Manager Error: {str(e)}")
            return None
        
        components = {
            'embedding_manager': embedding_manager,
            'vector_store': vector_store,
            'llm_manager': llm_manager
        }
        
        return components
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        st.info("Please check your .env file and ensure all required API keys are set correctly.")
        return None
    

components = initialize_components()

if components is None:
    st.stop()

embedding_manager = components['embedding_manager']
vector_store = components['vector_store']
llm_manager = components['llm_manager']

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "context_window" not in st.session_state:
    st.session_state.context_window = 5
if "max_history" not in st.session_state:
    st.session_state.max_history = 10
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

st.title(config.APP_TITLE)

st.markdown("""
Get answers to all your Finance related queries.
""")



# Floating "New Conversation" Button at bottom-right
st.markdown("""
    <style>
    .new-convo-button {
        position: fixed;
        bottom: 20px;
        right: 30px;
        z-index: 9999;
    }
    </style>
    <div class="new-convo-button">
        <form action="" method="post">
            <button type="submit">🔄 New Conversation</button>
        </form>
    </div>
""", unsafe_allow_html=True)

# Clear session state on button click (handle post request)
if st.session_state.get("reset_chat", False):
    st.session_state.chat_history = []
    st.session_state.current_sources = []
    st.session_state.reset_chat = False
    st.rerun()

# Use JS to detect button submit and set Streamlit state
st.markdown("""
    <script>
    const form = document.querySelector('.new-convo-button form');
    form.addEventListener('submit', async function(event) {
        event.preventDefault();
        await fetch('', { method: 'POST' });
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: true}, '*');
    });
    </script>
""", unsafe_allow_html=True)


# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display sources if enabled
if st.session_state.show_sources and st.session_state.current_sources:
    with st.expander("Source Documents", expanded=False):
        for i, source in enumerate(st.session_state.current_sources):
            st.markdown(f"**Source {i+1}**")
            st.write(source["text"])
            if "metadata" in source and "url" in source["metadata"]:
                st.markdown(f"[Link to source]({source['metadata']['url']})")
            st.divider()


# User input
user_input = st.chat_input("Ask me anything about Finance...")

# Update the query processing in the main chat interface
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # Generate embedding for query
            query_embedding = embedding_manager.generate_embeddings([user_input])[0]
            relevant_docs = vector_store.search(
                user_input,
                query_embedding,
                k=st.session_state.context_window
            )
            
            # Save the current sources for potential display
            st.session_state.current_sources = relevant_docs

            # Generate response with enhanced LLM manager
            response = llm_manager.generate_response(
                user_input,
                relevant_docs,
                st.session_state.chat_history[-st.session_state.max_history:],
                streaming_container=response_placeholder
            )
            
            # Display the response
            response_placeholder.markdown(response)
            
            # Display sources separately
            if st.session_state.show_sources:
                display_sources(relevant_docs)

            # Update chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            st.error(f"An error occurred during query processing: {str(e)}")
            st.error("Full error details:")
            st.exception(e)