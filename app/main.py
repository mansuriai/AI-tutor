# # app/main.py
import streamlit as st
from pathlib import Path
import time
from typing import List, Dict
import os, sys
from urllib.parse import urlencode
from pinecone import Pinecone, ServerlessSpec
import re

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

def convert_latex_to_unicode(text: str) -> str:
    """Convert common LaTeX mathematical expressions to Unicode symbols."""
    # Dictionary of LaTeX to Unicode conversions
    latex_to_unicode = {
        r'\\times': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\mp': '∓',
        r'\\cdot': '·',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\equiv': '≡',
        r'\\propto': '∝',
        r'\\infty': '∞',
        r'\\sum': '∑',
        r'\\prod': '∏',
        r'\\int': '∫',
        r'\\partial': '∂',
        r'\\nabla': '∇',
        r'\\Delta': 'Δ',
        r'\\delta': 'δ',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\pi': 'π',
        r'\\sigma': 'σ',
        r'\\tau': 'τ',
        r'\\phi': 'φ',
        r'\\chi': 'χ',
        r'\\psi': 'ψ',
        r'\\omega': 'ω',
        r'\\sqrt': '√',
        r'\\frac': '',  # Will be handled separately
        r'\\text\{([^}]+)\}': r'\1',  # Remove \text{} wrapper
        r'\\_': '_',  # Underscore
        r'\\\\': '\n',  # Line break
    }
    
    # Apply basic substitutions
    result = text
    for latex, unicode_char in latex_to_unicode.items():
        if latex == r'\\text\{([^}]+)\}':
            result = re.sub(latex, unicode_char, result)
        else:
            result = result.replace(latex, unicode_char)
    
    return result

def format_mathematical_expression(text: str) -> str:
    """Format mathematical expressions for better readability."""
    # Remove LaTeX delimiters and convert to readable format
    
    # Handle inline math expressions $...$
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    # Handle display math expressions $$...$$
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    
    # Handle LaTeX brackets \[ ... \] and \( ... \)
    text = re.sub(r'\\\[([^\]]+)\\\]', r'\1', text)
    text = re.sub(r'\\\(([^)]+)\\\)', r'\1', text)
    
    # Handle curly braces in subscripts and superscripts
    text = re.sub(r'_\{([^}]+)\}', r'_\1', text)
    text = re.sub(r'\^\{([^}]+)\}', r'^\1', text)
    
    # Handle \text{} wrapper
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    
    # Handle fractions \frac{numerator}{denominator}
    def replace_frac(match):
        num = match.group(1)
        den = match.group(2)
        return f"({num})/({den})"
    
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, text)
    
    # Convert LaTeX symbols to Unicode
    text = convert_latex_to_unicode(text)
    
    # Clean up extra spaces and formatting
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def render_mathematical_content(content: str) -> str:
    """Process content to render mathematical expressions properly."""
    # Split content into parts, identifying mathematical expressions
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line contains mathematical expressions
        if any(marker in line for marker in ['$', '\\(', '\\[', '\\text{', '\\frac', '_', '^']):
            # This line likely contains math - format it
            formatted_line = format_mathematical_expression(line)
            processed_lines.append(formatted_line)
        else:
            # Regular text line
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

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
# st.markdown("""
#     <style>
#     .new-convo-button {
#         position: fixed;
#         bottom: 20px;
#         right: 30px;
#         z-index: 9999;
#     }
#     </style>
#     <div class="new-convo-button">
#         <form action="" method="post">
#             <button type="submit">🔄 New Conversation</button>
#         </form>
#     </div>
# """, unsafe_allow_html=True)

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
        # Process mathematical content in chat history
        processed_content = render_mathematical_content(message["content"])
        st.write(processed_content)

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
            
            # Process mathematical content in the response
            processed_response = render_mathematical_content(response)
            
            # Display the processed response
            response_placeholder.markdown(processed_response)
            
            # Display sources separately
            if st.session_state.show_sources:
                display_sources(relevant_docs)

            # Update chat history with processed response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": processed_response
            })
            
        except Exception as e:
            st.error(f"An error occurred during query processing: {str(e)}")
            st.error("Full error details:")
            st.exception(e)