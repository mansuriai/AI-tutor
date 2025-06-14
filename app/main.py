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
            preview_text = source['text'][:300] + "..." if len(source['text']) > 300 else source['text']
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
    st.session_state.show_sources = False

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
user_input = st.chat_input("Ask me anything about IndiGo Airlines...")

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



################################################################# New 2 #############################



# import streamlit as st
# from pathlib import Path
# import time
# from typing import List, Dict
# import os, sys
# from urllib.parse import urlencode
# from pinecone import Pinecone, ServerlessSpec
# import json
# from datetime import datetime
# import pymysql
# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import SQLAlchemyError

# #################
# # Please comment this line while working on local machine
# # import sys
# # sys.modules["sqlite3"] = __import__("pysqlite3")
# ####################

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from utils.config import config

# # Import the ResponseLogger class (only if enabled)
# try:
#     # Check if the module exists before importing
#     response_logger_path = Path(__file__).parent.parent / "utils" / "response_logger.py"
#     RESPONSE_LOGGING_ENABLED = response_logger_path.exists()
    
#     if RESPONSE_LOGGING_ENABLED:
#         sys.path.append(str(Path(__file__).parent.parent))
#         from utils.response_logger import ResponseLogger
# except ImportError:
#     RESPONSE_LOGGING_ENABLED = False

# # Set page config as the first Streamlit command
# st.set_page_config(
#     page_title=config.APP_TITLE,
#     layout="wide",
# )

# from core.embeddings import EmbeddingManager
# from core.vector_store import VectorStore
# from core.llm import LLMManager

# # Add CSS for styling the feedback mechanism
# st.markdown("""
# <style>
# /* Feedback styling */
# .feedback-container {
#     background-color: #f8f9fa;
#     border-radius: 10px;
#     padding: 20px;
#     margin: 20px 0;
#     box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
# }

# .feedback-header {
#     font-size: 1.2rem;
#     font-weight: bold;
#     margin-bottom: 15px;
#     color: #2c3e50;
# }

# .feedback-button-container {
#     display: flex;
#     gap: 10px;
#     margin-bottom: 20px;
# }

# .feedback-button-positive {
#     background-color: #4CAF50;
#     color: white;
#     border: none;
#     border-radius: 20px;
#     padding: 10px 24px;
#     text-align: center;
#     text-decoration: none;
#     font-size: 16px;
#     cursor: pointer;
#     transition: background-color 0.3s;
# }

# .feedback-button-negative {
#     background-color: #f44336;
#     color: white;
#     border: none;
#     border-radius: 20px;
#     padding: 10px 24px;
#     text-align: center;
#     text-decoration: none;
#     font-size: 16px;
#     cursor: pointer;
#     transition: background-color 0.3s;
# }

# .feedback-button-positive:hover {
#     background-color: #45a049;
# }

# .feedback-button-negative:hover {
#     background-color: #d32f2f;
# }

# .feedback-form {
#     margin-top: 20px;
#     padding: 15px;
#     background-color: white;
#     border-radius: 8px;
#     border-left: 4px solid #3498db;
# }

# .feedback-section-title {
#     font-weight: 600;
#     color: #3498db;
#     margin-bottom: 10px;
# }

# .feedback-submit-button {
#     background-color: #3498db;
#     color: white;
#     border: none;
#     border-radius: 5px;
#     padding: 10px 20px;
#     margin-top: 15px;
#     cursor: pointer;
#     transition: background-color 0.3s;
# }

# .feedback-submit-button:hover {
#     background-color: #2980b9;
# }

# .feedback-success {
#     background-color: #d4edda;
#     color: #155724;
#     padding: 15px;
#     border-radius: 5px;
#     margin-top: 20px;
# }

# /* Custom styling for Streamlit components */
# .st-eb {
#     border-radius: 5px !important;
# }

# .stTextInput > div > div > input {
#     border-radius: 5px !important;
# }

# .stSelectbox > div > div > div {
#     border-radius: 5px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# def check_environment():
#     """Check if all required environment variables are set."""
#     missing_vars = []
    
#     if not config.OPENAI_API_KEY:
#         missing_vars.append("OPENAI_API_KEY")
#     if not config.PINECONE_API_KEY:
#         missing_vars.append("PINECONE_API_KEY")
#     if not config.PINECONE_ENVIRONMENT:
#         missing_vars.append("PINECONE_ENVIRONMENT")
    
#     if missing_vars:
#         error_msg = f"Missing required environment variables: {', '.join(missing_vars)}\n"
#         error_msg += "Please ensure these variables are set in your .env file or environment."
#         raise ValueError(error_msg)

# def display_sources(sources: List[Dict]):
#     """Display sources with proper formatting and links."""
#     if not sources:
#         return
    
#     with st.expander("📚 Source References", expanded=False):
#         for i, source in enumerate(sources, 1):
#             metadata = source.get('metadata', {})
#             url = metadata.get('url', '')
            
#             st.markdown(f"### Reference {i}")
#             if url:
#                 st.markdown(f"[🔗 {metadata.get('source', 'Source')}]({url})")
#             else:
#                 st.markdown(f"**{metadata.get('source', 'Source')}**")
            
#             # Show preview text
#             preview_text = source['text'][:300] + "..." if len(source['text']) > 300 else source['text']
#             st.caption(preview_text)
#             st.divider()

# def get_database_connection():
#     """Create a connection to the MySQL database."""
#     try:
#         # Database connection parameters
#         # Replace these with your actual database credentials or add them to config
#         hostname = "your_hostname"  # Replace with your actual hostname or config.DB_HOSTNAME
#         database = "your_database"  # Replace with your actual database name or config.DB_NAME
#         username = "your_username"  # Replace with your actual username or config.DB_USERNAME
#         password = "your_password"  # Replace with your actual password or config.DB_PASSWORD
#         port = "your_port_id"       # Replace with your actual port or config.DB_PORT
        
#         # Create connection string
#         connection_string = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/{database}"
        
#         # Create engine
#         engine = create_engine(connection_string)
        
#         # Test connection
#         with engine.connect() as connection:
#             connection.execute(text("SELECT 1"))
        
#         return engine
#     except Exception as e:
#         print(f"Database connection error: {str(e)}")
#         return None

# def create_feedback_table(engine):
#     """Create the feedback table if it doesn't exist."""
#     if engine is None:
#         return False
    
#     try:
#         # SQL statement to create table
#         create_table_sql = """
#         CREATE TABLE IF NOT EXISTS ai_assistant_feedback (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             feedback_type VARCHAR(20) NOT NULL,
#             query TEXT,
#             response TEXT,
#             time_saved VARCHAR(50),
#             rating INT,
#             recommend VARCHAR(5),
#             liked_aspects TEXT,
#             other_liked TEXT,
#             improvement_suggestions TEXT,
#             customer_experience VARCHAR(20),
#             usage_frequency VARCHAR(50),
#             issues TEXT,
#             other_feedback TEXT,
#             query_improvement TEXT,
#             timestamp DATETIME NOT NULL
#         );
#         """
        
#         # Execute the SQL
#         with engine.connect() as connection:
#             connection.execute(text(create_table_sql))
#             connection.commit()
        
#         return True
#     except SQLAlchemyError as e:
#         print(f"Error creating feedback table: {str(e)}")
#         return False

# def save_feedback(feedback_data, response=None):
#     """Save feedback data to both JSON file and SQL database."""
#     # Add timestamp to feedback
#     feedback_data["timestamp"] = datetime.now().isoformat()
    
#     # Add response if provided
#     if response:
#         feedback_data["response"] = response
    
#     # 1. Save to JSON file (keeping this for backward compatibility)
#     feedback_file = config.BASE_DIR / "feedback_data.json"
    
#     # Load existing feedback if file exists
#     if feedback_file.exists():
#         with open(feedback_file, "r") as f:
#             try:
#                 existing_data = json.load(f)
#             except json.JSONDecodeError:
#                 existing_data = []
#     else:
#         existing_data = []
    
#     # Append new feedback and save
#     existing_data.append(feedback_data)
#     with open(feedback_file, "w") as f:
#         json.dump(existing_data, f, indent=2)
    
#     # 2. Save to SQL database
#     try:
#         # Get database connection
#         engine = get_database_connection()
#         if engine is None:
#             print("Could not connect to database. Feedback saved to JSON file only.")
#             return
        
#         # Ensure table exists
#         table_created = create_feedback_table(engine)
#         if not table_created:
#             print("Could not create feedback table. Feedback saved to JSON file only.")
#             return
        
#         # Prepare data for insertion
#         db_feedback = {
#             "feedback_type": feedback_data.get("feedback_type", ""),
#             "query": feedback_data.get("query", ""),
#             "response": feedback_data.get("response", ""),
#             "time_saved": feedback_data.get("time_saved", ""),
#             "rating": feedback_data.get("rating", 0),
#             "recommend": feedback_data.get("recommend", ""),
#             "liked_aspects": json.dumps(feedback_data.get("liked_aspects", [])),
#             "other_liked": feedback_data.get("other_liked", ""),
#             "improvement_suggestions": feedback_data.get("improvement_suggestions", ""),
#             # "customer_experience": feedback_data.get("customer_experience", ""),  ##
#             "usage_frequency": feedback_data.get("usage_frequency", ""),
#             "issues": json.dumps(feedback_data.get("issues", [])),
#             "other_feedback": feedback_data.get("other_feedback", ""),
#             # "query_improvement": feedback_data.get("query_improvement", ""),   ##
#             "timestamp": datetime.strptime(feedback_data["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
#         }
        
#         # Convert to DataFrame for easy insertion
#         df = pd.DataFrame([db_feedback])
        
#         # Insert into database
#         df.to_sql("ai_assistant_feedback", engine, if_exists="append", index=False)
#         print("Feedback successfully saved to SQL database.")
        
#     except Exception as e:
#         print(f"Database error: {str(e)}. Feedback saved to JSON file only.")

# @st.cache_resource
# def initialize_components():
#     try:
#         # Check environment variables first
#         check_environment()
        
#         # Initialize Pinecone with better error handling
#         try:
#             pc = Pinecone(
#                 api_key=config.PINECONE_API_KEY,
#                 environment=config.PINECONE_ENVIRONMENT
#             )
#             # Verify Pinecone index exists and is accessible
#             index = pc.Index(config.PINECONE_INDEX_NAME)
#         except Exception as e:
#             st.error(f"Pinecone initialization error: {str(e)}")
#             return None
            
#         # Initialize components one by one with better error handling
#         try:
#             embedding_manager = EmbeddingManager()
#         except Exception as e:
#             st.error(f"Embedding Manager Error: {str(e)}")
#             return None
            
#         try:
#             vector_store = VectorStore()
#         except Exception as e:
#             st.error(f"Vector Store Error: {str(e)}")
#             return None
            
#         try:
#             llm_manager = LLMManager()
#         except Exception as e:
#             st.error(f"LLM Manager Error: {str(e)}")
#             return None
        
#         components = {
#             'embedding_manager': embedding_manager,
#             'vector_store': vector_store,
#             'llm_manager': llm_manager
#         }
        
#         return components
#     except Exception as e:
#         st.error(f"Initialization Error: {str(e)}")
#         st.info("Please check your .env file and ensure all required API keys are set correctly.")
#         return None

# # Initialize response logger if enabled
# response_logger = None
# if RESPONSE_LOGGING_ENABLED:
#     try:
#         # Get database credentials from config or use direct values
#         db_credentials = {
#             "hostname": "your_hostname",  # Replace with actual values
#             "database": "your_database",  # or use config values
#             "username": "your_username",
#             "password": "your_password",
#             "port": "your_port"
#         }
        
#         response_logger = ResponseLogger(
#             log_to_file=True,
#             log_to_db=True,  # Set to False if you don't want to log to the database
#             file_path=Path("response_logs.json"),
#             db_credentials=db_credentials
#         )
#         print("Response logger initialized successfully")
#     except Exception as e:
#         print(f"Failed to initialize response logger: {str(e)}")
#         RESPONSE_LOGGING_ENABLED = False
    
# components = initialize_components()

# if components is None:
#     st.stop()

# embedding_manager = components['embedding_manager']
# vector_store = components['vector_store']
# llm_manager = components['llm_manager']

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "current_sources" not in st.session_state:
#     st.session_state.current_sources = []
# if "context_window" not in st.session_state:
#     st.session_state.context_window = 5
# if "max_history" not in st.session_state:
#     st.session_state.max_history = 10
# if "show_sources" not in st.session_state:
#     st.session_state.show_sources = False
# # Initialize feedback state variables
# if "feedback_submitted" not in st.session_state:
#     st.session_state.feedback_submitted = False
# if "show_feedback" not in st.session_state:
#     st.session_state.show_feedback = False
# if "feedback_positive" not in st.session_state:
#     st.session_state.feedback_positive = False
# if "feedback_negative" not in st.session_state:
#     st.session_state.feedback_negative = False
# if "latest_query" not in st.session_state:
#     st.session_state.latest_query = None
# if "latest_response" not in st.session_state:
#     st.session_state.latest_response = None

# # Add click handlers for feedback buttons
# def on_yes_click():
#     st.session_state.feedback_positive = True
#     st.session_state.feedback_negative = False
    
# def on_no_click():
#     st.session_state.feedback_positive = False
#     st.session_state.feedback_negative = True

# # Add handlers for submit buttons
# def on_submit_positive():
#     time_saved = st.session_state.time_saved
#     rating = st.session_state.rating
#     recommend = st.session_state.recommend
#     liked_aspects = st.session_state.liked_aspects if "liked_aspects" in st.session_state else []
#     other_liked = st.session_state.other_liked if "other_liked" in st.session_state else ""
#     improvement_suggestions = st.session_state.improvement_suggestions if "improvement_suggestions" in st.session_state else ""
#     # query_share = st.session_state.query_share      ### 
#     # customer_experience = st.session_state.customer_experience  ##
#     # usage_frequency = st.session_state.usage_frequency   ###
    
#     feedback_data = {
#         "feedback_type": "positive",
#         # "query": st.session_state.latest_query if query_share == "Yes" else "Not shared",  ##
#         "time_saved": time_saved,
#         "rating": rating,
#         "recommend": recommend,
#         "liked_aspects": liked_aspects,
#         "other_liked": other_liked,
#         "improvement_suggestions": improvement_suggestions,
#         # "customer_experience": customer_experience, ##
#         # "usage_frequency": usage_frequency    ###
#     }
    
#     # Save feedback with response
#     save_feedback(feedback_data, st.session_state.latest_response)
    
#     # Update response logs if enabled
#     if RESPONSE_LOGGING_ENABLED and response_logger:
#         try:
#             # Update the latest log entry with feedback info
#             timestamp = datetime.now().isoformat()
            
#             # Load the existing logs
#             if Path("response_logs.json").exists():
#                 with open("response_logs.json", "r") as f:
#                     logs = json.load(f)
                    
#                 # Update the most recent entry that matches the query
#                 for log in reversed(logs):
#                     if log.get("query") == st.session_state.latest_query:
#                         log["metadata"]["has_feedback"] = True
#                         log["metadata"]["feedback_type"] = "positive"
#                         log["metadata"]["rating"] = rating
#                         log["metadata"]["feedback_timestamp"] = timestamp
#                         break
                
#                 # Save the updated logs
#                 with open("response_logs.json", "w") as f:
#                     json.dump(logs, f, indent=2)
#         except Exception as e:
#             print(f"Error updating response logs: {str(e)}")
    
#     st.session_state.feedback_submitted = True

# def on_submit_negative():
#     issues = st.session_state.issues if "issues" in st.session_state else []
#     other_feedback = st.session_state.other_feedback if "other_feedback" in st.session_state else ""
#     rating = st.session_state.rating
#     recommend = st.session_state.recommend
#     improvement = st.session_state.improvement if "improvement" in st.session_state else ""
#     # query_improvement = st.session_state.query_improvement if "query_improvement" in st.session_state else "" ##
#     # query_share = st.session_state.query_share     ##
#     # customer_experience = st.session_state.customer_experience   ##
#     # usage_frequency = st.session_state.usage_frequency     #
    
#     feedback_data = {
#         "feedback_type": "negative",
#         # "query": st.session_state.latest_query if query_share == "Yes" else "Not shared",  ##
#         "issues": issues,
#         "other_feedback": other_feedback,
#         "rating": rating,
#         "recommend": recommend,
#         "improvement_suggestions": improvement,
#         # "query_improvement": query_improvement,    ##
#         # "customer_experience": customer_experience,   ##
#         # "usage_frequency": usage_frequency          #
#     }
    
#     # Save feedback with response
#     save_feedback(feedback_data, st.session_state.latest_response)
    
#     # Update response logs if enabled
#     if RESPONSE_LOGGING_ENABLED and response_logger:
#         try:
#             # Update the latest log entry with feedback info
#             timestamp = datetime.now().isoformat()
            
#             # Load the existing logs
#             if Path("response_logs.json").exists():
#                 with open("response_logs.json", "r") as f:
#                     logs = json.load(f)
                    
#                 # Update the most recent entry that matches the query
#                 for log in reversed(logs):
#                     if log.get("query") == st.session_state.latest_query:
#                         log["metadata"]["has_feedback"] = True
#                         log["metadata"]["feedback_type"] = "negative"
#                         log["metadata"]["rating"] = rating
#                         log["metadata"]["feedback_timestamp"] = timestamp
#                         break
                
#                 # Save the updated logs
#                 with open("response_logs.json", "w") as f:
#                     json.dump(logs, f, indent=2)
#         except Exception as e:
#             print(f"Error updating response logs: {str(e)}")
    
#     st.session_state.feedback_submitted = True

# st.title(config.APP_TITLE)

# st.markdown("""
# Get answers to all your IndiGo Airlines related queries, from flight information to 
# travel tips, booking procedures and more.
# """)

# # Floating "New Conversation" Button at bottom-right
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

# # Clear session state on button click (handle post request)
# if st.session_state.get("reset_chat", False):
#     st.session_
    
#     ################## new code
#     st.session_state.chat_history = []
#     st.session_state.current_sources = []
#     st.session_state.reset_chat = False
#     st.session_state.feedback_submitted = False
#     st.session_state.show_feedback = False
#     st.session_state.feedback_positive = False
#     st.session_state.feedback_negative = False
#     st.session_state.latest_query = None
#     st.session_state.latest_response = None
#     st.rerun()

# # Use JS to detect button submit and set Streamlit state
# st.markdown("""
#     <script>
#     const form = document.querySelector('.new-convo-button form');
#     form.addEventListener('submit', async function(event) {
#         event.preventDefault();
#         await fetch('', { method: 'POST' });
#         window.parent.postMessage({type: 'streamlit:setComponentValue', value: true}, '*');
#     });
#     </script>
# """, unsafe_allow_html=True)


# # Chat interface
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Display sources if enabled
# if st.session_state.show_sources and st.session_state.current_sources:
#     with st.expander("Source Documents", expanded=False):
#         for i, source in enumerate(st.session_state.current_sources):
#             st.markdown(f"**Source {i+1}**")
#             st.write(source["text"])
#             if "metadata" in source and "url" in source["metadata"]:
#                 st.markdown(f"[Link to source]({source['metadata']['url']})")
#             st.divider()


# # User input
# user_input = st.chat_input("Ask me anything about IndiGo Airlines...")

# # Update the query processing in the main chat interface
# if user_input:
#     # Reset feedback state for new question
#     st.session_state.feedback_submitted = False
#     st.session_state.show_feedback = False
#     st.session_state.feedback_positive = False
#     st.session_state.feedback_negative = False
    
#     # Add user message to chat history
#     st.session_state.chat_history.append({
#         "role": "user",
#         "content": user_input
#     })
    
#     # Display user message
#     with st.chat_message("user"):
#         st.write(user_input)
    
#     # Create a placeholder for the streaming response
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
        
#         try:
#             # Generate embedding for query
#             query_embedding = embedding_manager.generate_embeddings([user_input])[0]
#             relevant_docs = vector_store.search(
#                 user_input,
#                 query_embedding,
#                 k=st.session_state.context_window
#             )
            
#             # Save the current sources for potential display
#             st.session_state.current_sources = relevant_docs

#             # Generate response with streaming
#             response = llm_manager.generate_response(
#                 user_input,
#                 relevant_docs,
#                 st.session_state.chat_history[-st.session_state.max_history:],
#                 streaming_container=response_placeholder
#             )
            
#             # Store the latest query and response for feedback
#             st.session_state.latest_query = user_input
#             st.session_state.latest_response = response
            
#             # Display sources separately
#             if st.session_state.show_sources:
#                 display_sources(relevant_docs)

#             # Update chat history
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": response
#             })
            
#             # Log response if enabled
#             if RESPONSE_LOGGING_ENABLED and response_logger:
#                 try:
#                     metadata = {
#                         "timestamp": datetime.now().isoformat(),
#                         "sources_count": len(relevant_docs),
#                         "has_feedback": False
#                     }
#                     response_logger.log_response(user_input, response, metadata)
#                 except Exception as e:
#                     print(f"Error logging response: {str(e)}")
            
#             # Display feedback request after response
#             st.session_state.show_feedback = True
            
#         except Exception as e:
#             st.error(f"An error occurred during query processing: {str(e)}")
#             st.error("Full error details:")
#             st.exception(e)

# # Display feedback section if needed
# if st.session_state.show_feedback and not st.session_state.feedback_submitted:
#     st.markdown("---")
#     st.markdown("### Feedback")
#     st.write("Did the AI Assistant help you resolve the customer's issue?")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.button("👍 Yes", key="yes_button", on_click=on_yes_click, use_container_width=True)
#     with col2:
#         st.button("👎 No", key="no_button", on_click=on_no_click, use_container_width=True)
    
#     # Show positive feedback form
#     if st.session_state.feedback_positive:
#         st.write("### Thank you for your positive feedback!")
        
#         st.selectbox(
#             "How much time did this save you?",
#             ["Less than 30 seconds", "30 seconds to 1 minute", "1-2 minutes", "More than 2 minutes"],
#             index=0,
#             key="time_saved"
#         )
        
#         st.slider("Rate the AI Assistant's response (1-10)", 1, 10, 7, key="rating")
        
#         st.radio(
#             "Would you recommend this AI Assistant to other friends?",
#             ["Yes", "No"],
#             index=0,
#             key="recommend"
#         )
        
#         st.multiselect(
#             "What specifically did you like about the response?",
#             ["Accuracy of information", "Completeness of answer", 
#             "Clear formatting/organization", "Relevant resources provided", 
#             "Quick retrieval", "Other"],
#             key="liked_aspects"
#         )
        
#         if "liked_aspects" in st.session_state and "Other" in st.session_state.liked_aspects:
#             st.text_input("Please specify what else you liked:", key="other_liked")
        
#         st.text_area("Any suggestions for improvement? (Optional)", key="improvement_suggestions")
        
#         # st.radio(
#         #     "Would you be willing to share your original query for system improvement?",
#         #     ["Yes", "No"],
#         #     index=0,
#         #     key="query_share"
#         # )
        
#         # st.radio(
#         #     "Did using the AI Assistant help improve the customer's experience?",
#         #     ["Yes", "No"],
#         #     index=0,
#         #     key="customer_experience"
#         # )
        
#         # st.selectbox(
#         #     "How often do you use the AI Assistant during customer calls?",
#         #     ["Daily, multiple times per day", "A few times per week", "Rarely", "This is my first time"],
#         #     index=0,
#         #     key="usage_frequency"
#         # )
        
#         st.button("Submit Feedback", key="submit_positive", on_click=on_submit_positive)
        
#         if st.session_state.feedback_submitted:
#             st.success("Thank you for your feedback!")
    
#     # Show negative feedback form
#     if st.session_state.feedback_negative:
#         st.write("### We're sorry the response didn't meet your needs.")
        
#         st.multiselect(
#             "Please select your feedback type:",
#             ["Missing information", "Incorrect information", "Response too complex",
#             "Response not relevant to query", "System too slow", "Other"],
#             key="issues"
#         )
        
#         if "issues" in st.session_state and "Other" in st.session_state.issues:
#             st.text_input("Please specify:", key="other_feedback")
        
#         st.slider("Rate the AI Assistant's response (1-10)", 1, 10, 3, key="rating")
        
#         st.radio(
#             "Would you recommend this AI Assistant to other friends?",
#             ["Yes", "No"],
#             index=1,
#             key="recommend"
#         )
        
#         st.text_area("What would have made this response more helpful? (Optional)", key="improvement")
        
#         # st.text_area("Is there a specific type of query where the AI Assistant needs improvement? (Optional)", key="query_improvement")
        
#         # st.radio(
#         #     "Would you be willing to share your original query for system improvement?",
#         #     ["Yes", "No"],
#         #     index=0,
#         #     key="query_share"
#         # )
        
#         # st.radio(
#         #     "Did using the AI Assistant affect the customer's experience?",
#         #     ["Improved", "No effect", "Worsened"],
#         #     index=1,
#         #     key="customer_experience"
#         # )
        
#         # st.selectbox(
#         #     "How often do you use the AI Assistant during customer calls?",
#         #     ["Daily, multiple times per day", "A few times per week", "Rarely", "This is my first time"],
#         #     index=0,
#         #     key="usage_frequency"
#         # )
        
#         st.button("Submit Feedback", key="submit_negative", on_click=on_submit_negative)
        
#         if st.session_state.feedback_submitted:
#             st.success("Thank you for your feedback!")

    