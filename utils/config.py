# utils/config.py
import os

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()

class Config:
    # Project structure
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DB_DIR = BASE_DIR / "storage" / "vectordb"
    # MODEL_DIR = BASE_DIR / "models" / "all-mpnet-base-v2"   ####
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    PDF_STORAGE_DIR = BASE_DIR / "storage" / "pdfs"
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # SQL Database
    DB_HOSTNAME = os.getenv("DB_HOSTNAME")
    DB_NAME = os.getenv("DB_NAME")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0" 
    # EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
    # model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    EMBEDDING_DIMENSION = 1024  # Adjust based on your specific embedding model
    # EMBEDDING_MODEL = str(MODEL_DIR)
    LLM_MODEL = "gpt-4.1-mini"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    PINECONE_INDEX_NAME = "ai-tutor" #os.getenv("PINECONE_INDEX_NAME", "indigo-assistant")
    
    # App settings
    APP_TITLE = "AI Tutor"
    MAX_HISTORY_LENGTH = 8
    
    # Vector DB settings
    COLLECTION_NAME = "ai-documents"
    DISTANCE_METRIC = "cosine"
    
    # Pinecone index settings
    PINECONE_INDEX_SPEC = {
        "cloud": "aws",
        "region": "us-east-1",
        "metric": "cosine"
    }
    
    # Web scraping settings
    WEB_SCRAPING_DELAY = 1  # Delay between requests in seconds
    
config = Config()