# app/api_main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
from pinecone import Pinecone, ServerlessSpec
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add the parent directory to the path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore
from core.llm import LLMManager

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A REST API for AI-Tutor RAG Chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []
    context_window: Optional[int] = 5
    max_history: Optional[int] = 10
    include_sources: Optional[bool] = False

class SourceDocument(BaseModel):
    text: str
    metadata: Dict

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[SourceDocument]] = []
    chat_history: List[ChatMessage]

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for components
embedding_manager = None
vector_store = None
llm_manager = None

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
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        raise ValueError(error_msg)

def initialize_components():
    """Initialize all components needed for the RAG system."""
    try:
        # Check environment variables first
        check_environment()
        
        # Initialize Pinecone
        pc = Pinecone(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT
        )
        
        # Verify Pinecone index exists and is accessible
        index = pc.Index(config.PINECONE_INDEX_NAME)
        
        # Initialize components
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        llm_manager = LLMManager()
        
        return embedding_manager, vector_store, llm_manager
        
    except Exception as e:
        raise Exception(f"Initialization Error: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedding_manager, vector_store, llm_manager
    try:
        embedding_manager, vector_store, llm_manager = initialize_components()
        print("✅ All components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {str(e)}")
        raise e

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running."""
    if embedding_manager is None or vector_store is None or llm_manager is None:
        raise HTTPException(status_code=503, detail="Service components not initialized")
    
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running successfully"
    )

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, stream: bool = True):
    """Main chat endpoint for processing user queries."""
    global embedding_manager, vector_store, llm_manager
    # Check if components are initialized
    if embedding_manager is None or vector_store is None or llm_manager is None:
        raise HTTPException(status_code=503, detail="Service components not initialized")
    try:
        # Generate embedding for the user query
        query_embedding = embedding_manager.generate_embeddings([request.message])[0]
        # Search for relevant documents
        relevant_docs = vector_store.search(
            request.message,
            query_embedding,
            k=request.context_window
        )
        # Convert chat history to the format expected by LLM manager
        chat_history = []
        for msg in request.chat_history:
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        # Limit chat history
        if len(chat_history) > request.max_history:
            chat_history = chat_history[-request.max_history:]
        if stream:
            def chat_stream():
                for token in llm_manager.stream_response(
                    request.message,
                    relevant_docs,
                    chat_history
                ):
                    yield token
            return StreamingResponse(chat_stream(), media_type="text/plain")
        # Generate response using LLM manager
        response = llm_manager.generate_response(
            request.message,
            relevant_docs,
            chat_history
        )
        # Prepare sources if requested
        sources = []
        if request.include_sources:
            for doc in relevant_docs:
                sources.append(SourceDocument(
                    text=doc["text"],
                    metadata=doc.get("metadata", {})
                ))
        # Update chat history with the new exchange
        updated_chat_history = chat_history + [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response)
        ]
        return ChatResponse(
            response=response,
            sources=sources if request.include_sources else [],
            chat_history=updated_chat_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Endpoint to search documents only
@app.post("/search")
async def search_documents(query: str, k: int = 5):
    """Search for relevant documents without generating a response."""
    global embedding_manager, vector_store
    
    if embedding_manager is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service components not initialized")
    
    try:
        # Generate embedding for the query
        query_embedding = embedding_manager.generate_embeddings([query])[0]
        
        # Search for relevant documents
        relevant_docs = vector_store.search(query, query_embedding, k=k)
        
        # Format response
        sources = []
        for doc in relevant_docs:
            sources.append(SourceDocument(
                text=doc["text"],
                metadata=doc.get("metadata", {})
            ))
        
        return {"query": query, "sources": sources}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# Endpoint to get system information
@app.get("/info")
async def get_system_info():
    """Get system information and configuration."""
    return {
        "app_title": config.APP_TITLE,
        "pinecone_index": config.PINECONE_INDEX_NAME,
        "environment": config.PINECONE_ENVIRONMENT,
        "components_initialized": all([
            embedding_manager is not None,
            vector_store is not None,
            llm_manager is not None
        ])
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "search": "/search",
            "info": "/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )