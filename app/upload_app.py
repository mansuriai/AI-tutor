# app/upload_app.py
import streamlit as st
from pathlib import Path
import time
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from fix_ssl import *
from utils.config import config
# from utils.s3_manager import S3Manager
from core.document_processor import EnhancedDocumentProcessor
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize components
doc_processor = EnhancedDocumentProcessor()
embedding_manager = EmbeddingManager()
vector_store = VectorStore()

st.set_page_config(
    page_title=f"{config.APP_TITLE} - Document Upload",
    layout="wide"
)

st.title(f"{config.APP_TITLE} - Document Upload")

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    st.write(f"{len(st.session_state.uploaded_files)} documents ready for processing.")
    
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            for file in st.session_state.uploaded_files:
                # Save PDF to storage directory
                pdf_path = config.PDF_STORAGE_DIR / file.name
                with open(pdf_path, 'wb') as f:
                    f.write(file.getvalue())
                
                # Process for vector store
                file_path = config.DATA_DIR / file.name
                with open(file_path, 'wb') as f:
                    f.write(file.getvalue())
                
                # Process documents
                chunks = doc_processor.process_file(file_path)
                
                # Generate embeddings
                embeddings = embedding_manager.generate_embeddings(
                    [chunk['text'] for chunk in chunks]
                )
                
                # Store in vector database
                vector_store.add_documents(chunks, embeddings)
            
            st.success("Documents processed and indexed!")
            st.session_state.uploaded_files = []

    ######### working
    # if st.button("Process Documents"):
    #     with st.spinner("Processing documents..."):
    #         for file in st.session_state.uploaded_files:
    #             file_path = config.DATA_DIR / file.name
    #             with open(file_path, 'wb') as f:
    #                 f.write(file.getvalue())
                
    #             # Process documents
    #             chunks = doc_processor.process_file(file_path)
                
    #             # Generate embeddings
    #             embeddings = embedding_manager.generate_embeddings(
    #                 [chunk['text'] for chunk in chunks]
    #             )
                
    #             # Store in vector database
    #             vector_store.add_documents(chunks, embeddings)
            
    #         st.success("Documents processed and indexed!")
    #         st.session_state.uploaded_files = []  # Clear uploaded files after processing
