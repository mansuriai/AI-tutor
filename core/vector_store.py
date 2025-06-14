# # core/vector_store.py
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import streamlit as st
from utils.config import config
import urllib3
import ssl
import requests
import traceback
import logging

#################
## Please comment this line while working on local machine
# import sys
# sys.modules["sqlite3"] = __import__("pysqlite3")
####################

class VectorStore:

    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize Pinecone with retries and detailed logging
            self.pc = Pinecone(
                api_key=config.PINECONE_API_KEY,
                environment=config.PINECONE_ENVIRONMENT
            )

            # Verify index exists or create it
            self._ensure_index_exists()

            # Connect to the index
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            
            # Log successful initialization
            self.logger.info(f"Successfully connected to Pinecone index: {config.PINECONE_INDEX_NAME}")

        except Exception as e:
            # Detailed error logging
            self.logger.error(f"Pinecone Initialization Error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if necessary."""
        try:
            # List existing indexes
            existing_indexes = self.pc.list_indexes().names()
            
            # Create index if it doesn't exist
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                self.logger.info(f"Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=config.EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                self.logger.info(f"Index {config.PINECONE_INDEX_NAME} created successfully")
        
        except Exception as e:
            self.logger.error(f"Error ensuring index exists: {str(e)}")
            raise
    
    def _initialize_cache(self):
        """Initialize an in-memory cache for frequent queries."""
        self.cache = {}
        self.cache_size = 1000  # This can be Adjusted based on our needs
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return str(hash(query))
    
    def add_documents(self, documents: List[Dict[str, str]], embeddings: List[List[float]]):
        """Add documents and their embeddings to Pinecone with improved error handling."""
        try:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                vectors = []
                for doc, embedding in zip(batch_docs, batch_embeddings):
                    vectors.append({
                        'id': doc['metadata'].get('chunk_id', str(hash(doc['text']))),
                        'values': embedding,
                        'metadata': {
                            'text': doc['text'],
                            **doc.get('metadata', {})
                        }
                    })
                
                try:
                    self.index.upsert(vectors=vectors)
                    self.logger.info(f"Successfully upserted {len(vectors)} vectors")
                except Exception as batch_error:
                    self.logger.error(f"Error upserting batch: {str(batch_error)}")
        
        except Exception as e:
            self.logger.error(f"Document addition error: {str(e)}")
            raise

    def search(self, query: str, embedding: List[float], k: int = 3) -> List[Dict]:
        """Enhanced search with better source handling."""
        try:
            results = self.index.query(
                vector=embedding,
                top_k=k,
                include_metadata=True
            )
            
            processed_results = []
            for match in results.matches:
                metadata = match.metadata or {}
                
                # Enhance metadata with heading information if available
                if 'text' in metadata and 'headings' in metadata:
                    text = metadata['text']
                    for heading in metadata['headings']:
                        if heading['text'].lower() in text.lower():
                            metadata['url'] = f"{metadata.get('url', '')}#{heading['id']}"
                            break
                
                processed_results.append({
                    'text': metadata.get('text', ''),
                    'metadata': metadata,
                    'distance': 1 - match.score
                })
            
            return processed_results
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []
    
    def get_existing_hashes(self) -> Dict[str, str]:
        """Get a mapping of section names to content hashes for change detection."""
        try:
            # Query for metadata containing hash information
            section_hashes = {}
            
            # Use a metadata filter to get distinct parent_hash values
            results = self.index.query(
                vector=[0.0] * config.EMBEDDING_DIMENSION,  # Dummy vector
                top_k=1000,  # Get a large number to capture all sections
                include_metadata=True,
                filter={
                    "section": {"$exists": True},
                    "parent_hash": {"$exists": True}
                }
            )
            
            # Extract unique section to hash mappings
            for match in results.matches:
                if "section" in match.metadata and "parent_hash" in match.metadata:
                    section = match.metadata["section"]
                    parent_hash = match.metadata["parent_hash"]
                    section_hashes[section] = parent_hash
            
            return section_hashes
            
        except Exception as e:
            self.logger.error(f"Error getting existing hashes: {str(e)}")
            return {}
    
    def delete_by_parent_hash(self, parent_hashes: List[str]) -> int:
        """Delete all vectors associated with specific parent hashes (deleted content)."""
        try:
            deleted_count = 0
            
            # Process in batches if there are many hashes
            batch_size = 10
            for i in range(0, len(parent_hashes), batch_size):
                batch_hashes = parent_hashes[i:i + batch_size]
                
                # Create a filter for this batch of hashes
                delete_filter = {
                    "parent_hash": {"$in": batch_hashes}
                }
                
                # Delete matching vectors
                result = self.index.delete(filter=delete_filter)
                
                # Track the number of deleted items
                if hasattr(result, 'deleted_count'):
                    deleted_count += result.deleted_count
                
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting vectors by parent hash: {str(e)}")
            return 0