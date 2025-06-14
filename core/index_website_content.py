# core/index_website_content.py
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.web_scraper import IndigoWebScraper
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore
from utils.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to scrape and index website content with change detection."""
    logger.info("Starting website content indexing with change detection")
    
    # Initialize components
    scraper = IndigoWebScraper()
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    
    # Get existing content hashes to detect changes
    existing_hashes = vector_store.get_existing_hashes()
    logger.info(f"Found {len(existing_hashes)} existing sections in vector store")
    
    # Scrape with change detection
    logger.info("Scraping website content with change detection...")
    all_chunks, deleted_hashes = scraper.scrape_with_changes(existing_hashes)
    
    # Handle deleted content
    if deleted_hashes:
        deleted_count = vector_store.delete_by_parent_hash(deleted_hashes)
        logger.info(f"Deleted {deleted_count} vectors from removed content")
    
    # Process new/updated content
    if all_chunks:
        logger.info(f"Processing {len(all_chunks)} new/updated chunks")
        
        # Generate embeddings only for changed content
        logger.info("Generating embeddings for changed content...")
        embeddings = embedding_manager.generate_embeddings(
            [chunk['text'] for chunk in all_chunks]
        )
        
        # Upload to vector database
        logger.info("Uploading to vector database...")
        vector_store.add_documents(all_chunks, embeddings)
        
        logger.info(f"Successfully indexed {len(all_chunks)} new/updated chunks")
    else:
        logger.info("No content changes detected - nothing to update")
    
    logger.info("Website content indexing completed")

if __name__ == "__main__":
    main()