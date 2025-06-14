# core/scheduled_update.py
import os
import sys
import logging
import time
import schedule
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.web_scraper import IndigoWebScraper
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_website_content():
    """Update the vector database with fresh website content."""
    try:
        logger.info(f"Starting scheduled content update at {datetime.now()}")
        
        # Initialize components
        scraper = IndigoWebScraper()
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        
        # Scrape all target sections
        logger.info("Scraping website content...")
        all_chunks = scraper.scrape_all_sections()
        
        if not all_chunks:
            logger.error("No content was scraped from the website")
            return
        
        logger.info(f"Successfully scraped {len(all_chunks)} chunks of content")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedding_manager.generate_embeddings(
            [chunk['text'] for chunk in all_chunks]
        )
        
        # Upload to vector database
        logger.info("Uploading to vector database...")
        vector_store.add_documents(all_chunks, embeddings)
        
        logger.info("Content update completed successfully")
    except Exception as e:
        logger.error(f"Error during scheduled update: {str(e)}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Schedule regular updates of website content")
    parser.add_argument("--interval", type=int, default=24, help="Update interval in hours")
    parser.add_argument("--run-now", action="store_true", help="Run an update immediately")
    args = parser.parse_args()
    
    if args.run_now:
        logger.info("Running immediate update...")
        update_website_content()
    
    # Schedule regular updates
    logger.info(f"Scheduling updates every {args.interval} hours")
    schedule.every(args.interval).hours.do(update_website_content)
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()