# response_logger.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import pandas as pd

class ResponseLogger:
    """A utility class to log and store bot responses for analysis."""
    
    def __init__(
        self, 
        log_to_file: bool = True,
        log_to_db: bool = False,
        file_path: Optional[Path] = None,
        db_credentials: Optional[Dict[str, str]] = None
    ):
        self.log_to_file = log_to_file
        self.log_to_db = log_to_db
        self.file_path = file_path or Path("response_logs.json")
        self.db_credentials = db_credentials
        self.db_engine = None
        
        if self.log_to_db and db_credentials:
            self._setup_database_connection()
    
    def _setup_database_connection(self):
        """Set up the database connection if credentials are provided."""
        try:
            # Extract database credentials
            hostname = self.db_credentials.get("hostname", "")
            database = self.db_credentials.get("database", "")
            username = self.db_credentials.get("username", "")
            password = self.db_credentials.get("password", "")
            port = self.db_credentials.get("port", "")
            
            # Create connection string
            connection_string = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/{database}"
            
            # Create engine
            self.db_engine = create_engine(connection_string)
            
            # Test connection
            with self.db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            # Create table if it doesn't exist
            self._create_responses_table()
            
            print("Database connection for response logging established successfully.")
        except Exception as e:
            print(f"Failed to set up database connection for response logging: {str(e)}")
            self.log_to_db = False
    
    def _create_responses_table(self):
        """Create the responses table if it doesn't exist."""
        if not self.db_engine:
            return
        
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS ai_assistant_responses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                query_time DATETIME NOT NULL,
                metadata TEXT
            );
            """
            
            with self.db_engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
        except Exception as e:
            print(f"Error creating responses table: {str(e)}")
    
    def log_response(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a response to file and/or database."""
        timestamp = datetime.now().isoformat()
        
        # Prepare the log entry
        log_entry = {
            "query": query,
            "response": response,
            "query_time": timestamp,
            "metadata": metadata or {}
        }
        
        # Log to file if enabled
        if self.log_to_file:
            self._log_to_file(log_entry)
        
        # Log to database if enabled and connected
        if self.log_to_db and self.db_engine:
            self._log_to_database(log_entry)
    
    def _log_to_file(self, log_entry: Dict[str, Any]):
        """Log the response to a JSON file."""
        try:
            # Load existing logs if file exists
            if self.file_path.exists():
                with open(self.file_path, "r") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            else:
                logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(self.file_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error logging response to file: {str(e)}")
    
    def _log_to_database(self, log_entry: Dict[str, Any]):
        """Log the response to the database."""
        try:
            # Convert metadata to JSON string
            log_entry_db = log_entry.copy()
            log_entry_db["metadata"] = json.dumps(log_entry["metadata"])
            
            # Convert timestamp to datetime object
            log_entry_db["query_time"] = datetime.fromisoformat(log_entry["query_time"])
            
            # Create DataFrame for easy insertion
            df = pd.DataFrame([log_entry_db])
            
            # Insert into database
            df.to_sql("ai_assistant_responses", self.db_engine, if_exists="append", index=False)
        except Exception as e:
            print(f"Error logging response to database: {str(e)}")