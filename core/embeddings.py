# core/embeddings.py

from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from utils.config import config

class EmbeddingManager:
    def __init__(self):
        print(f"Loading model from: {config.EMBEDDING_MODEL}")
        # Load the model using transformers directly
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL, add_pooling_layer=False)
        self.model.eval()
        self.query_prefix = 'query: '
        
        # Verify dimensions
        test_embedding = self.generate_embeddings(["Test dimension check"])
        if test_embedding and len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
            raise ValueError(f"Model produces embeddings with dimension {len(test_embedding[0])}, but config.EMBEDDING_DIMENSION is set to {config.EMBEDDING_DIMENSION}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # Add query prefix for all texts (assuming they could be either documents or queries)
        # In a production system, you might want to separate query and document embedding functions
        texts_with_prefix = [f"{self.query_prefix}{text}" for text in texts]
        
        # Tokenize and generate embeddings
        tokens = self.tokenizer(texts_with_prefix, padding=True, truncation=True, 
                               return_tensors='pt', max_length=8192)
        
        with torch.no_grad():
            outputs = self.model(**tokens)[0]
            # Get CLS token embeddings (first token of each sequence)
            embeddings = outputs[:, 0]
            
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to Python lists for compatibility with the rest of your code
        return embeddings.tolist()