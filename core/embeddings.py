# # core/embeddings.py
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import os
# from utils.config import config
# from typing import List

# class EmbeddingManager:
#     def __init__(self): 
#         print(f"Loading model from: {config.EMBEDDING_MODEL}")
#         # Verify environment variable is set
#         assert 'HF_ENDPOINT' in os.environ, "HF_ENDPOINT not set!"
#         # Make sure we're using the correct model with the right dimensions
#         self.model = HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             model_kwargs={'device': 'cpu', "trust_remote_code": True },
#             encode_kwargs={'normalize_embeddings': True},
#         )
#         # Add a dimension check during initialization
#         test_embedding = self.model.embed_documents(["Test dimension check"])
#         if test_embedding and len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
#             raise ValueError(f"Model produces embeddings with dimension {len(test_embedding[0])}, but config.EMBEDDING_DIMENSION is set to {config.EMBEDDING_DIMENSION}")
    
#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for a list of texts."""
#         embeddings = self.model.embed_documents(texts)
        
#         # Double-check dimensions before returning
#         if embeddings and len(embeddings[0]) != config.EMBEDDING_DIMENSION:
#             raise ValueError(f"Generated embeddings have dimension {len(embeddings[0])}, but expected {config.EMBEDDING_DIMENSION}")
            
#         return embeddings




# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
# import os
# from utils.config import config
# from typing import List
# import warnings
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Suppress unnecessary warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# class EmbeddingManager:
#     def __init__(self):
#         print(f"Initializing embedding model: {config.EMBEDDING_MODEL}")
        
#         # Set environment variables as fallback
#         os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
#         os.environ['HF_HUB_CACHE'] = './models'
        
#         try:
#             self.model = self._initialize_model()
#             self._verify_embedding_dimensions()
#         except Exception as e:
#             print(f"Failed to load primary model: {str(e)}")
#             self.model = self._initialize_fallback_model()

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#     def _initialize_model(self):
#         """Initialize with retry logic"""
#         return HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             model_kwargs={
#                 'device': 'cpu',
#                 'trust_remote_code': True,
#                 'cache_dir': './models'
#             },
#             encode_kwargs={
#                 'normalize_embeddings': True,
#                 'show_progress_bar': False
#             }
#         )

#     def _initialize_fallback_model(self):
#         """Fallback to smaller local model"""
#         print("Using fallback model: all-MiniLM-L6-v2")
#         return HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )

#     def _verify_embedding_dimensions(self):
#         """Verify embedding dimensions match config"""
#         test_text = "Dimension verification"
#         embedding = self.generate_embeddings([test_text])[0]
#         if len(embedding) != config.EMBEDDING_DIMENSION:
#             raise ValueError(
#                 f"Embedding dimension mismatch. Expected {config.EMBEDDING_DIMENSION}, "
#                 f"got {len(embedding)}"
#             )

#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings with error handling"""
#         try:
#             return self.model.embed_documents(texts)
#         except Exception as e:
#             print(f"Embedding generation error: {str(e)}")
#             raise RuntimeError("Failed to generate embeddings") from e















#############################################

# core/embeddings.py
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# import os
# from utils.config import config
# from typing import List
# from sentence_transformers import SentenceTransformer

# class EmbeddingManager:
#     def __init__(self): 
#         print(f"Loading model from: {config.EMBEDDING_MODEL}")
#         # Make sure we're using the correct model with the right dimensions
#         self.model = HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True},
#         )
        
#         # Add a dimension check during initialization
#         test_embedding = self.model.embed_documents(["Test dimension check"])
#         if test_embedding and len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
#             raise ValueError(f"Model produces embeddings with dimension {len(test_embedding[0])}, but config.EMBEDDING_DIMENSION is set to {config.EMBEDDING_DIMENSION}")
    
#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for a list of texts."""
#         embeddings = self.model.embed_documents(texts)
        
#         # Double-check dimensions before returning
#         if embeddings and len(embeddings[0]) != config.EMBEDDING_DIMENSION:
#             raise ValueError(f"Generated embeddings have dimension {len(embeddings[0])}, but expected {config.EMBEDDING_DIMENSION}")
            
#         return embeddings

# class EmbeddingManager:
#     def __init__(self): 
#         print(f"Loading model from: {config.EMBEDDING_MODEL}")

        

#         # Load the model
#         # model_name = config.EMBEDDING_MODEL #'Snowflake/snowflake-arctic-embed-l-v2.0'
#         model = SentenceTransformer(config.EMBEDDING_MODEL)
#         del model

#         # Updated initialization with proper error handling
#         try:
#             self.model = HuggingFaceEmbeddings(
#                 model_name=config.EMBEDDING_MODEL,
#                 model_kwargs={'device': 'cpu'},
#                 encode_kwargs={'normalize_embeddings': True},
#             )
            
#             # Add a dimension check during initialization
#             test_embedding = self.model.embed_documents(["Test dimension check"])
#             if test_embedding and len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
#                 print(f"Warning: Model produces embeddings with dimension {len(test_embedding[0])}, but config.EMBEDDING_DIMENSION is set to {config.EMBEDDING_DIMENSION}")
#                 # Update the dimension in config rather than raising an error
#                 config.EMBEDDING_DIMENSION = len(test_embedding[0])
#         except Exception as e:
#             print(f"Error initializing embedding model: {str(e)}")
#             # Fallback to a simpler model if the main one fails
#             try:
#                 # Use a more reliable, simpler embedding model as fallback
#                 self.model = HuggingFaceEmbeddings(
#                     model_name="sentence-transformers/all-mpnet-base-v2",
#                     model_kwargs={'device': 'cpu'},
#                     encode_kwargs={'normalize_embeddings': True},
#                 )
#                 test_embedding = self.model.embed_documents(["Test dimension check"])
#                 config.EMBEDDING_DIMENSION = len(test_embedding[0])
#                 print(f"Using fallback model with dimension {config.EMBEDDING_DIMENSION}")
#             except Exception as e2:
#                 raise RuntimeError(f"Failed to initialize embedding model: {str(e2)}")
    
#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for a list of texts."""
#         try:
#             embeddings = self.model.embed_documents(texts)
#             return embeddings
#         except Exception as e:
#             print(f"Error generating embeddings: {str(e)}")
#             raise



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



# from sentence_transformers import SentenceTransformer
# import os
# from utils.config import config
# from typing import List
# import warnings

# warnings.filterwarnings("ignore")

# class EmbeddingManager:
#     def __init__(self):
#         print(f"Loading model from: {config.EMBEDDING_MODEL}")
#         os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
#         try:
#             # Try loading with SentenceTransformers directly
#             self.model = SentenceTransformer(
#                 config.EMBEDDING_MODEL,
#                 device='cpu',
#                 cache_folder='./models'
#             )
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             # # Fallback to smaller model
#             # self.model = SentenceTransformer(
#             #     'all-MiniLM-L6-v2',
#             #     device='cpu'
#             # )
        
#         # Verify dimensions
#         test_embedding = self.encode(["Test"])
#         if len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
#             raise ValueError(
#                 f"Dimension mismatch: Expected {config.EMBEDDING_DIMENSION}, "
#                 f"got {len(test_embedding[0])}"
#             )

#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings using the SentenceTransformer directly"""
#         return self.encode(texts)

#     def encode(self, texts: List[str]) -> List[List[float]]:
#         """Wrapper for SentenceTransformer's encode"""
#         return self.model.encode(
#             texts,
#             normalize_embeddings=True,
#             show_progress_bar=False
#         ).tolist()