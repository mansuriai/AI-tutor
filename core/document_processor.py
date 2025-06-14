# core/document_processor.py
from typing import List, Dict, Union, Optional
from pathlib import Path
import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config
from utils.helpers import generate_document_id
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentContent:
    """Class to represent different types of content in a document."""
    def __init__(self, content: str, content_type: str, page_num: int, metadata: Dict = None):
        self.content = content
        self.content_type = content_type  # 'text', 'table', 'image_caption'
        self.page_num = page_num
        self.metadata = metadata or {}

# class SmartChunker:
#     """Handles intelligent chunking of different content types."""
#     def __init__(self, embeddings_model: str = config.EMBEDDING_MODEL):
#         self.embedder = HuggingFaceEmbeddings(
#             model_name=embeddings_model,
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=config.CHUNK_SIZE,
#             chunk_overlap=config.CHUNK_OVERLAP,
#             length_function=len,
#         )

class SmartChunker:
    """Handles intelligent chunking of different content types."""
    def __init__(self, embeddings_model: str = config.EMBEDDING_MODEL):
        try:
            self.embedder = HuggingFaceEmbeddings(
                model_name=embeddings_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Error initializing embedder in SmartChunker: {str(e)}")
            # Fallback to a simpler model
            self.embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text chunks."""
        emb1 = self.embedder.embed_documents([text1])[0]
        emb2 = self.embedder.embed_documents([text2])[0]
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def _merge_similar_chunks(self, chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
        """Merge chunks that are semantically similar."""
        if not chunks:
            return chunks

        merged_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            similarity = self._calculate_semantic_similarity(merged_chunks[-1], chunk)
            if similarity > similarity_threshold:
                merged_chunks[-1] = f"{merged_chunks[-1]} {chunk}"
            else:
                merged_chunks.append(chunk)
        return merged_chunks

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text content with semantic understanding."""
        initial_chunks = self.text_splitter.split_text(text)
        return self._merge_similar_chunks(initial_chunks)

    def chunk_table(self, table_df: pd.DataFrame) -> List[str]:
        """Convert table into textual chunks with context."""
        chunks = []
        
        # Add table summary
        summary = f"Table with {len(table_df)} rows and {len(table_df.columns)} columns. "
        # summary += f"Columns: {', '.join(table_df.columns)}. "
        summary += f"Columns: {', '.join(str(col) for col in table_df.columns if col is not None)}. "

        chunks.append(summary)
        
        # Process table in smaller chunks
        row_chunk_size = 5
        for i in range(0, len(table_df), row_chunk_size):
            chunk_df = table_df.iloc[i:i + row_chunk_size]
            chunk_text = chunk_df.to_string(index=False)
            chunks.append(chunk_text)
            
        return chunks

class EnhancedDocumentProcessor:
    """Enhanced document processor with smart chunking and content type handling."""
    def __init__(self):
        self.chunker = SmartChunker()

    def _extract_tables(self, pdf_path: Path) -> List[Dict]:
        """Extract tables from PDF with enhanced metadata."""
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table and any(any(cell for cell in row) for row in table):
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables_data.append({
                            'table': df,
                            'page_num': page_num,
                            'table_num': table_num,
                            'row_count': len(df),
                            'col_count': len(df.columns)
                        })
        return tables_data

    def _extract_text_by_page(self, pdf_reader: PyPDF2.PdfReader) -> List[DocumentContent]:
        """Extract text content by page with structural understanding."""
        contents = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                contents.append(DocumentContent(
                    content=text,
                    content_type='text',
                    page_num=page_num,
                    metadata={'type': 'main_text'}
                ))
        return contents

    def process_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a file with enhanced content type handling and chunking."""
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Extract all content
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_contents = self._extract_text_by_page(pdf_reader)
            tables_data = self._extract_tables(file_path)

        # Process and chunk all content
        processed_chunks = []
        
        # Process text content
        for text_content in text_contents:
            chunks = self.chunker.chunk_text(text_content.content)
            for chunk in chunks:
                processed_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': file_path.name,
                        'chunk_id': generate_document_id(chunk),
                        'page_num': text_content.page_num,
                        'content_type': 'text',
                        'total_pages': len(pdf_reader.pages)
                    }
                })

        # Process tables
        for table_data in tables_data:
            table_chunks = self.chunker.chunk_table(table_data['table'])
            for chunk in table_chunks:
                processed_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': file_path.name,
                        'chunk_id': generate_document_id(chunk),
                        'page_num': table_data['page_num'],
                        'content_type': 'table',
                        'table_num': table_data['table_num'],
                        'row_count': table_data['row_count'],
                        'col_count': table_data['col_count'],
                        'total_pages': len(pdf_reader.pages)
                    }
                })

        return processed_chunks