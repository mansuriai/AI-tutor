# # # utils/helpers.py
# import hashlib
# from typing import List, Dict, Any

# def generate_document_id(content: str) -> str:
#     """Generate a unique ID for a document based on its content."""
#     return hashlib.md5(content.encode()).hexdigest()

# def format_chat_history(history: List[Dict[str, Any]]) -> str:
#     """Format chat history for context window."""
#     formatted = []
#     for message in history:
#         role = message["role"]
#         content = message["content"]
#         formatted.append(f"{role.capitalize()}: {content}")
#     return "\n".join(formatted)




################## new code 2 ##################
import hashlib
import re
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse, parse_qs, quote

def generate_document_id(content: str) -> str:
    """Generate a unique ID for a document based on its content."""
    return hashlib.md5(content.encode()).hexdigest()

def format_chat_history(history: List[Dict[str, Any]]) -> str:
    """Format chat history for context window."""
    formatted = []
    for message in history:
        role = message["role"]
        content = message["content"]
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

def create_fragment_identifier(text: str) -> str:
    """Create a URL fragment identifier based on text content."""
    # Clean the text to create a valid fragment
    # Take first 10 words or less
    words = re.sub(r'[^\w\s]', '', text).strip().split()[:10]
    fragment = "-".join(words).lower()
    return quote(fragment)

def calculate_relevance_score(doc_text: str, query: str) -> float:
    """Calculate a simple relevance score between document chunk and query."""
    # Convert to lowercase for comparison
    doc_lower = doc_text.lower()
    query_lower = query.lower()
    
    # Split into words
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Count how many query words appear in the document
    matches = sum(1 for word in query_words if word in doc_lower)
    
    # Calculate score based on percentage of matched words
    if not query_words:
        return 0
    
    return matches / len(query_words)

def generate_source_links(context: List[Dict], query: str) -> str:
    """Generate source links with relevance-based ordering and fragment identifiers."""
    if not context:
        return ""
    
    # Add relevance score to each document
    for doc in context:
        doc['relevance'] = calculate_relevance_score(doc['text'], query)
    
    # Sort by relevance score, highest first
    sorted_docs = sorted(context, key=lambda x: x['relevance'], reverse=True)
    
    # Deduplicate by URL
    unique_sources = {}
    for doc in sorted_docs:
        metadata = doc.get('metadata', {})
        url = metadata.get('url', '')
        
        if url and url not in unique_sources:
            # Create fragment identifier from text
            fragment = create_fragment_identifier(doc['text'])
            source_url = f"{url}#{fragment}" if fragment else url
            
            # Store source with section info if available
            section = metadata.get('section', '')
            source_name = f"IndiGo Airlines - {section.replace('_', ' ').title()}" if section else "IndiGo Airlines"
            
            unique_sources[url] = {
                'name': source_name,
                'url': source_url,
                'relevance': doc['relevance']
            }
    
    # Format links in markdown, limiting to top 3 most relevant
    links = []
    for i, source in enumerate(list(unique_sources.values())[:3], 1):
        links.append(f"{i}. [{source['name']}]({source['url']})")
    
    return "\n".join(links)