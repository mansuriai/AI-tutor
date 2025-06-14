from typing import List, Dict
import asyncio

class RetrievalOptimizer:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    async def get_relevant_chunks(
        self,
        query: str,
        embedding: List[float],
        k: int = 3
    ) -> List[Dict]:
        """Get relevant chunks asynchronously."""
        # Perform search
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_store.search(query, embedding, k)
        )
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x['distance'])
        