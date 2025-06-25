# # core/llm.py

from typing import List, Dict, Optional, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from utils.config import config
from utils.helpers import format_chat_history
import json
import re
from urllib.parse import urlparse

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,
            api_key=config.OPENAI_API_KEY,
            streaming=True
        )

        self.system_prompt = """You are an AI assistant designed to provide clear, detailed, and accurate answers to user queries based on the provided context.
 
            IMPORTANT GUIDELINES:
            1. Provide comprehensive, detailed responses that fully answer the user's question.
            2. Use only the information provided in the context to answer questions.
            3. If the context doesn't contain enough information, clearly state what you know and what information would be helpful.
            4. Use a friendly, professional tone.
            5. Structure your answers clearly using paragraphs, bullet points, or numbered lists when appropriate.
            6. Include relevant details such as policies, procedures, requirements, or exceptions when applicable.
            7. Do not mention or reference that you're using "context" or "sources" in your response.
            8. If you don't have enough information to answer accurately, clearly state this rather than guessing.
            9. Add "$$" and "$$" around any latex generated content
            
            Current context information:
            {{context}}
            
            Previous conversation history:
            {{chat_history}}
            
            Please provide a helpful and accurate response based on the above information.
            
            Provide answers in detailed steps like if its a process question or a 'How to' types question eg "How to add a new location"
            Step 1 : Instructions of Step 1
            Step 2: Instructions of Step 2
        """

        self.human_prompt = "{question}"
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.human_prompt)
        ])
        
        # Non-streaming LLM for clarification assessment
        self.analysis_llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY,
            streaming=False
        )
        
        self.clarification_system_prompt = """You are an AI assistant helping to determine if a user query needs clarification before providing a full response.
        
        Given the user's question and the available context information, determine:
        1. If the question is clear and specific enough to answer accurately with the available context
        2. If not, what specific clarifying questions would help provide a better answer
        3. Add "$$" and "$$" around any latex generated content

        Context information:
        {context}

        User question:
        {question}

        Previous conversation:
        {chat_history}

        Respond in JSON format with two fields:
        - "needs_clarification": Boolean (true/false)
        - "clarifying_questions": Array of strings (1-3 specific questions to ask the user if needed)
        - "reasoning": Brief explanation of why clarification is or isn't needed
        """
        
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", self.clarification_system_prompt)
        ])
        
        self.clarification_chain = (
            {"context": RunnablePassthrough(), 
             "chat_history": RunnablePassthrough(), 
             "question": RunnablePassthrough()}
            | self.clarification_prompt
            | self.analysis_llm
            | StrOutputParser()
        )
    
    def extract_source_links(self, context_docs: List[Dict]) -> List[str]:
        """Extract unique source URLs from context documents."""
        sources = []
        for doc in context_docs:
            if 'metadata' in doc and 'url' in doc['metadata']:
                url = doc['metadata']['url']
                if url and url not in sources:
                    sources.append(url)
        return sources
    
    def needs_clarification(
        self,
        question: str,
        context: List[Dict],
        chat_history: Optional[List[Dict]] = None
    ) -> Tuple[bool, List[str]]:
        """Determine if the query needs clarification and suggest clarifying questions."""
        formatted_context = "\n\n".join([doc['text'] for doc in context])
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        
        try:
            response = self.clarification_chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
            
            # Parse JSON response
            result = json.loads(response)
            needs_clarification = result.get("needs_clarification", False)
            questions = result.get("clarifying_questions", [])
            
            return needs_clarification, questions
        except Exception as e:
            # If any error occurs, default to not needing clarification
            print(f"Error in clarification assessment: {str(e)}")
            return False, []
    
    def _extract_anchor_text(self, text: str, url: str) -> str:
        """Extract relevant anchor text for a URL from the content."""
        # Simple implementation - can be enhanced with NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if urlparse(url).path.lower() in sentence.lower():
                return sentence
        return ""
    
    def format_source_references(self, context_docs: List[Dict]) -> str:
        """Format source references with anchor texts and direct links."""
        source_lines = []
        seen_urls = set()
        
        for doc in context_docs:
            if 'metadata' not in doc:
                continue
                
            url = doc['metadata'].get('url', '')
            if not url or url in seen_urls:
                continue
                
            seen_urls.add(url)
            anchor_text = self._extract_anchor_text(doc['text'], url)
            
            if anchor_text:
                source_lines.append(f"🔗 {anchor_text} [Read more]({url})")
            else:
                source_lines.append(f"🔗 [Source]({url})")
        
        if source_lines:
            return "\n\n**References:**\n" + "\n".join(source_lines)
        return ""
    
    def post_process_mathematical_content(self, text: str) -> str:
        """Post-process the LLM response to ensure mathematical content is readable."""
        
        # Dictionary of LaTeX expressions to replace
        latex_replacements = {
            r'\\text\{([^}]+)\}': r'\1',  # Remove \text{} wrapper
            r'\\times': '×',
            r'\\div': '÷',
            r'\\cdot': '·',
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',  # Convert fractions
            r'\\leq': '≤',
            r'\\geq': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'\\sum': '∑',
            r'\\pi': 'π',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\sigma': 'σ',
            r'\\mu': 'μ',
            r'\\lambda': 'λ',
            r'\\theta': 'θ',
            r'\\phi': 'φ',
            r'\\omega': 'ω',
            r'\\sqrt': '√',
            r'\\_': '_',  # Handle escaped underscores
        }
        
        # Apply replacements
        result = text
        for pattern, replacement in latex_replacements.items():
            if pattern == r'\\text\{([^}]+)\}':
                result = re.sub(pattern, replacement, result)
            elif pattern == r'\\frac\{([^}]+)\}\{([^}]+)\}':
                result = re.sub(pattern, replacement, result)
            else:
                result = result.replace(pattern.replace('\\', ''), replacement)
        
        # Clean up subscripts and superscripts formatting
        result = re.sub(r'_\{([^}]+)\}', r'_\1', result)
        result = re.sub(r'\^\{([^}]+)\}', r'^\1', result)
        
        # Remove remaining LaTeX delimiters
        result = re.sub(r'\$\$?([^$]+)\$\$?', r'\1', result)
        result = re.sub(r'\\\[([^\]]+)\\\]', r'\1', result)
        result = re.sub(r'\\\(([^)]+)\\\)', r'\1', result)
        
        return result
    
    def generate_response(
        self,
        question: str,
        context: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        streaming_container = None
    ) -> str:
        """Generate a comprehensive response with proper source attribution and mathematical formatting."""
        # Check for clarification needs
        needs_clarification, clarifying_questions = self.needs_clarification(
            question, context, chat_history
        )
        
        # Generate the main response
        formatted_context = "\n\n".join([
            f"CONTEXT {i+1}:\n{doc['text']}\n" 
            for i, doc in enumerate(context)
        ])
        
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        
        # If streaming is requested, use a streaming handler
        if streaming_container:
            stream_handler = StreamHandler(streaming_container)
            
            # Create a streaming LLM
            streaming_llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0.7,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            # Create a chain with the streaming LLM
            streaming_chain = (
                self.prompt 
                | streaming_llm 
                | StrOutputParser()
            )
            
            # Run the chain
            response = streaming_chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
            
            # Use the accumulated text from the stream handler
            response = stream_handler.text
        else:
            # No streaming, use the normal chain
            chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            response = chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
        
        # Post-process mathematical content
        # response = self.post_process_mathematical_content(response)
        
        # Add formatted source references
        if not response.strip().endswith(("?", "...")):
            source_references = self.format_source_references(context)
            if source_references:
                response += f"\n\n{source_references}"
        
        return response

    def stream_response(
        self,
        question: str,
        context: List[Dict],
        chat_history: Optional[List[Dict]] = None
    ):
        """Yield tokens as they are generated for FastAPI StreamingResponse."""
        from langchain.callbacks.base import BaseCallbackHandler
        import queue
        import threading

        class FastAPIStreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.queue = queue.Queue()
                self.done = False

            def on_llm_new_token(self, token: str, **kwargs):
                self.queue.put(token)

            def on_llm_end(self, *args, **kwargs):
                self.done = True

        handler = FastAPIStreamHandler()
        streaming_llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,
            streaming=True,
            callbacks=[handler]
        )
        formatted_context = "\n\n".join([
            f"CONTEXT {i+1}:\n{doc['text']}\n" 
            for i, doc in enumerate(context)
        ])
        formatted_history = format_chat_history(chat_history) if chat_history else ""
        chain = (
            self.prompt 
            | streaming_llm 
            | StrOutputParser()
        )
        def run_chain():
            chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": question
            })
            handler.done = True
        thread = threading.Thread(target=run_chain)
        thread.start()
        buffer = ""
        while not handler.done or not handler.queue.empty():
            try:
                token = handler.queue.get(timeout=0.1)
                buffer += token
                print(f"[STREAM CHUNK] {token}")  # Print each streaming chunk
                yield token
            except queue.Empty:
                continue
        # Optionally, add source references at the end
        if not buffer.strip().endswith(("?", "...")):
            source_references = self.format_source_references(context)
            if source_references:
                yield f"\n\n{source_references}"
