# core/llm_custom.py

import requests
import json
from typing import List, Dict, Optional
from utils.config import config

class CustomLLMManager:
    def __init__(self):
        self.llm_url = config.CUSTOM_LLM_URL
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
            {context}
            
            Previous conversation history:
            {chat_history}
            
            Please provide a helpful and accurate response based on the above information.
            
            Provide answers in detailed steps like if its a process question or a 'How to' types question eg "How to add a new location"
            Step 1 : Instructions of Step 1
            Step 2: Instructions of Step 2
        """

    def _format_input(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None) -> str:
        """Formats the input for the LLM by combining the prompt, context, and history."""
        formatted_context = "\n\n".join([doc.get('text', '') for doc in context])
        
        if chat_history:
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            formatted_history = "No previous conversation."

        prompt = self.system_prompt.format(context=formatted_context, chat_history=formatted_history)
        return f"{prompt}\n\nHuman: {question}\nAssistant:"

    def stream_response(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None):
        """Yields tokens as they are generated for a streaming response."""
        formatted_question = self._format_input(question, context, chat_history)
        
        payload = {
            "question": formatted_question,
            "max_tokens": 2048  # You can adjust this as needed
        }
        
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            with requests.post(self.llm_url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()  # Raise an exception for bad status codes
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
        except requests.exceptions.RequestException as e:
            print(f"Error streaming from custom LLM: {e}")
            yield "Error: Could not get a response from the language model."

    def generate_response(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None) -> str:
        """Generates a complete response by consuming the stream."""
        stream = self.stream_response(question, context, chat_history)
        return "".join(stream)

    def needs_clarification(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None):
        """Dummy implementation for compatibility."""
        return False, []
