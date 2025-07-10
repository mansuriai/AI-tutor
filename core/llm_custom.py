# core/llm_custom.py

import requests
import json
from typing import List, Dict, Optional
from utils.config import config

class CustomLLMManager:
    def __init__(self):
        self.llm_url = config.CUSTOM_LLM_URL
        self.model = config.CUSTOM_LLM_MODEL
        self.system_prompt_template = """You are an AI assistant designed to provide clear, detailed, and accurate answers to user queries based on the provided context.
 
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
            
            Please provide a helpful and accurate response based on the above information.
            
            Provide answers in detailed steps like if its a process question or a 'How to' types question eg "How to add a new location"
            Step 1 : Instructions of Step 1
            Step 2: Instructions of Step 2
        """

    def _prepare_messages(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None, language: str = "English") -> List[Dict]:
        """Constructs the list of messages for the chat completions API."""
        formatted_context = "\n\n".join([doc.get('text', '') for doc in context])
        system_content = self.system_prompt_template.format(context=formatted_context)
        
        messages = [{"role": "system", "content": system_content}]
        
        # Add past conversation history
        if chat_history:
            messages.extend(chat_history)
            
        # Add the current user question
        user_message_content = f"{question}\n\nALWAYS (answer in {language})"
        messages.append({"role": "user", "content": user_message_content})
        
        return messages

    def stream_response(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None, language: str = "English"):
        """Yields tokens as they are generated for a streaming response."""
        messages = self._prepare_messages(question, context, chat_history, language)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "stream": True
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }

        try:
            with requests.post(self.llm_url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[len('data: '):]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(json_str)
                                if data['choices'] and 'content' in data['choices'][0]['delta']:
                                    token = data['choices'][0]['delta']['content']
                                    if token:
                                        yield token
                            except json.JSONDecodeError:
                                print(f"Could not decode JSON: {json_str}")
                                continue
        except requests.exceptions.RequestException as e:
            print(f"Error streaming from custom LLM: {e}")
            yield "Error: Could not get a response from the language model."

    def generate_response(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None, language: str = "English") -> str:
        """Generates a complete response by consuming the stream."""
        stream = self.stream_response(question, context, chat_history, language)
        return "".join(stream)

    def needs_clarification(self, question: str, context: List[Dict], chat_history: Optional[List[Dict]] = None):
        """Dummy implementation for compatibility."""
        return False, []
