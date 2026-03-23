#!/usr/bin/env python3
"""
OpenAI Agent - Simple module to send messages to OpenAI API
"""

from openai import OpenAI


class OpenAIAgent:
    """Agent that handles communication with OpenAI API"""
    
    def __init__(self, api_key, model="gpt-4o"):
        """
        Initialize the OpenAI agent
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def send_message(self, prompt, temperature=1):
        """
        Send a message to OpenAI and get response
        
        Args:
            prompt: The prompt/message to send
            temperature: Temperature for response generation (default: 0.7)
            max_tokens: Maximum tokens in response (default: 4096)
            
        Returns:
            The response content as a string, or None if error occurs
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"  ✗ Error sending message to OpenAI: {e}")
            return None


