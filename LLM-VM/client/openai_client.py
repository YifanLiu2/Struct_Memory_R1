"""
OpenAI Client - Unified client for OpenAI models.

Supports:
- GPT-4 series: Uses temperature for creativity control
- GPT-5 series: Uses reasoning_effort for reasoning depth control

Configuration is loaded from config.yaml:
- model: "gpt-4o", "gpt-5", "gpt-5-mini", etc.
- temperature: 0-1 (for GPT-4 series)
- reasoning_effort: none/low/medium/high/ (for GPT-5 series)
"""

import os
import yaml
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config() -> dict:
    """Load configuration from config.yaml with env var substitution."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    key = config.get("openai", {}).get("api_key")
    
    if key and ("${OPENAI_API_KEY}" in key or "${OPENROUTER_API_KEY}" in key or "${HUGGINGFACE_API_KEY}" in key):
        if "OPENROUTER_API_KEY" in key:
            env_var = "OPENROUTER_API_KEY"
        elif "HUGGINGFACE_API_KEY" in key:
            env_var = "HUGGINGFACE_API_KEY"
        else:
            env_var = "OPENAI_API_KEY"
        
        api_key = os.getenv(env_var)
        if not api_key:
            # Try to find .env file explicitly if not found
            env_path = Path(__file__).parent.parent / ".env"
            load_dotenv(dotenv_path=env_path)
            api_key = os.getenv(env_var)
        
        if api_key:
            config["openai"]["api_key"] = api_key
        else:
            print(f"Warning: {env_var} not found in environment or .env file")
            
    return config


@dataclass
class TokenUsage:
    """Token usage statistics from an API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class CompletionResult:
    """Result from a completion call including response and token usage."""
    content: str
    usage: TokenUsage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "usage": self.usage.to_dict()
        }


class OpenAIClient:
    """
    Unified OpenAI API client.
    
    Automatically handles model-specific parameters:
    - GPT-4 series: temperature for creativity control
    - GPT-5/o1/o3 series: reasoning_effort for reasoning depth
    """
    
    # Models that use max_completion_tokens instead of max_tokens
    NEW_API_MODELS = ["gpt-5", "o1", "o3"]
    
    # Models that don't support custom temperature (reasoning models)
    NO_TEMPERATURE_MODELS = ["gpt-5", "o1", "o3"]
    
    # Models that support reasoning effort parameter
    REASONING_MODELS = ["gpt-5", "o1", "o3"]
    
    # Valid reasoning effort values
    VALID_REASONING_EFFORTS = ["none", "low", "medium", "high", "xhigh"]
    
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        
        self.config = config["openai"]
        base_url = self.config.get("base_url")  # Optional: allows OpenRouter or other gateways
        
        # Handle API key env var substitution (supports ${OPENAI_API_KEY} / ${OPENROUTER_API_KEY} placeholders)
        api_key = self.config.get("api_key", "")
        if api_key and ("${OPENAI_API_KEY}" in api_key or "${OPENROUTER_API_KEY}" in api_key or "${HUGGINGFACE_API_KEY}" in api_key):
            if "OPENROUTER_API_KEY" in api_key:
                env_var = "OPENROUTER_API_KEY"
            elif "HUGGINGFACE_API_KEY" in api_key:
                env_var = "HUGGINGFACE_API_KEY"
            else:
                env_var = "OPENAI_API_KEY"
            
            env_key = os.getenv(env_var)
            if not env_key:
                env_path = Path(__file__).parent.parent / ".env"
                load_dotenv(dotenv_path=env_path)
                env_key = os.getenv(env_var)
            if env_key:
                api_key = env_key
            else:
                print(f"Warning: {env_var} not found in environment or .env file")
        
        # Initialize OpenAI-compatible client.
        # If base_url is set (e.g., OpenRouter: https://openrouter.ai/api/v1),
        # the same OpenAI SDK is used against that endpoint. For providers
        # like OpenRouter that expect additional metadata headers, you can
        # configure them here.
        default_headers = None
        if base_url and "openrouter.ai" in base_url:
            # OpenRouter recommends sending at least one of HTTP-Referer or X-Title
            default_headers = {
                "HTTP-Referer": "https://llm-vm.local",  # arbitrary identifier for this app
                "X-Title": "LLM-VM Experiments",
            }
        
        if base_url:
            if default_headers:
                self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = self.config.get("model", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 4096)
        
        # Model-specific defaults
        if self._is_reasoning_model(self.model):
            # GPT-5/o1/o3: use reasoning_effort, ignore temperature
            self.reasoning_effort = self.config.get("reasoning_effort", "medium")
            self.temperature = None
        else:
            # GPT-4: use temperature, no reasoning_effort
            self.temperature = self.config.get("temperature", 0.7)
            self.reasoning_effort = None
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model (GPT-5/o1/o3)."""
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in self.REASONING_MODELS)
    
    def _uses_new_api(self, model: str) -> bool:
        """Check if model uses the new API with max_completion_tokens."""
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in self.NEW_API_MODELS)
    
    def _build_completion_kwargs(
        self, 
        model: str, 
        max_tokens: int,
        temperature: float = None,
        reasoning_effort: str = None
    ) -> dict:
        """Build kwargs for completion API, handling model-specific parameters."""
        kwargs = {"model": model}
        
        # Use appropriate token limit parameter based on model
        if self._uses_new_api(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        
        # Model-specific parameters
        if self._is_reasoning_model(model):
            # Reasoning models: use reasoning_effort (top-level parameter)
            if reasoning_effort and reasoning_effort in self.VALID_REASONING_EFFORTS:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            # Traditional models: use temperature
            if temperature is not None:
                kwargs["temperature"] = temperature
        
        return kwargs
    
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request."""
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        
        completion_kwargs = self._build_completion_kwargs(
            model, max_tokens, temperature, reasoning_effort
        )
        completion_kwargs["messages"] = messages
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                try:
                    response = self.client.chat.completions.create(**completion_kwargs)
                except TypeError as e:
                    if "reasoning_effort" in str(e):
                        # Library doesn't support reasoning_effort yet, retry without it
                        completion_kwargs.pop("reasoning_effort", None)
                        response = self.client.chat.completions.create(**completion_kwargs)
                    else:
                        raise
                break  # Success
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str or "too_many_requests" in err_str:
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts due to rate limit: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit or quota error hit. Retrying in {delay} seconds... ({e})")
                    time.sleep(delay)
                else:
                    raise
        
        content = response.choices[0].message.content
        return content if content is not None else ""
    
    def chat_with_usage(self, messages: list[dict], **kwargs) -> CompletionResult:
        """Send a chat completion request and return response with token usage."""
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        
        completion_kwargs = self._build_completion_kwargs(
            model, max_tokens, temperature, reasoning_effort
        )
        completion_kwargs["messages"] = messages
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                try:
                    response = self.client.chat.completions.create(**completion_kwargs)
                except TypeError as e:
                    if "reasoning_effort" in str(e):
                        # Library doesn't support reasoning_effort yet, retry without it
                        completion_kwargs.pop("reasoning_effort", None)
                        response = self.client.chat.completions.create(**completion_kwargs)
                    else:
                        raise
                break  # Success
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str or "too_many_requests" in err_str:
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts due to rate limit: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit or quota error hit. Retrying in {delay} seconds... ({e})")
                    time.sleep(delay)
                else:
                    raise
        
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # Handle None content (can happen with reasoning models)
        content = response.choices[0].message.content
        if content is None:
            content = ""
        
        return CompletionResult(content=content, usage=usage)
    
    def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Simple completion with optional system prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
    
    def complete_with_usage(self, prompt: str, system_prompt: str = None, **kwargs) -> CompletionResult:
        """Simple completion with optional system prompt, returns response with token usage."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_with_usage(messages, **kwargs)


def get_client() -> OpenAIClient:
    """Get an OpenAI client instance based on config.yaml settings."""
    return OpenAIClient()


if __name__ == "__main__":
    # Quick test
    client = get_client()
    print(f"Model: {client.model}")
    print(f"Reasoning model: {client._is_reasoning_model(client.model)}")
    if client.reasoning_effort:
        print(f"Reasoning effort: {client.reasoning_effort}")
    if client.temperature is not None:
        print(f"Temperature: {client.temperature}")
    
    response = client.complete("Say hello in one word.")
    print(f"Response: {response}")

