import yaml
from pathlib import Path
from openai import OpenAI


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class OpenAIClient:
    """OpenAI API client wrapper"""
    
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        
        self.config = config["openai"]
        self.client = OpenAI(api_key=self.config["api_key"])
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 4096)
    
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request"""
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content
    
    def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Simple completion with optional system prompt"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)


# Convenience function to get a client instance
def get_client() -> OpenAIClient:
    """Get an OpenAI client instance"""
    return OpenAIClient()


if __name__ == "__main__":
    # Quick test
    client = get_client()
    response = client.complete("Say hello in one word.")
    print(f"Response: {response}")

