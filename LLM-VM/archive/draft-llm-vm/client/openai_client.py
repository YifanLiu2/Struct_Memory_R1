import yaml
from pathlib import Path
from openai import OpenAI


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_client():
    config = load_config()
    return OpenAI(api_key=config["openai"]["api_key"])


def get_model():
    config = load_config()
    return config["openai"]["model"]


def chat(messages, **kwargs):

    client = get_client()
    model = get_model()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return response


def make_request(prompt, system_prompt=None):

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = chat(messages)
    return response.choices[0].message.content