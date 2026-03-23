#!/usr/bin/env python3
"""
Initial Request Processing - Handles reading, processing, and saving travel planning requests
"""

import yaml
from pathlib import Path
from open_ai_agent import OpenAIAgent


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_request_file(agent, request_file):
    """
    Process a single request file and return the response
    
    Args:
        agent: OpenAIAgent instance
        request_file: Path to the request file
        
    Returns:
        The response from OpenAI, or None if processing fails
    """
    print(f"Processing: {request_file.name}")
    
    # Read the prompt from the file
    with open(request_file, 'r') as f:
        prompt = f.read().strip()
    
    # Skip empty files
    if not prompt:
        print(f"  Skipping empty file: {request_file.name}")
        return None
    
    # Send message to OpenAI
    result = agent.send_message(prompt)
    
    if result:
        print(f"  ✓ Completed: {request_file.name}")
    else:
        print(f"  ✗ Failed: {request_file.name}")
    
    return result


def main():
    """Main function to process all request files"""
    # Load configuration
    config = load_config()
    api_key = config['openai_api_key']
    model = config['model']
    
    # Initialize OpenAI agent
    agent = OpenAIAgent(api_key=api_key, model=model)
    
    # Setup directories
    request_dir = Path('initial_request')
    result_dir = Path('initial_itinarary')
    result_dir.mkdir(exist_ok=True)
    
    # Get all files in the request directory
    request_files = sorted([f for f in request_dir.iterdir() if f.is_file()])
    
    if not request_files:
        print("No request files found in initial_request directory")
        return
    
    print(f"Found {len(request_files)} request file(s) to process")
    print(f"Using model: {model}\n")
    
    # Process each request file
    for request_file in request_files:
        result = process_request_file(agent, request_file)
        
        if result:
            # Save result with same filename
            output_file = result_dir / f"{request_file.name}_result.txt"
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"  Saved to: {output_file}\n")
    
    print("All requests processed!")


if __name__ == "__main__":
    main()


