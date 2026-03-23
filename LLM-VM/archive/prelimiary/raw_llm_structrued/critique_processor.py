#!/usr/bin/env python3
"""
Critique Processor - Applies critiques to initial itineraries
"""

import argparse
import yaml
from pathlib import Path
from open_ai_agent import OpenAIAgent


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_critique_prompt(itinerary, critique):
    """
    Build a prompt that applies critique to itinerary
    
    Args:
        itinerary: The original itinerary JSON
        critique: The critique/feedback to apply
        
    Returns:
        The formatted prompt string
    """
    prompt = f"""You are a travel-planning assistant. Below is an existing travel itinerary and user feedback/critiques.

Your task:
1. Deal with the critique
2. Output the updated itinerary in the same JSON format

## Original Itinerary:
{itinerary}

## User Critique/Feedback:
{critique}

## Instructions:
- Maintain the same JSON structure
- Output only valid JSON
"""
    return prompt


def get_next_run_number(directory, prefix):
    """
    Find the next available run number for output files
    
    Args:
        directory: Path to the output directory
        prefix: File prefix to search for
        
    Returns:
        Next available run number (1, 2, 3, ...)
    """
    if not directory.exists():
        return 1
    
    existing_files = list(directory.glob(f'{prefix}_*.txt'))
    if not existing_files:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for f in existing_files:
        try:
            # Get the number after the last underscore
            num = int(f.stem.split('_')[-1])
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1


def process_critique(days):
    """
    Process critique for a specific number of days itinerary
    
    Args:
        days: Number of days (1, 3, or 10)
    """
    # Load configuration
    config = load_config()
    agent = OpenAIAgent(api_key=config['openai_api_key'], model=config['model'])
    
    # Setup paths
    itinerary_path = Path(f'initial_itinarary/initial_request_{days}_day_result.txt')
    critique_path = Path(f'critique/critique_{days}_day')
    prompt_dir = Path(f'prompts/{days}_day')
    output_dir = Path(f'after_critique/{days}_day')
    
    # Create directories if needed
    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read itinerary
    if not itinerary_path.exists():
        print(f"Error: Itinerary not found at {itinerary_path}")
        return
    
    with open(itinerary_path, 'r') as f:
        itinerary = f.read().strip()
    
    # Read critique
    if not critique_path.exists():
        print(f"Error: Critique not found at {critique_path}")
        return
    
    with open(critique_path, 'r') as f:
        critique = f.read().strip()
    
    if not critique:
        print(f"Warning: Critique file is empty for {days}-day itinerary")
        return
    
    print(f"Processing {days}-day itinerary critique...")
    print(f"Critique: {critique}\n")
    
    # Build prompt
    prompt = build_critique_prompt(itinerary, critique)
    
    # Get next run number
    run_number = get_next_run_number(output_dir, 'after_critique')
    
    # Save prompt
    prompt_file = prompt_dir / f'critique_prompt_{run_number}.txt'
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Saved prompt to: {prompt_file}")
    
    # Send to OpenAI
    result = agent.send_message(prompt)
    
    if result:
        # Save result
        output_file = output_dir / f'after_critique_{run_number}.txt'
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"✓ Saved result to: {output_file}")
    else:
        print(f"✗ Failed to process {days}-day critique")


def main():
    parser = argparse.ArgumentParser(description='Apply critiques to travel itineraries')
    parser.add_argument('--days', type=int, required=True, choices=[1, 3, 10],
                        help='Number of days (1, 3, or 10)')
    parser.add_argument('--all', action='store_true',
                        help='Process all available critiques')
    
    args = parser.parse_args()
    
    if args.all:
        for days in [1, 3, 10]:
            process_critique(days)
            print()
    else:
        process_critique(args.days)


if __name__ == "__main__":
    main()

