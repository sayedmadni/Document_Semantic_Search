#!/usr/bin/env python3
"""
shashi_tharoor.py - Persona impersonation script

This script allows you to impersonate a specific person (like Shashi Tharoor) 
by providing a preface/prompt that defines their persona, and then generates
responses in that person's style.
"""

from ollama import Client
from typing import Optional


# ============================================================
# Configuration
# ============================================================

# Ollama client configuration
OLLAMA_HOST = "localhost:11434"
MODEL_NAME = "llama3.1"  # Change to your preferred model

# Default persona preface (can be overridden)
DEFAULT_PERSONA = """
You are Shashi Tharoor, the renowned Indian politician, diplomat, and author known for:
- His exceptional command of the English language and eloquent speaking style
- Deep knowledge of Indian history, politics, and international relations
- Wit, humor, and intellectual prowess
- Ability to articulate complex ideas with clarity and elegance

Respond in Shashi Tharoor's distinctive style: eloquent, articulate, well-researched, 
and sprinkled with historical references and literary flourishes. The typical response must be exaggerated to accentuate Shashi Tharoor's fondeness for multi syllable words in lieu of brevity.
"""


# ============================================================
# Core Functions
# ============================================================

def impersonate_person(prompt: str, persona_preface: Optional[str] = None, model: str = MODEL_NAME) -> str:
    """
    Generate a response impersonating a specific person.
    
    Parameters
    ----------
    prompt : str
        The question or prompt to respond to
    persona_preface : str, optional
        The preface/persona description. If None, uses DEFAULT_PERSONA
    model : str
        The Ollama model to use (default: llama3.1)
    
    Returns
    -------
    str
        The generated response in the persona's style
    """
    client = Client(host=OLLAMA_HOST)
    
    # Use provided persona or default
    preface = persona_preface or DEFAULT_PERSONA
    
    # Construct the full prompt
    full_prompt = f"{preface}\n\nUser: {prompt}\n\nResponse:"
    
    try:
        response = client.generate(model=model, prompt=full_prompt)
        return response['response']
    except Exception as e:
        return f"Error generating response: {e}"


def interactive_mode(persona_preface: Optional[str] = None):
    """
    Run an interactive session where you can chat with the impersonated persona.
    
    Parameters
    ----------
    persona_preface : str, optional
        The persona description to use
    """
    print("="*60)
    print("PERSONA IMPERSONATION MODE")
    print("="*60)
    print("\nEnter your questions below. Type 'exit' or 'quit' to end.")
    print("Type 'newpersona' to change the persona.")
    print("-"*60)
    
    current_preface = persona_preface
    
    while True:
        user_prompt = input("\nYou: ").strip()
        
        if not user_prompt:
            continue
        
        if user_prompt.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_prompt.lower() == 'newpersona':
            print("\nEnter new persona preface (or press Enter to cancel):")
            new_persona = input().strip()
            if new_persona:
                current_preface = new_persona
                print("✓ Persona updated!")
            continue
        
        print("\nGenerating response...")
        response = impersonate_person(user_prompt, current_preface)
        print(f"\nResponse:\n{response}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Main function with command-line arguments support.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Impersonate a person and generate responses in their style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default persona
  python shashi_tharoor.py

  # Interactive mode with custom persona
  python shashi_tharoor.py --persona "You are Albert Einstein..."

  # Single query mode
  python shashi_tharoor.py --query "What is your view on climate change?"

  # Custom persona from file
  python shashi_tharoor.py --persona-file my_persona.txt --query "Hello"
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single question to ask (non-interactive mode)'
    )
    
    parser.add_argument(
        '--persona', '-p',
        type=str,
        help='Persona preface/prompt to use'
    )
    
    parser.add_argument(
        '--persona-file',
        type=str,
        help='Path to file containing persona preface'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=MODEL_NAME,
        help=f'Ollama model to use (default: {MODEL_NAME})'
    )
    
    args = parser.parse_args()
    
    # Load persona from file if provided
    persona_preface = args.persona
    if args.persona_file:
        try:
            with open(args.persona_file, 'r') as f:
                persona_preface = f.read()
        except Exception as e:
            print(f"Error reading persona file: {e}")
            return
    
    # Single query mode
    if args.query:
        response = impersonate_person(args.query, persona_preface, args.model)
        print("\n" + "="*60)
        print("RESPONSE")
        print("="*60)
        print(response)
        return
    
    # Interactive mode
    interactive_mode(persona_preface)


if __name__ == "__main__":
    main()

