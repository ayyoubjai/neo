import json
import sys
from typing import Any, Dict

from model_server.llamacpp_client import generate, LlamacppError

SYSTEM_PROMPT = """You are an AI Initialization Wizard.
Your job is to interview the user to determine the optimal configuration for an AI agent.
Ask the user questions to figure out:
1. The agent's Name
2. Any Aliases the agent should respond to
3. The Purpose or Role of the agent
4. The Personality or Behavioral traits
Converse with the user naturally. Do not ask all questions at once. Ask one or two at a time.
Once you have enough information to confidently define the agent identity and behavior, output the final configuration as a JSON block wrapped in ```json ... ``` markdown.
The JSON must have this exact structure:
{
  "name": "string",
  "aliases": ["string", "string"],
  "purpose": "string",
  "personality": "string (A detailed system prompt instruction)"
}
Only output the JSON block when you are completely finished with the interview.
"""

def run_interactive_setup(model_name: str) -> Dict[str, Any]:
    print("======================================================")
    print(" Starting Interactive Initialization Wizard")
    print(f" LLM Provider: llama.cpp (Model: {model_name})")
    print("======================================================\n")
    
    chat_history = f"System: {SYSTEM_PROMPT}\n\n"
    
    print("Wizard: Hello! I'm here to help set up your new AI agent. What kind of agent would you like to create today? (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            sys.exit(0)
            
        chat_history += f"User: {user_input}\nWizard: "
        
        try:
            # We use the generic generate method to talk to the local model
            response = generate(
                prompt=chat_history,
                model=model_name,
                options={"temperature": 0.7, "max_tokens": 512}
            )
        except LlamacppError as e:
            print(f"\n[Error connecting to local LLM: {e}]")
            print("Please ensure your llama.cpp server is running.")
            sys.exit(1)
            
        print(f"\nWizard: {response}")
        chat_history += f"{response}\n"
        
        # Check if the LLM output the final JSON
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                config = json.loads(json_str)
                print("\n[+] Configuration successfully parsed!")
                return config
            except (IndexError, json.JSONDecodeError):
                print("\n[!] Wizard tried to finalize but the JSON was malformed. Continuing interview...")

if __name__ == "__main__":
    # Test stub
    run_interactive_setup("Llama-3.1-8B-Instruct-Q4_K_M.gguf")
