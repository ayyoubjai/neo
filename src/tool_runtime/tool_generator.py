import os
import sys
from pathlib import Path
from typing import List

from model_server.llamacpp_client import generate, LlamacppError

TOOL_GENERATION_PROMPT = """You are an AI Tool Architect.
Your task is to write a Python function that serves as a tool for an autonomous agent.
The tool must be named according to its purpose, accept specific arguments, and return a tuple of (result_dict, error_dict).
Do not provide explanations. ONLY output raw, valid Python code wrapped in ```python ... ``` markdown.

Requirements for the tool:
- Must have a concise, descriptive name (e.g. `read_csv_file`).
- Must take `(args: Dict[str, Any], workspace_root: str)` as parameters.
- Must return `Tuple[Dict[str, Any], Dict[str, Any]]`.
- If successful, return `({"result": ...}, {})`.
- If an error occurs, return `({}, {"error": "description"})`.

Generate the tool for the following requirement:
{requirement}
"""

def generate_tools(tools_needed: List[str], auto_inject: bool = False) -> None:
    print("\n======================================================")
    print(" Starting Autonomic Tool Generation Stage")
    print(f" Target Tools: {len(tools_needed)}")
    print(f" Auto-Inject: {auto_inject}")
    print("======================================================")
    
    # Determine where to save the generated tools
    project_root = Path(__file__).resolve().parents[2]
    
    if auto_inject:
        target_dir = project_root / "src" / "dynamic_tools"
        print("[!] Warning: Auto-injecting tools directly into the execution path.")
    else:
        target_dir = project_root / "workspace" / "tools" / "pending"
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, requirement in enumerate(tools_needed):
        print(f"\n[{idx+1}/{len(tools_needed)}] Generating tool for: '{requirement}'...")
        prompt = TOOL_GENERATION_PROMPT.replace("{requirement}", requirement)
        
        try:
            response = generate(
                prompt=prompt,
                model="Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                options={"temperature": 0.2, "max_tokens": 1024}
            )
        except LlamacppError as e:
            print(f"[-] Failed to generate tool: {e}")
            continue
            
        if "```python" in response:
            try:
                code = response.split("```python")[1].split("```")[0].strip()
                # Create a safe filename from the requirement
                safe_name = "".join([c if c.isalnum() else "_" for c in requirement.lower()])[:20]
                file_path = target_dir / f"tool_{safe_name}.py"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code + "\n")
                    
                print(f"[+] Successfully generated tool and saved to: {file_path}")
                if not auto_inject:
                    print(f"    (Review the tool and manually move to active tool registry to use it)")
            except IndexError:
                print("[-] Failed to parse Python code from the response.")
        else:
            print("[-] Model did not output standard python formatting. Skipping.")
            
    print("\n[+] Tool Generation Stage Complete.")
