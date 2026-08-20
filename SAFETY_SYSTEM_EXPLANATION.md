SAFETY SYSTEM EXPLANATION
================================================================================

YOUR SETTING:
  create_tool_code_safety_enabled: true  ✓ (ENABLED)

HOW IT WORKS:
================================================================================

When you enable safety (which you have), the system:

1. Generates tool code using LLM
2. Parses the code into an Abstract Syntax Tree (AST)
3. Walks through every line looking for dangerous patterns
4. Rejects code with blocked operations
5. Tells LLM to try again (up to 4 times)

BLOCKED PERMANENTLY (ALWAYS):
================================================================================

DANGEROUS FUNCTIONS:
  eval()        ← Execute arbitrary code
  exec()        ← Execute Python code dynamically  
  compile()     ← Create code objects
  __import__()  ← Dynamic imports

DANGEROUS FUNCTION CALLS:
  os.system()   ← Shell command execution
  os.popen()    ← Pipe to command
  os.spawn*()   ← Process spawning
  shutil.rmtree() ← Recursive deletion
  pty.*()       ← Pseudo-terminal access
  subprocess.*  ← BLOCKED unless embodiment capability

BLOCKED NETWORK MODULES (DEFAULT):
  socket       ← Raw sockets
  http         ← HTTP protocol
  urllib       ← URL requests
  ftplib       ← FTP protocol
  telnetlib    ← Telnet protocol
  requests     ← HTTP library
  paramiko     ← SSH library


CONFIGURABLE (IN YOUR SETTINGS.JSON):
================================================================================

YOUR CURRENT VALUES:
  "create_tool_blocked_modules": []
    → Empty = use default network blocklist above
    → Or specify custom list to override defaults
    → Example: ["requests", "paramiko", "mysql"]

  "create_tool_extra_blocked_modules": []
    → Add EXTRA modules to the blocklist
    → Example: ["redis", "pymongo"]

  "create_tool_allowed_imports": []
    → Whitelist: Remove items from blocklist
    → Example: ["requests", "socket"]
    → Will allow these even if they're blocked


EXCEPTIONS (EMBODIMENT TOOLS):
================================================================================

Tools with embodiment capability can bypass subprocess restrictions:

HOW TO ENABLE:
  - Use tool_id starting with "embodiment.generated."
  - OR add "embodiment" to capabilities list

ALLOWED FOR EMBODIMENT:
  ✓ subprocess.Popen()
  ✓ subprocess.run()
  ✓ subprocess.call()
  ✓ os system calls
  ✗ BUT NOT shell=True (still blocked)

EXAMPLE - YOUR FAILURES EXPLAINED:
================================================================================

FAILURE 1: proc.script_exec
  Blueprint Created: YES ✓
  Code Generation Failed: YES ✗
  Reason: Generated code used subprocess.Popen()
  Why Blocked: "proc.*" tools don't have embodiment capability

FAILURE 2: proc.exec_python
  Blueprint Created: YES ✓
  Code Generation Failed: YES ✗
  Reason: Generated code used exec()
  Why Blocked: exec() is always blocked (not embodiment)

FIX BOTH:
  Change: namespace_prefix = "proc"
  To:     namespace_prefix = "embodiment"
  Result: subprocess and os calls will be allowed


VALIDATION PROCESS (IN CODE):
================================================================================

Location: src/orchestrator/main.py, lines 7040-7125

Step 1: Check if safety enabled
  if create_tool_code_safety_enabled is False:
    return None  → No validation, all code accepted

Step 2: Build blocklist
  default_blocked = {socket, http, urllib, ftplib, telnetlib, requests, paramiko}
  + configured_blocked_modules (if set)
  + extra_blocked_modules (if set)
  - allowed_imports (if set)
  + subprocess (if not embodiment)

Step 3: Scan generated code
  Walk AST looking for:
    - Import statements → check module names
    - Function calls → check function names
    - Call prefixes → check if starts with blocked pattern

Step 4: Return error or None
  If dangerous found: return "Blocked <type> in generated code: <name>"
  If all clean: return None (validation passes)


WHY THIS MATTERS:
================================================================================

WITHOUT SAFETY (if disabled):
  ✗ LLM could generate code that:
    - Steals data
    - Deletes files
    - Exfiltrates network traffic
    - Executes arbitrary commands
    - Installs backdoors

WITH SAFETY (your setting):
  ✓ Dangerous code is rejected before deployment
  ✓ LLM gets feedback and tries to fix it
  ✓ Only safe patterns make it through


TURNING OFF SAFETY:
================================================================================

To disable (NOT RECOMMENDED):
  "create_tool_code_safety_enabled": false

This means:
  • All code accepted without validation
  • No checks on imports
  • No checks on function calls
  • Fast but dangerous
  • Use only if you fully understand the risks


YOUR OPTIONS:
================================================================================

OPTION 1: Keep As-Is (RECOMMENDED)
  - Safety enabled
  - Default network blocklist
  - Use embodiment prefix for execution tools
  - Your current setup is good

OPTION 2: Allow Specific Modules
  "create_tool_allowed_imports": ["subprocess", "socket"]
  - Removes these from blocklist
  - Good for specific use cases

OPTION 3: Disable Safety (NOT RECOMMENDED)
  "create_tool_code_safety_enabled": false
  - All code accepted
  - Only if you know what you're doing

OPTION 4: Custom Blocklist
  "create_tool_blocked_modules": ["requests", "paramiko", "custom_module"]
  - Replaces default with your custom list
  - Careful: need to think through all risks


SUMMARY:
================================================================================

Your safety system:
  ✓ ENABLED and ACTIVE
  ✓ Uses default blocklist (network modules)
  ✓ Blocks dangerous operations (exec, os.system, etc)
  ✓ Allows embodiment tools to bypass subprocess restrictions
  ✓ Validates code before deployment

Why your tool generation failed:
  ✗ Requested execution-based tools (need subprocess/exec)
  ✗ But not marked as embodiment
  ✗ Safety validation rejected dangerous code

How to fix:
  ✓ Use embodiment prefix or capability
  ✓ Or disable safety (not recommended)
  ✓ Or add to allowed_imports (careful)
