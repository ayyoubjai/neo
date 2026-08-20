import os


class SandboxViolation(Exception):
    pass


def resolve_workspace_path(workspace_root: str, workspace_path: str) -> str:
    if not workspace_path.startswith("workspace:/"):
        import re
        clean_path = re.sub(r"^([a-zA-Z]:)?[\\/]*", "", workspace_path)
        workspace_path = f"workspace:/{clean_path}"
    rel = workspace_path[len("workspace:/"):].lstrip("/")
    abs_path = os.path.abspath(os.path.join(workspace_root, rel))
    root = os.path.abspath(workspace_root)
    if not abs_path.startswith(root + os.sep) and abs_path != root:
        raise SandboxViolation(f"Path escapes workspace sandbox. Requested path '{workspace_path}' resolves outside workspace root. Use relative paths within the workspace or absolute paths under '{root}'.")
    return abs_path
