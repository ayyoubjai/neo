from typing import Any, Dict, List


def compose_with_tools(user_text: str, tool_results: List[Dict[str, Any]], mode: str) -> str:
    lines = [f"Mode: {mode}."]
    lines.append(f"User: {user_text}")
    if tool_results:
        lines.append("Tool results:")
        for result in tool_results:
            tool_id = result.get("tool_id")
            output = result.get("result")
            lines.append(f"- {tool_id}: {output}")
    else:
        lines.append("No tool calls were executed.")
    return "\n".join(lines)
