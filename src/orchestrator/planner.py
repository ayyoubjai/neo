import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Plan:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)


class Planner:
    def plan(self, text: str) -> Plan:
        lower = text.lower()
        plan = Plan()

        if "time" in lower:
            plan.steps.append({"tool_id": "sys.time", "args": {}})
            plan.verification.append("Confirm time format returned")

        calc_match = re.search(r"(?:calc:|calculate|compute)\s+(.+)", text, re.IGNORECASE)
        if calc_match:
            expr = calc_match.group(1).strip()
            plan.steps.append({"tool_id": "math.eval", "args": {"expr": expr}})
            plan.verification.append("Confirm numeric result returned")

        read_match = re.search(r"(?:read file|read|open)\s+([^\s]+)", text, re.IGNORECASE)
        if read_match:
            path = read_match.group(1)
            plan.steps.append({"tool_id": "fs.read_file", "args": {"path": path}})
            plan.verification.append("Confirm file content returned")

        write_match = re.search(r"(?:write|save)\s+([^\s]+)\s+::\s+(.+)", text, re.IGNORECASE)
        if write_match:
            path = write_match.group(1)
            content = write_match.group(2)
            plan.steps.append({"tool_id": "fs.write_file", "args": {"path": path, "content": content}})
            plan.verification.append("Confirm bytes written and audit log")

        return plan
