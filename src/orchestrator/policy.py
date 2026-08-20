from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PolicyDecision:
    status: str
    reason: str
    tier: int


TIER_MAP = {
    "tier0": 0,
    "tier1": 1,
    "tier2": 2,
    "tier3": 3,
}


def _highest_tier(perms: List[str]) -> int:
    tier = 0
    for perm in perms:
        tier = max(tier, TIER_MAP.get(perm, 0))
    return tier


class PolicyEngine:
    def evaluate_tool(self, tool_def: Dict[str, Any], args: Dict[str, Any]) -> PolicyDecision:
        perms = tool_def.get("required_permissions", [])
        tier = _highest_tier(perms)
        if tier == 0:
            return PolicyDecision(status="ALLOW", reason="Tier0 tool", tier=tier)
        if tier == 1:
            return PolicyDecision(status="CONFIRM", reason="Tier1 requires confirmation", tier=tier)
        if tier == 2:
            return PolicyDecision(status="CONFIRM", reason="Tier2 requires confirmation each time", tier=tier)
        return PolicyDecision(status="PASSPHRASE", reason="Tier3 requires passphrase", tier=tier)
