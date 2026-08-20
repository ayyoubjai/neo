from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from adaptive_agent.device_probe import DeviceProfile
from adaptive_agent.gene_catalog import Gene, group_by_type


@dataclass
class PurposeProfile:
    purpose: str
    personality: str
    offline_first: bool = True
    allow_network: bool = False
    allow_background: bool = False
    cognition_mode: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationPlan:
    device: Dict[str, Any]
    purpose: Dict[str, Any]
    cognition: Dict[str, Any] | None
    model: Dict[str, Any] | None
    tools: List[Dict[str, Any]]
    prompts: List[Dict[str, Any]]
    services: List[Dict[str, Any]]
    disabled: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _matches_any(value: str, allowed: Iterable[str]) -> bool:
    allowed_set = {str(item) for item in allowed}
    return not allowed_set or value in allowed_set or "*" in allowed_set


def _requirement_failure(gene: Gene, device: DeviceProfile, purpose: PurposeProfile) -> str | None:
    req = gene.requirements
    if not _matches_any(device.device_class, req.get("device_classes", ["*"])):
        return f"device_class {device.device_class} not allowed"
    if req.get("android") is True and not device.is_android:
        return "android required"
    if req.get("termux") is True and not device.is_termux:
        return "termux required"
    if req.get("termux") is False and device.is_termux:
        return "termux disallowed"
    min_ram = int(req.get("min_ram_mb", 0) or 0)
    if min_ram and device.ram_mb and device.ram_mb < min_ram:
        return f"ram {device.ram_mb}MB below {min_ram}MB"
    min_storage = int(req.get("min_storage_free_mb", 0) or 0)
    if min_storage and device.storage_free_mb and device.storage_free_mb < min_storage:
        return f"free storage {device.storage_free_mb}MB below {min_storage}MB"
    if req.get("network") is True and not purpose.allow_network:
        return "network not allowed by purpose profile"
    if req.get("background") is True and not purpose.allow_background:
        return "background execution not allowed by purpose profile"
    return None


def _purpose_score(gene: Gene, purpose: PurposeProfile) -> int:
    purposes = set(gene.purposes)
    if "*" in purposes:
        return 1
    if purpose.purpose in purposes:
        return 20
    return 0


def resolve_genes(
    genes: Iterable[Gene],
    device: DeviceProfile,
    purpose: PurposeProfile,
) -> ActivationPlan:
    enabled: List[Gene] = []
    disabled: List[Dict[str, Any]] = []
    for gene in genes:
        reason = _requirement_failure(gene, device, purpose)
        if reason:
            disabled.append({"id": gene.gene_id, "type": gene.gene_type, "reason": reason})
            continue
        if _purpose_score(gene, purpose) <= 0:
            disabled.append({"id": gene.gene_id, "type": gene.gene_type, "reason": "purpose mismatch"})
            continue
        enabled.append(gene)

    enabled.sort(key=lambda item: (_purpose_score(item, purpose), item.priority), reverse=True)
    grouped = group_by_type(enabled)
    models = grouped.get("model", [])
    selected_model = models[0].to_dict() if models else None
    cognition_genes = grouped.get("cognition", [])
    selected_cognition = _select_cognition_gene(cognition_genes, purpose)

    return ActivationPlan(
        device=device.to_dict(),
        purpose=purpose.to_dict(),
        cognition=selected_cognition.to_dict() if selected_cognition else None,
        model=selected_model,
        tools=[item.to_dict() for item in grouped.get("tool", [])],
        prompts=[item.to_dict() for item in grouped.get("prompt", [])],
        services=[item.to_dict() for item in grouped.get("service", [])],
        disabled=disabled,
    )


def _select_cognition_gene(genes: List[Gene], purpose: PurposeProfile) -> Gene | None:
    if not genes:
        return None
    requested = (purpose.cognition_mode or "auto").strip().lower()
    if requested and requested != "auto":
        for gene in genes:
            payload = gene.payload or {}
            aliases = {str(item).lower() for item in payload.get("aliases", [])}
            mode = str(payload.get("mode", "")).lower()
            if requested == mode or requested in aliases:
                return gene
    return genes[0]
