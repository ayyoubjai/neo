from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "adaptive_agent" / "genes.json"


@dataclass
class Gene:
    gene_id: str
    gene_type: str
    priority: int
    purposes: List[str]
    requirements: Dict[str, Any]
    payload: Dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Gene":
        return cls(
            gene_id=str(raw["id"]),
            gene_type=str(raw.get("type", "tool")),
            priority=int(raw.get("priority", 0)),
            purposes=list(raw.get("purposes", [])),
            requirements=dict(raw.get("requires", {})),
            payload=dict(raw.get("payload", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.gene_id,
            "type": self.gene_type,
            "priority": self.priority,
            "purposes": list(self.purposes),
            "requires": dict(self.requirements),
            "payload": dict(self.payload),
        }


def load_gene_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> List[Gene]:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    genes = raw.get("genes", raw if isinstance(raw, list) else [])
    if not isinstance(genes, list):
        raise ValueError("Gene catalog must contain a list under 'genes'")
    return [Gene.from_dict(item) for item in genes]


def group_by_type(genes: Iterable[Gene]) -> Dict[str, List[Gene]]:
    grouped: Dict[str, List[Gene]] = {}
    for gene in genes:
        grouped.setdefault(gene.gene_type, []).append(gene)
    return grouped

