from typing import Any, Dict

from common.system_entity import load_system_entity
from common.time_utils import utc_now_iso


def build_system_memory() -> Dict[str, Any]:
    entity = load_system_entity()
    return {
        "mem_id": entity.mem_id,
        "type": "entity",
        "name": entity.name,
        "data": {
            "object_type": "system_config_snapshot",
            "name": entity.name,
            "aliases": entity.aliases,
            "properties": entity.properties,
            "pins": entity.pins,
        },
        "created_at": utc_now_iso(),
        "use_count": 0,
    }
