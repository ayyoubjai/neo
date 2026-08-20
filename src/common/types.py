from typing import Any, Dict, Optional


EVENT_WAKE = "WakeEvent"
EVENT_STT_PARTIAL = "STTPartial"
EVENT_STT_FINAL = "STTFinal"
EVENT_TURN_PATCH = "TurnPatch"
EVENT_TTS_START = "TTSStart"
EVENT_TTS_CANCEL = "TTSCancel"
EVENT_ASSISTANT_DRAFT = "AssistantDraft"
EVENT_ASSISTANT_FINAL = "AssistantFinal"
EVENT_PERMISSION_REQUEST = "PermissionRequest"
EVENT_PERMISSION_RESPONSE = "PermissionResponse"


def make_event(event_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None) -> Dict[str, Any]:
    event = {"event_type": event_type, "payload": payload}
    if trace_id:
        event["trace_id"] = trace_id
    return event
