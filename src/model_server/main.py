import asyncio
from typing import Any, Dict

from common.config import load_settings
from common.jsonl_rpc import start_server
from model_server.embed_model import embed_text
from model_server.image_embed_model import embed_image
from model_server.router_model import route
from model_server.text_model import generate
from model_server.video_model import analyze as analyze_video
from model_server.vision_model import analyze


class ModelService:
    async def Route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        trace_id = params.get("trace_id")
        turn_id = params.get("turn_id")
        result = route(text, trace_id=trace_id, turn_id=turn_id)
        return {"status": "OK", "result": result}

    async def Generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        ctx = params.get("context", {})
        model_override = params.get("model_override")
        options_override = params.get("options_override")
        output = generate(
            text,
            ctx,
            model_override=model_override,
            options_override=options_override,
        )
        return {"status": "OK", "text": output}

    async def EmbedText(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        vec, model = embed_text(text)
        return {"status": "OK", "embedding": vec, "model": model}

    async def EmbedImage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_ref = params.get("image_ref", "")
        image_b64 = params.get("image_b64", "")
        vec, model = embed_image(image_ref=image_ref, image_b64=image_b64)
        return {"status": "OK", "embedding": vec, "model": model}

    async def Vision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_ref = params.get("image_ref", "")
        question = params.get("question")
        if question is not None and not isinstance(question, str):
            question = str(question)
        result = analyze(image_ref, question=question)
        return {"status": "OK", "result": result}

    async def Video(self, params: Dict[str, Any]) -> Dict[str, Any]:
        video_ref = params.get("video_ref", "")
        question = params.get("question")
        start_s = params.get("start_s")
        end_s = params.get("end_s")
        if question is not None and not isinstance(question, str):
            question = str(question)
        result = analyze_video(video_ref, question=question, start_s=start_s, end_s=end_s)
        return {"status": "OK", "result": result}

    async def Ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "OK"}


async def serve() -> None:
    settings = load_settings()
    service = ModelService()

    handlers = {
        "model.Route": service.Route,
        "model.Generate": service.Generate,
        "model.EmbedText": service.EmbedText,
        "model.EmbedImage": service.EmbedImage,
        "model.Vision": service.Vision,
        "model.Video": service.Video,
        "model.Ping": service.Ping,
    }

    listen_host = settings.rpc["orch_host"]
    listen_port = settings.rpc["model_port"]
    server = await start_server(listen_host, listen_port, handlers)
    print(f"[model] listening on {listen_host}:{listen_port}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve())
