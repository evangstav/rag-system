from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import os

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": request.message}],
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
