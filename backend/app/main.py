from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, scratchpad, rag


app = FastAPI(title="RAG Chat System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(scratchpad.router, prefix="/api/scratchpad", tags=["scratchpad"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])


@app.get("/health")
async def health():
    return {"status": "healthy"}
