from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

# Simple in-memory storage for MVP (use Qdrant in production)
documents_store: List[Dict[str, Any]] = []


class SearchRequest(BaseModel):
    query: str
    pool_ids: List[str] = []
    top_k: int = 5


class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    documents: List[Document]


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents in knowledge pools
    For MVP, returns empty results. In production, use Qdrant vector search.
    """
    # TODO: Implement actual RAG search with embeddings + Qdrant
    return SearchResponse(documents=[])


@router.post("/upload")
async def upload_document(text: str, pool_id: str = "default"):
    """
    Upload document to knowledge pool
    For MVP, stores in memory. In production, embed and store in Qdrant.
    """
    documents_store.append(
        {"content": text, "pool_id": pool_id, "metadata": {"source": "upload"}}
    )
    return {"status": "uploaded", "pool_id": pool_id}
