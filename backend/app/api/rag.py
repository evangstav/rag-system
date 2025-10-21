"""
RAG API endpoints.

Handles document upload, knowledge pool management, and semantic search.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import os
import tempfile

from app.dependencies import get_db
from app.models.database import KnowledgePool, Document as DBDocument, DocumentStatus
from app.models.schemas import (
    RAGSearchRequest,
    RAGSearchResponse,
    RAGDocument,
    DocumentUploadResponse,
    KnowledgePoolCreate,
    KnowledgePoolResponse,
)
from app.services.rag_service import RAGService

router = APIRouter()

# TODO: Replace with actual user authentication
DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

# Global RAG service instance (in production, use dependency injection)
rag_service = RAGService()


@router.post("/pools", response_model=KnowledgePoolResponse, status_code=201)
async def create_knowledge_pool(
    pool_data: KnowledgePoolCreate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Create a new knowledge pool.

    A knowledge pool is a collection of documents that can be searched together.
    """
    # Generate collection name from pool name
    collection_name = f"user_{str(user_id)[:8]}_{pool_data.name.lower().replace(' ', '_')}"

    # Check if pool already exists
    result = await db.execute(
        select(KnowledgePool).where(KnowledgePool.collection_name == collection_name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Knowledge pool already exists")

    # Create vector collection in Qdrant
    await rag_service.create_knowledge_pool(collection_name)

    # Create database record
    pool = KnowledgePool(
        user_id=user_id,
        name=pool_data.name,
        description=pool_data.description,
        collection_name=collection_name,
    )
    db.add(pool)
    await db.commit()
    await db.refresh(pool)

    return pool


@router.get("/pools", response_model=List[KnowledgePoolResponse])
async def list_knowledge_pools(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    List all knowledge pools for the authenticated user.
    """
    result = await db.execute(
        select(KnowledgePool).where(KnowledgePool.user_id == user_id)
    )
    pools = result.scalars().all()
    return pools


@router.delete("/pools/{pool_id}")
async def delete_knowledge_pool(
    pool_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Delete a knowledge pool and all its documents.
    """
    # Get pool
    result = await db.execute(
        select(KnowledgePool).where(
            KnowledgePool.id == pool_id,
            KnowledgePool.user_id == user_id,
        )
    )
    pool = result.scalar_one_or_none()

    if not pool:
        raise HTTPException(status_code=404, detail="Knowledge pool not found")

    # Delete vector collection
    await rag_service.delete_knowledge_pool(pool.collection_name)

    # Delete database record (cascade deletes documents)
    await db.delete(pool)
    await db.commit()

    return {"status": "deleted"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    pool_id: UUID = ...,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Upload a document to a knowledge pool.

    Supports: PDF, DOCX, TXT, MD, and other text formats.
    Document is processed in the background (embedded and indexed).
    """
    # Verify pool exists and belongs to user
    result = await db.execute(
        select(KnowledgePool).where(
            KnowledgePool.id == pool_id,
            KnowledgePool.user_id == user_id,
        )
    )
    pool = result.scalar_one_or_none()

    if not pool:
        raise HTTPException(status_code=404, detail="Knowledge pool not found")

    # Save uploaded file to temporary location
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create document record
    doc = DBDocument(
        knowledge_pool_id=pool_id,
        filename=file.filename,
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type,
        source_type="upload",
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Process document in background
    background_tasks.add_task(
        process_document_background,
        document_id=doc.id,
        file_path=file_path,
        collection_name=pool.collection_name,
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status="processing",
        message="Document is being processed in the background",
    )


async def process_document_background(
    document_id: UUID,
    file_path: str,
    collection_name: str,
):
    """
    Background task to process and index a document.

    This runs asynchronously after upload to avoid blocking the API response.
    """
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Get document from database
            result = await db.execute(
                select(DBDocument).where(DBDocument.id == document_id)
            )
            doc = result.scalar_one()

            # Update status to processing
            doc.status = DocumentStatus.PROCESSING
            await db.commit()

            # Ingest document using RAG service
            stats = await rag_service.ingest_document(
                source=file_path,
                collection_name=collection_name,
                document_id=document_id,
                metadata={"filename": doc.filename},
            )

            # Update document with stats
            doc.num_chunks = stats["num_chunks"]
            doc.num_tokens = stats["num_tokens"]
            doc.status = DocumentStatus.COMPLETED
            await db.commit()

        except Exception as e:
            # Mark document as failed
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            await db.commit()
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)


@router.post("/search", response_model=RAGSearchResponse)
async def search_documents(
    request: RAGSearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Search for relevant documents using semantic search.

    Uses vector embeddings to find documents similar to the query.
    """
    # Get collection names for the requested pools
    if request.knowledge_pool_ids:
        result = await db.execute(
            select(KnowledgePool).where(
                KnowledgePool.id.in_(request.knowledge_pool_ids),
                KnowledgePool.user_id == user_id,
            )
        )
        pools = result.scalars().all()

        if not pools:
            raise HTTPException(status_code=404, detail="No knowledge pools found")

        collection_names = [pool.collection_name for pool in pools]

        # Search across multiple pools
        results = await rag_service.search_multiple_pools(
            query=request.query,
            collection_names=collection_names,
            limit=request.limit,
        )
    else:
        # Search all user's pools
        result = await db.execute(
            select(KnowledgePool).where(KnowledgePool.user_id == user_id)
        )
        pools = result.scalars().all()

        if not pools:
            return RAGSearchResponse(
                query=request.query,
                results=[],
                num_results=0,
            )

        collection_names = [pool.collection_name for pool in pools]

        # Search across all pools
        results = await rag_service.search_multiple_pools(
            query=request.query,
            collection_names=collection_names,
            limit=request.limit,
        )

    # Convert to response format
    rag_documents = [
        RAGDocument(
            document_id=result.document_id or UUID(int=0),
            filename=result.filename,
            content=result.content,
            score=result.score,
            metadata=result.metadata,
        )
        for result in results
    ]

    return RAGSearchResponse(
        query=request.query,
        results=rag_documents,
        num_results=len(rag_documents),
    )
