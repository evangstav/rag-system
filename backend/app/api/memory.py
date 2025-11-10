"""
Memory API endpoints.

Handles user memory management (CRUD operations and extraction).
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from app.dependencies import get_db, get_current_active_user
from app.models.database import User
from app.models.schemas import UserMemoryCreate, UserMemoryResponse
from app.services.memory_service import MemoryService

router = APIRouter()


def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    """Dependency to get memory service instance."""
    return MemoryService(db=db)


@router.get("/", response_model=List[UserMemoryResponse])
async def get_memories(
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Get all memories for the authenticated user.

    Returns memories ordered by importance and recency.

    Args:
        limit: Maximum number of memories to return (default 100)
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        List of UserMemoryResponse objects
    """
    memories = await memory_service.get_all_memories(
        user_id=current_user.id,
        limit=limit,
    )

    return memories


@router.post(
    "/", response_model=UserMemoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_memory(
    memory_data: UserMemoryCreate,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Manually create a new memory.

    Args:
        memory_data: Memory content and importance
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        Created UserMemoryResponse object
    """
    memory = await memory_service.add_memory(
        user_id=current_user.id,
        content=memory_data.content,
        importance=memory_data.importance,
        source_conversation_id=memory_data.source_conversation_id,
    )

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create memory",
        )

    return memory


@router.put("/{memory_id}", response_model=UserMemoryResponse)
async def update_memory(
    memory_id: UUID,
    content: Optional[str] = None,
    importance: Optional[float] = None,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Update an existing memory.

    Args:
        memory_id: Memory ID to update
        content: New content (optional)
        importance: New importance score (optional)
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        Updated UserMemoryResponse object

    Raises:
        HTTPException: If memory not found
    """
    memory = await memory_service.update_memory(
        memory_id=memory_id,
        user_id=current_user.id,
        content=content,
        importance=importance,
    )

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    # Manually construct response to avoid greenlet issues with ORM lazy loading
    return UserMemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        content=memory.content,
        importance=memory.importance,
        source_conversation_id=memory.source_conversation_id,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Delete a specific memory.

    Args:
        memory_id: Memory ID to delete
        current_user: Authenticated user
        memory_service: Memory service instance

    Raises:
        HTTPException: If memory not found
    """
    success = await memory_service.delete_memory(
        memory_id=memory_id,
        user_id=current_user.id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )


@router.delete("/", status_code=status.HTTP_200_OK)
async def delete_all_memories(
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Delete all memories for the authenticated user.

    Args:
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        Number of memories deleted
    """
    count = await memory_service.delete_all_memories(user_id=current_user.id)
    return {"deleted_count": count}


@router.post(
    "/extract/conversation/{conversation_id}", status_code=status.HTTP_202_ACCEPTED
)
async def extract_from_conversation(
    conversation_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Extract memories from a conversation (background task).

    Args:
        conversation_id: Conversation ID to extract from
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        Accepted message (extraction runs in background)
    """
    # Add background task
    background_tasks.add_task(
        memory_service.extract_memories_from_conversation,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    return {
        "status": "accepted",
        "message": "Memory extraction started in background",
    }


@router.post("/extract/journal", response_model=List[UserMemoryResponse])
async def extract_from_journal(
    days_back: int = 7,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Extract memories from recent journal entries.

    Args:
        days_back: Number of days back to analyze (default 7)
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        List of newly extracted memories
    """
    memories = await memory_service.extract_memories_from_journal(
        user_id=current_user.id,
        days_back=days_back,
    )

    return memories


@router.get("/search", response_model=List[UserMemoryResponse])
async def search_memories(
    query: str,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """
    Search memories using semantic search with multi-factor scoring.

    Args:
        query: Search query
        limit: Maximum number of results (default 10)
        current_user: Authenticated user
        memory_service: Memory service instance

    Returns:
        List of relevant memories, scored and ranked
    """
    memories = await memory_service.retrieve_memories(
        user_id=current_user.id,
        query=query,
        limit=limit,
    )

    return memories
