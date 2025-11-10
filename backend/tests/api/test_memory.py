"""
Tests for Memory API endpoints.

Tests cover:
- GET /api/memory/ - List all memories
- POST /api/memory/ - Create memory
- PUT /api/memory/{id} - Update memory
- DELETE /api/memory/{id} - Delete memory
- DELETE /api/memory/ - Delete all memories
- POST /api/memory/extract/conversation/{id} - Extract from conversation
- POST /api/memory/extract/journal - Extract from journal
- GET /api/memory/search - Search memories
- Authorization and data isolation
"""

from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Conversation,
    Message,
    MessageRole,
    ScratchpadEntry,
    ScratchpadEntryType,
    User,
    UserMemory,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_memory(db_session: AsyncSession, test_user: User):
    """Create a test memory."""
    memory = UserMemory(
        user_id=test_user.id,
        content="User is a Python developer",
        importance=0.8,
        qdrant_id="test_memory_1",
    )
    db_session.add(memory)
    await db_session.commit()
    await db_session.refresh(memory)
    return memory


@pytest_asyncio.fixture
async def multiple_memories(db_session: AsyncSession, test_user: User):
    """Create multiple test memories."""
    memories = []
    for i in range(5):
        memory = UserMemory(
            user_id=test_user.id,
            content=f"Memory fact {i}",
            importance=0.5 + (i * 0.1),
            qdrant_id=f"test_memory_{i}",
        )
        db_session.add(memory)
        memories.append(memory)

    await db_session.commit()
    for memory in memories:
        await db_session.refresh(memory)
    return memories


@pytest_asyncio.fixture
async def test_conversation_with_messages(db_session: AsyncSession, test_user: User):
    """Create a conversation with messages for memory extraction."""
    conversation = Conversation(
        user_id=test_user.id,
        title="Test Conversation",
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    # Add messages
    messages = [
        Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="I prefer Python for backend development",
        ),
        Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content="Great choice! Python is excellent for backend work.",
        ),
    ]

    for msg in messages:
        db_session.add(msg)

    await db_session.commit()
    return conversation


@pytest_asyncio.fixture
async def test_journal_entries(db_session: AsyncSession, test_user: User):
    """Create test journal entries."""
    for i in range(3):
        entry = ScratchpadEntry(
            user_id=test_user.id,
            entry_type=ScratchpadEntryType.JOURNAL,
            content=f"Day {i + 1}: Learning about vector databases",
            entry_date=datetime.now() - timedelta(days=i),
        )
        db_session.add(entry)

    await db_session.commit()


# =============================================================================
# GET /api/memory/ - List Memories
# =============================================================================


@pytest.mark.asyncio
async def test_get_memories_success(
    client: AsyncClient,
    auth_headers: dict,
    multiple_memories,
):
    """Test successfully getting all memories."""
    response = await client.get(
        "/api/memory/",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 5

    # Verify structure
    for memory in data:
        assert "id" in memory
        assert "content" in memory
        assert "importance" in memory
        assert "user_id" in memory
        assert "created_at" in memory
        assert "updated_at" in memory

    # Should be sorted by importance (descending)
    importances = [m["importance"] for m in data]
    assert importances == sorted(importances, reverse=True)


@pytest.mark.asyncio
async def test_get_memories_empty(
    client: AsyncClient,
    auth_headers: dict,
):
    """Test getting memories when none exist."""
    response = await client.get(
        "/api/memory/",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_get_memories_with_limit(
    client: AsyncClient,
    auth_headers: dict,
    multiple_memories,
):
    """Test getting memories with limit parameter."""
    response = await client.get(
        "/api/memory/?limit=3",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


@pytest.mark.asyncio
async def test_get_memories_unauthorized(client: AsyncClient):
    """Test getting memories without authentication."""
    response = await client.get("/api/memory/")

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


@pytest.mark.asyncio
async def test_get_memories_user_isolation(
    client: AsyncClient,
    auth_headers: dict,
    db_session: AsyncSession,
    test_user: User,
    another_user: User,
):
    """Test that users can only see their own memories."""
    # Add memory for test_user
    test_memory = UserMemory(
        user_id=test_user.id,
        content="Test user's memory",
        importance=0.8,
        qdrant_id="test_1",
    )
    db_session.add(test_memory)

    # Add memory for another_user
    other_memory = UserMemory(
        user_id=another_user.id,
        content="Another user's memory",
        importance=0.8,
        qdrant_id="test_2",
    )
    db_session.add(other_memory)
    await db_session.commit()

    # Get memories with test_user's auth
    response = await client.get(
        "/api/memory/",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    # Should only get test_user's memory
    assert len(data) == 1
    assert data[0]["content"] == "Test user's memory"


# =============================================================================
# POST /api/memory/ - Create Memory
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.add_memory")
async def test_create_memory_success(
    mock_add_memory,
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
):
    """Test successfully creating a memory."""
    # Mock the add_memory method
    mock_memory = UserMemory(
        id=uuid4(),
        user_id=test_user.id,
        content="New memory",
        importance=0.7,
        qdrant_id="new_mem_1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_add_memory.return_value = mock_memory

    response = await client.post(
        "/api/memory/",
        headers=auth_headers,
        json={
            "content": "New memory",
            "importance": 0.7,
        },
    )

    assert response.status_code == 201
    data = response.json()

    assert data["content"] == "New memory"
    assert data["importance"] == 0.7
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_memory_invalid_importance(
    client: AsyncClient,
    auth_headers: dict,
):
    """Test creating memory with invalid importance value."""
    response = await client.post(
        "/api/memory/",
        headers=auth_headers,
        json={
            "content": "Test memory",
            "importance": 1.5,  # Invalid: > 1.0
        },
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_memory_unauthorized(client: AsyncClient):
    """Test creating memory without authentication."""
    response = await client.post(
        "/api/memory/",
        json={
            "content": "Test memory",
            "importance": 0.7,
        },
    )

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# PUT /api/memory/{id} - Update Memory
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.update_memory")
async def test_update_memory_content(
    mock_update,
    client: AsyncClient,
    auth_headers: dict,
    test_memory: UserMemory,
):
    """Test updating memory content."""
    # Create a fresh, detached memory object to avoid greenlet issues
    updated_memory = UserMemory(
        id=test_memory.id,
        user_id=test_memory.user_id,
        content="Updated content",
        importance=test_memory.importance,
        qdrant_id=test_memory.qdrant_id,
        source_conversation_id=test_memory.source_conversation_id,
        created_at=test_memory.created_at,
        updated_at=test_memory.updated_at,
    )
    mock_update.return_value = updated_memory

    response = await client.put(
        f"/api/memory/{test_memory.id}?content=Updated%20content",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Updated content"


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.update_memory")
async def test_update_memory_importance(
    mock_update,
    client: AsyncClient,
    auth_headers: dict,
    test_memory: UserMemory,
):
    """Test updating memory importance."""
    # Create a fresh, detached memory object to avoid greenlet issues
    updated_memory = UserMemory(
        id=test_memory.id,
        user_id=test_memory.user_id,
        content=test_memory.content,
        importance=0.9,
        qdrant_id=test_memory.qdrant_id,
        source_conversation_id=test_memory.source_conversation_id,
        created_at=test_memory.created_at,
        updated_at=test_memory.updated_at,
    )
    mock_update.return_value = updated_memory

    response = await client.put(
        f"/api/memory/{test_memory.id}?importance=0.9",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["importance"] == 0.9


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.update_memory")
async def test_update_memory_not_found(
    mock_update,
    client: AsyncClient,
    auth_headers: dict,
):
    """Test updating non-existent memory."""
    mock_update.return_value = None

    response = await client.put(
        f"/api/memory/{uuid4()}?content=Updated",
        headers=auth_headers,
    )

    assert response.status_code == 404


# =============================================================================
# DELETE /api/memory/{id} - Delete Memory
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.delete_memory")
async def test_delete_memory_success(
    mock_delete,
    client: AsyncClient,
    auth_headers: dict,
    test_memory: UserMemory,
):
    """Test successfully deleting a memory."""
    mock_delete.return_value = True

    response = await client.delete(
        f"/api/memory/{test_memory.id}",
        headers=auth_headers,
    )

    assert response.status_code == 204


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.delete_memory")
async def test_delete_memory_not_found(
    mock_delete,
    client: AsyncClient,
    auth_headers: dict,
):
    """Test deleting non-existent memory."""
    mock_delete.return_value = False

    response = await client.delete(
        f"/api/memory/{uuid4()}",
        headers=auth_headers,
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_memory_unauthorized(
    client: AsyncClient,
    test_memory: UserMemory,
):
    """Test deleting memory without authentication."""
    response = await client.delete(f"/api/memory/{test_memory.id}")

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# DELETE /api/memory/ - Delete All Memories
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.delete_all_memories")
async def test_delete_all_memories_success(
    mock_delete_all,
    client: AsyncClient,
    auth_headers: dict,
):
    """Test successfully deleting all memories."""
    mock_delete_all.return_value = 5

    response = await client.delete(
        "/api/memory/",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["deleted_count"] == 5


@pytest.mark.asyncio
async def test_delete_all_memories_unauthorized(client: AsyncClient):
    """Test deleting all memories without authentication."""
    response = await client.delete("/api/memory/")

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# POST /api/memory/extract/conversation/{id} - Extract from Conversation
# =============================================================================


@pytest.mark.asyncio
async def test_extract_from_conversation_success(
    client: AsyncClient,
    auth_headers: dict,
    test_conversation_with_messages: Conversation,
):
    """Test extracting memories from conversation."""
    response = await client.post(
        f"/api/memory/extract/conversation/{test_conversation_with_messages.id}",
        headers=auth_headers,
    )

    assert response.status_code == 202  # Accepted
    data = response.json()
    assert data["status"] == "accepted"
    assert "background" in data["message"]


@pytest.mark.asyncio
async def test_extract_from_conversation_unauthorized(
    client: AsyncClient,
    test_conversation_with_messages: Conversation,
):
    """Test extracting memories without authentication."""
    response = await client.post(
        f"/api/memory/extract/conversation/{test_conversation_with_messages.id}"
    )

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# POST /api/memory/extract/journal - Extract from Journal
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.extract_memories_from_journal")
async def test_extract_from_journal_success(
    mock_extract,
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
):
    """Test extracting memories from journal entries."""
    # Mock extracted memories
    mock_memories = [
        UserMemory(
            id=uuid4(),
            user_id=test_user.id,
            content="Extracted memory 1",
            importance=0.8,
            qdrant_id="mem_1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        UserMemory(
            id=uuid4(),
            user_id=test_user.id,
            content="Extracted memory 2",
            importance=0.7,
            qdrant_id="mem_2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]
    mock_extract.return_value = mock_memories

    response = await client.post(
        "/api/memory/extract/journal?days_back=7",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["content"] == "Extracted memory 1"


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.extract_memories_from_journal")
async def test_extract_from_journal_no_memories(
    mock_extract,
    client: AsyncClient,
    auth_headers: dict,
):
    """Test extracting when no new memories are found."""
    mock_extract.return_value = []

    response = await client.post(
        "/api/memory/extract/journal",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_extract_from_journal_unauthorized(client: AsyncClient):
    """Test extracting from journal without authentication."""
    response = await client.post("/api/memory/extract/journal")

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# GET /api/memory/search - Search Memories
# =============================================================================


@pytest.mark.asyncio
@patch("app.services.memory_service.MemoryService.retrieve_memories")
async def test_search_memories_success(
    mock_retrieve,
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
):
    """Test searching memories."""
    # Mock search results
    mock_results = [
        UserMemory(
            id=uuid4(),
            user_id=test_user.id,
            content="Relevant memory about Python",
            importance=0.9,
            qdrant_id="mem_1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]
    mock_retrieve.return_value = mock_results

    response = await client.get(
        "/api/memory/search?query=Python&limit=10",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 1
    assert "Python" in data[0]["content"]


@pytest.mark.asyncio
async def test_search_memories_no_results(
    client: AsyncClient,
    auth_headers: dict,
):
    """Test searching with no results."""
    response = await client.get(
        "/api/memory/search?query=nonexistent",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_search_memories_missing_query(
    client: AsyncClient,
    auth_headers: dict,
):
    """Test searching without query parameter."""
    response = await client.get(
        "/api/memory/search",
        headers=auth_headers,
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_search_memories_unauthorized(client: AsyncClient):
    """Test searching memories without authentication."""
    response = await client.get("/api/memory/search?query=test")

    assert response.status_code == 403  # HTTPBearer returns 403 for missing credentials


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_memory_lifecycle(
    client: AsyncClient,
    auth_headers: dict,
):
    """Test complete memory lifecycle: create, update, retrieve, delete."""
    # 1. Create memory (mocked)
    with patch("app.services.memory_service.MemoryService.add_memory") as mock_add:
        mock_memory = UserMemory(
            id=uuid4(),
            user_id=uuid4(),
            content="Test lifecycle memory",
            importance=0.7,
            qdrant_id="lifecycle_1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_add.return_value = mock_memory

        create_response = await client.post(
            "/api/memory/",
            headers=auth_headers,
            json={"content": "Test lifecycle memory", "importance": 0.7},
        )
        assert create_response.status_code == 201
        memory_id = create_response.json()["id"]

    # 2. Update memory
    with patch(
        "app.services.memory_service.MemoryService.update_memory"
    ) as mock_update:
        mock_memory.importance = 0.9
        mock_update.return_value = mock_memory

        update_response = await client.put(
            f"/api/memory/{memory_id}?importance=0.9",
            headers=auth_headers,
        )
        assert update_response.status_code == 200

    # 3. Delete memory
    with patch(
        "app.services.memory_service.MemoryService.delete_memory"
    ) as mock_delete:
        mock_delete.return_value = True

        delete_response = await client.delete(
            f"/api/memory/{memory_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 204
