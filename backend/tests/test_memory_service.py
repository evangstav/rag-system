"""
Tests for Memory Service.

Tests cover:
- Memory extraction from conversations
- Memory extraction from journal entries
- Memory storage with deduplication
- Memory retrieval with multi-factor scoring
- Memory CRUD operations
- Qdrant integration
- Helper method unit tests (fetch, format, extract, store)
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.memory_service import MemoryService
from app.models.database import (
    User,
    UserMemory,
    Message,
    MessageRole,
    Conversation,
    ScratchpadEntry,
    ScratchpadEntryType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embeddings():
    """Mock embeddings provider that creates unique embeddings for different content."""
    import random

    mock = AsyncMock()
    mock.dimensions = 1536

    # Generate deterministic but VERY different embeddings based on text content
    async def embed_text(text: str):
        import hashlib

        # Use sha512 for more bits and create truly unique embeddings per text
        hash_bytes = hashlib.sha512(text.encode()).digest()

        # Create a seeded random generator for this specific text
        seed = int.from_bytes(hash_bytes[:8], "big")
        rng = random.Random(seed)

        # Generate random but deterministic embedding for this text
        return [rng.random() for _ in range(1536)]

    async def embed_batch(texts: list[str]):
        return [await embed_text(text) for text in texts]

    mock.embed_text = embed_text
    mock.embed_batch = embed_batch
    return mock


@pytest_asyncio.fixture
async def mock_vector_store():
    """Mock Qdrant vector store."""
    from tests.mocks.qdrant_mock import MockQdrantClient

    mock_store = MagicMock()
    mock_store.client = MockQdrantClient()

    # Create the collection upfront
    await mock_store.client.create_collection(
        collection_name="user_memories",
        vectors_config={"size": 1536, "distance": "Cosine"},
    )

    # Mock create_collection to handle future calls
    async def create_collection(collection_name, vector_size, distance):
        if collection_name not in mock_store.client.collections:
            await mock_store.client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": distance},
            )

    mock_store.create_collection = create_collection

    # Mock the search method to return structured results with proper similarity scores
    async def mock_search(collection_name, query_vector, limit=10, **kwargs):
        import numpy as np

        # Return mock search results with proper structure
        results = []
        points = mock_store.client.collections.get(collection_name, [])

        for i, point in enumerate(points[:limit]):
            # Calculate actual cosine similarity between query and point vectors
            if hasattr(point, "vector") and query_vector:
                try:
                    # Compute cosine similarity
                    dot_product = np.dot(query_vector, point.vector)
                    norm_query = np.linalg.norm(query_vector)
                    norm_point = np.linalg.norm(point.vector)
                    similarity = (
                        dot_product / (norm_query * norm_point)
                        if norm_query > 0 and norm_point > 0
                        else 0.0
                    )
                    score = max(0.0, min(1.0, similarity))  # Clamp to [0,1]
                except Exception:
                    # Fallback to low scores if calculation fails
                    score = 0.5 - (i * 0.1)
            else:
                # If no vectors, return decreasing low scores
                score = 0.5 - (i * 0.1)

            result = MagicMock()
            result.id = point.id if hasattr(point, "id") else f"test_id_{i}"
            result.score = score
            result.metadata = point.payload if hasattr(point, "payload") else {}
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    mock_store.search = mock_search
    return mock_store


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI LLM client."""
    mock = AsyncMock()

    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "memories": [
            {
                "content": "User is a Python developer working on RAG systems",
                "category": "context",
                "importance": 0.8
            },
            {
                "content": "User prefers detailed technical explanations",
                "category": "preference",
                "importance": 0.7
            }
        ]
    }
    """

    # Set up the mock to return the response
    mock.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock


@pytest_asyncio.fixture
async def test_conversation(db_session: AsyncSession, test_user: User):
    """Create a test conversation with messages."""
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
            content="I'm a Python developer working on a RAG system",
        ),
        Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content="That's great! How can I help you with your RAG system?",
        ),
        Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="I prefer detailed technical explanations with code examples",
        ),
    ]

    for msg in messages:
        db_session.add(msg)

    await db_session.commit()
    return conversation


@pytest_asyncio.fixture
async def test_journal_entries(db_session: AsyncSession, test_user: User):
    """Create test journal entries."""
    entries = []
    for i in range(3):
        entry = ScratchpadEntry(
            user_id=test_user.id,
            entry_type=ScratchpadEntryType.JOURNAL,
            content=f"Day {i + 1}: Working on implementing memory system. Learning about vector embeddings and semantic search.",
            entry_date=datetime.utcnow() - timedelta(days=i),
        )
        db_session.add(entry)
        entries.append(entry)

    await db_session.commit()
    return entries


# =============================================================================
# Memory Extraction Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extract_memories_from_conversation(
    db_session: AsyncSession,
    test_user: User,
    test_conversation: Conversation,
    mock_embeddings,
    mock_vector_store,
    mock_llm_client,
):
    """Test extracting memories from a conversation."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
    )

    # Extract memories
    memories = await service.extract_memories_from_conversation(
        conversation_id=test_conversation.id,
        user_id=test_user.id,
        limit_messages=20,
    )

    print(memories)

    # Assertions
    assert len(memories) == 2
    assert all(m.user_id == test_user.id for m in memories)
    assert all(m.source_conversation_id == test_conversation.id for m in memories)

    # Check that expected content is present (order may vary due to deduplication)
    contents = [m.content for m in memories]
    assert any("Python developer" in c for c in contents)
    assert any("technical explanations" in c for c in contents)

    # Check importance values
    importances = [m.importance for m in memories]
    assert 0.7 in importances or 0.8 in importances

    # Verify LLM was called
    mock_llm_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_extract_memories_from_journal(
    db_session: AsyncSession,
    test_user: User,
    test_journal_entries,
    mock_embeddings,
    mock_vector_store,
    mock_llm_client,
):
    """Test extracting memories from journal entries."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
    )

    # Extract memories
    memories = await service.extract_memories_from_journal(
        user_id=test_user.id,
        days_back=7,
    )

    # Assertions
    assert len(memories) == 2
    assert memories[0].user_id == test_user.id
    assert memories[0].source_conversation_id is None  # From journal, not conversation

    # Verify LLM was called
    mock_llm_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_extract_memories_no_new_info(
    db_session: AsyncSession,
    test_user: User,
    test_conversation: Conversation,
    mock_embeddings,
    mock_vector_store,
):
    """Test extraction when no new memories are found."""
    # Mock LLM to return empty memories
    mock_llm = AsyncMock()

    async def create_empty(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"memories": []}'
        return response

    mock_llm.chat.completions.create = create_empty

    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )

    memories = await service.extract_memories_from_conversation(
        conversation_id=test_conversation.id,
        user_id=test_user.id,
    )

    assert len(memories) == 0


# =============================================================================
# Memory Storage Tests
# =============================================================================


@pytest.mark.asyncio
async def test_add_memory_success(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test successfully adding a new memory."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory
    memory = await service.add_memory(
        user_id=test_user.id,
        content="User prefers Python for backend development",
        importance=0.8,
        category="preference",
    )

    # Assertions
    assert memory is not None
    assert memory.user_id == test_user.id
    assert memory.content == "User prefers Python for backend development"
    assert memory.importance == 0.8
    assert memory.qdrant_id is not None
    # Verify it's a valid UUID string
    from uuid import UUID
    UUID(memory.qdrant_id)  # Will raise ValueError if not a valid UUID


@pytest.mark.asyncio
async def test_add_memory_deduplication(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test that similar memories are deduplicated."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add first memory
    memory1 = await service.add_memory(
        user_id=test_user.id,
        content="User likes Python",
        importance=0.7,
    )

    # Try to add very similar memory
    memory2 = await service.add_memory(
        user_id=test_user.id,
        content="User likes Python",  # Nearly identical
        importance=0.8,
    )

    # Should update existing memory instead of creating new one
    assert memory2.id == memory1.id
    assert memory2.importance == 0.8  # Updated to higher importance


# =============================================================================
# Memory Retrieval Tests
# =============================================================================


@pytest.mark.asyncio
async def test_retrieve_memories_multi_factor_scoring(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test memory retrieval with multi-factor scoring."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add test memories with different characteristics
    old_important = await service.add_memory(
        user_id=test_user.id,
        content="Old but very important memory",
        importance=0.9,
    )
    # Make it old
    old_important.updated_at = datetime.utcnow() - timedelta(days=30)
    await db_session.commit()

    recent_less_important = await service.add_memory(
        user_id=test_user.id,
        content="Recent but less important memory",
        importance=0.5,
    )

    # Retrieve memories
    memories = await service.retrieve_memories(
        user_id=test_user.id,
        query="important memory",
        limit=10,
    )

    # Should have results
    assert len(memories) > 0

    # Results should be sorted by combined score
    # (semantic + recency + importance)
    assert all(isinstance(m, UserMemory) for m in memories)


@pytest.mark.asyncio
async def test_retrieve_memories_user_isolation(
    db_session: AsyncSession,
    test_user: User,
    another_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test that users can only retrieve their own memories."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory for test_user
    await service.add_memory(
        user_id=test_user.id,
        content="Test user's private memory",
        importance=0.8,
    )

    # Add memory for another_user
    await service.add_memory(
        user_id=another_user.id,
        content="Another user's private memory",
        importance=0.8,
    )

    # Retrieve memories for test_user
    memories = await service.retrieve_memories(
        user_id=test_user.id,
        query="memory",
        limit=10,
    )

    # Should only get test_user's memories
    assert all(m.user_id == test_user.id for m in memories)
    assert not any(m.user_id == another_user.id for m in memories)


@pytest.mark.asyncio
async def test_get_all_memories(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test getting all memories for a user."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add multiple memories with very distinct content to avoid deduplication
    for i in range(5):
        await service.add_memory(
            user_id=test_user.id,
            content=f"Unique memory about topic {i}: {'x' * (i + 10)}",  # Make each very distinct
            importance=0.5 + (i * 0.1),
        )

    # Get all memories
    memories = await service.get_all_memories(
        user_id=test_user.id,
        limit=100,
    )

    # Assertions
    assert len(memories) >= 5  # Should have at least 5 (may have more from other tests)

    # Should be sorted by importance (descending) then by updated_at
    for i in range(len(memories) - 1):
        assert memories[i].importance >= memories[i + 1].importance or (
            memories[i].importance == memories[i + 1].importance
            and memories[i].updated_at >= memories[i + 1].updated_at
        )


# =============================================================================
# Memory Update Tests
# =============================================================================


@pytest.mark.asyncio
async def test_update_memory_content(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test updating memory content."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory
    memory = await service.add_memory(
        user_id=test_user.id,
        content="Original content",
        importance=0.7,
    )
    original_id = memory.id

    # Update memory
    updated = await service.update_memory(
        memory_id=memory.id,
        user_id=test_user.id,
        content="Updated content",
    )

    # Assertions
    assert updated is not None
    assert updated.id == original_id
    assert updated.content == "Updated content"
    assert updated.importance == 0.7  # Unchanged


@pytest.mark.asyncio
async def test_update_memory_importance(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test updating memory importance."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory
    memory = await service.add_memory(
        user_id=test_user.id,
        content="Test memory",
        importance=0.5,
    )

    # Update importance
    updated = await service.update_memory(
        memory_id=memory.id,
        user_id=test_user.id,
        importance=0.9,
    )

    assert updated.importance == 0.9
    assert updated.content == "Test memory"  # Unchanged


@pytest.mark.asyncio
async def test_update_memory_not_found(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test updating non-existent memory."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Try to update non-existent memory
    result = await service.update_memory(
        memory_id=uuid4(),
        user_id=test_user.id,
        content="New content",
    )

    assert result is None


# =============================================================================
# Memory Deletion Tests
# =============================================================================


@pytest.mark.asyncio
async def test_delete_memory_success(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test successfully deleting a memory."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory
    memory = await service.add_memory(
        user_id=test_user.id,
        content="Memory to delete",
        importance=0.5,
    )
    memory_id = memory.id

    # Delete memory
    success = await service.delete_memory(
        memory_id=memory_id,
        user_id=test_user.id,
    )

    assert success is True

    # Verify memory is gone
    from sqlalchemy import select

    result = await db_session.execute(
        select(UserMemory).where(UserMemory.id == memory_id)
    )
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_all_memories(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test deleting all memories for a user."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add multiple memories with distinct content
    for i in range(5):
        await service.add_memory(
            user_id=test_user.id,
            content=f"Distinct memory for deletion test {i}: {'y' * (i + 15)}",
            importance=0.5,
        )

    # Delete all
    count = await service.delete_all_memories(user_id=test_user.id)

    assert count >= 5  # Should delete at least 5

    # Verify all are gone
    from sqlalchemy import select

    result = await db_session.execute(
        select(UserMemory).where(UserMemory.user_id == test_user.id)
    )
    assert len(result.scalars().all()) == 0


@pytest.mark.asyncio
async def test_delete_memory_wrong_user(
    db_session: AsyncSession,
    test_user: User,
    another_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test that users cannot delete other users' memories."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memory for test_user
    memory = await service.add_memory(
        user_id=test_user.id,
        content="Test user's memory",
        importance=0.5,
    )

    # Try to delete with another_user
    success = await service.delete_memory(
        memory_id=memory.id,
        user_id=another_user.id,
    )

    # Should fail
    assert success is False

    # Memory should still exist
    from sqlalchemy import select

    result = await db_session.execute(
        select(UserMemory).where(UserMemory.id == memory.id)
    )
    assert result.scalar_one_or_none() is not None


# =============================================================================
# Helper Method Tests (New Refactored Methods)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_conversation_messages(
    db_session: AsyncSession,
    test_user: User,
    test_conversation: Conversation,
    mock_embeddings,
    mock_vector_store,
):
    """Test _fetch_conversation_messages helper returns messages in chronological order."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Fetch messages
    messages = await service._fetch_conversation_messages(
        conversation_id=test_conversation.id,
        limit=20,
    )

    # Assertions
    assert len(messages) == 3
    # First message should be USER role
    assert messages[0].role == MessageRole.USER
    # Check that conversation content is present
    all_content = " ".join([m.content for m in messages])
    assert "Python developer" in all_content or "RAG system" in all_content
    # Verify chronological order (oldest first)
    for i in range(len(messages) - 1):
        assert messages[i].created_at <= messages[i + 1].created_at


@pytest.mark.asyncio
async def test_fetch_conversation_messages_empty(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test _fetch_conversation_messages with no messages."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Create empty conversation
    conversation = Conversation(user_id=test_user.id, title="Empty")
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    messages = await service._fetch_conversation_messages(
        conversation_id=conversation.id,
        limit=20,
    )

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_fetch_journal_entries_recent(
    db_session: AsyncSession,
    test_user: User,
    test_journal_entries,
    mock_embeddings,
    mock_vector_store,
):
    """Test _fetch_journal_entries for recent entries."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Fetch recent entries (7 days back)
    entries = await service._fetch_journal_entries(
        user_id=test_user.id,
        entry_date=None,
        days_back=7,
    )

    # Should get all 3 test entries (created 0, 1, 2 days ago)
    assert len(entries) == 3
    assert all(entry.entry_type == ScratchpadEntryType.JOURNAL for entry in entries)
    assert all(entry.user_id == test_user.id for entry in entries)


@pytest.mark.asyncio
async def test_fetch_journal_entries_specific_date(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test _fetch_journal_entries for a specific date."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Create entry for specific date
    specific_date = datetime(2024, 1, 15, 10, 0, 0)
    entry = ScratchpadEntry(
        user_id=test_user.id,
        entry_type=ScratchpadEntryType.JOURNAL,
        content="Entry on Jan 15",
        entry_date=specific_date,
    )
    db_session.add(entry)
    await db_session.commit()

    # Fetch entries for that specific date
    entries = await service._fetch_journal_entries(
        user_id=test_user.id,
        entry_date=specific_date,
        days_back=7,
    )

    assert len(entries) >= 1
    assert any(e.content == "Entry on Jan 15" for e in entries)


@pytest.mark.asyncio
async def test_get_existing_memories_text(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test _get_existing_memories_text helper."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # With no memories
    text = await service._get_existing_memories_text(test_user.id)
    assert text == "None"

    # Add some memories
    await service.add_memory(
        user_id=test_user.id,
        content="User prefers Python",
        importance=0.8,
    )
    await service.add_memory(
        user_id=test_user.id,
        content="User works on RAG systems",
        importance=0.7,
    )

    # Get formatted text
    text = await service._get_existing_memories_text(test_user.id)

    # At least one of the memories should be present
    assert "User prefers Python" in text or "User works on RAG systems" in text
    assert text.startswith("- ")
    # Should have formatted content


@pytest.mark.asyncio
async def test_call_llm_for_extraction_success(
    db_session: AsyncSession,
    mock_embeddings,
    mock_vector_store,
    mock_llm_client,
):
    """Test _call_llm_for_extraction with successful extraction."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
    )

    prompt = "Test prompt"
    result = await service._call_llm_for_extraction(prompt, "conversation")

    # Should return parsed JSON
    assert result is not None
    assert "memories" in result
    assert len(result["memories"]) == 2
    assert (
        result["memories"][0]["content"]
        == "User is a Python developer working on RAG systems"
    )


@pytest.mark.asyncio
async def test_call_llm_for_extraction_failure(
    db_session: AsyncSession,
    mock_embeddings,
    mock_vector_store,
):
    """Test _call_llm_for_extraction with LLM failure."""
    # Mock LLM that raises error
    mock_llm = AsyncMock()
    mock_llm.chat.completions.create.side_effect = Exception("API Error")

    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )

    result = await service._call_llm_for_extraction("test prompt", "journal")

    # Should return None on error
    assert result is None


@pytest.mark.asyncio
async def test_store_extracted_memories(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test _store_extracted_memories helper."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Mock extracted data
    extracted_data = {
        "memories": [
            {
                "content": "User likes FastAPI",
                "category": "preference",
                "importance": 0.8,
            },
            {
                "content": "User is learning embeddings",
                "category": "goal",
                "importance": 0.7,
            },
        ]
    }

    # Store memories
    memories = await service._store_extracted_memories(
        user_id=test_user.id,
        extracted_data=extracted_data,
        source_conversation_id=None,
    )

    # Assertions
    assert len(memories) == 2
    # Check that both memories were stored (order may vary)
    contents = [m.content for m in memories]
    importances = [m.importance for m in memories]
    assert "User likes FastAPI" in contents
    assert "User is learning embeddings" in contents
    assert 0.8 in importances
    assert 0.7 in importances
    assert all(m.source_conversation_id is None for m in memories)


@pytest.mark.asyncio
async def test_store_extracted_memories_with_conversation(
    db_session: AsyncSession,
    test_user: User,
    test_conversation: Conversation,
    mock_embeddings,
    mock_vector_store,
):
    """Test _store_extracted_memories with source conversation."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    extracted_data = {
        "memories": [
            {
                "content": "User mentioned a project",
                "category": "context",
                "importance": 0.6,
            },
        ]
    }

    memories = await service._store_extracted_memories(
        user_id=test_user.id,
        extracted_data=extracted_data,
        source_conversation_id=test_conversation.id,
    )

    assert len(memories) == 1
    assert memories[0].source_conversation_id == test_conversation.id


# =============================================================================
# Formatting Helper Tests
# =============================================================================


@pytest.mark.asyncio
async def test_format_memories_for_context(
    db_session: AsyncSession,
    test_user: User,
    mock_embeddings,
    mock_vector_store,
):
    """Test formatting memories for LLM context."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Add memories
    memories = []
    for i in range(3):
        memory = await service.add_memory(
            user_id=test_user.id,
            content=f"Important fact {i}",
            importance=0.7,
        )
        memories.append(memory)

    # Format for context
    context = service.format_memories_for_context(memories, max_length=500)

    # Assertions
    assert isinstance(context, str)
    assert len(context) <= 500
    # At least some of the memories should be present
    assert any(f"Important fact {i}" in context for i in range(5))


@pytest.mark.asyncio
async def test_format_memories_empty_list(
    db_session: AsyncSession,
    mock_embeddings,
    mock_vector_store,
):
    """Test formatting empty memories list."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    context = service.format_memories_for_context([], max_length=500)

    assert context == "No relevant memories."


@pytest.mark.asyncio
async def test_format_messages(
    db_session: AsyncSession,
    test_user: User,
    test_conversation: Conversation,
    mock_embeddings,
    mock_vector_store,
):
    """Test _format_messages helper."""
    service = MemoryService(
        db=db_session,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
    )

    # Fetch and format messages
    messages = await service._fetch_conversation_messages(
        conversation_id=test_conversation.id,
        limit=20,
    )
    formatted = service._format_messages(messages)

    # Assertions
    assert isinstance(formatted, str)
    assert "User:" in formatted
    assert "Assistant:" in formatted
    assert "Python developer" in formatted
