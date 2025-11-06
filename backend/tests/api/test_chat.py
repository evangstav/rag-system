"""
Tests for chat API endpoints.

Tests cover:
- Streaming chat responses
- RAG context injection
- Scratchpad context injection
- Conversation persistence
- Message history
- Error handling
"""

import pytest
from httpx import AsyncClient
from uuid import uuid4
import json
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.database import User, Conversation, Message, MessageRole


# =============================================================================
# Helper Functions
# =============================================================================

def parse_sse_events(sse_text: str) -> list[dict]:
    """Parse Server-Sent Events (SSE) response into list of events."""
    events = []
    for line in sse_text.split('\n'):
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])  # Skip 'data: ' prefix
                events.append(data)
            except json.JSONDecodeError:
                pass
    return events


# =============================================================================
# Basic Chat Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stream_chat_without_context(client: AsyncClient, auth_headers: dict):
    """Test basic chat streaming without RAG or scratchpad context."""
    # Mock OpenAI streaming response
    mock_stream = AsyncMock()
    mock_chunks = [
        # Chunk with content
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content="Hello"))],
            usage=None
        ),
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content=" there"))],
            usage=None
        ),
        # Final chunk with usage
        MagicMock(
            choices=[MagicMock(delta=MagicMock(content="!"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        ),
    ]
    mock_stream.__aiter__.return_value = iter(mock_chunks)

    with patch('app.api.chat.client') as mock_openai:
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = parse_sse_events(response.text)

        # Verify event structure
        event_types = [e.get('type') for e in events]
        assert 'conversation_id' in event_types
        assert 'content' in event_types
        assert 'done' in event_types

        # Verify content was streamed
        content_events = [e for e in events if e.get('type') == 'content']
        assert len(content_events) > 0

        # Verify conversation ID was returned
        conv_id_event = next(e for e in events if e.get('type') == 'conversation_id')
        assert 'conversation_id' in conv_id_event


@pytest.mark.asyncio
async def test_stream_chat_creates_conversation(
    client: AsyncClient,
    auth_headers: dict,
    db_session
):
    """Test that streaming chat creates a new conversation."""
    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Response"))],
                usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            ),
        ])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "content": "Test message"}],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200

        # Parse response to get conversation ID
        events = parse_sse_events(response.text)
        conv_id_event = next(e for e in events if e.get('type') == 'conversation_id')
        conversation_id = conv_id_event['conversation_id']

        # Verify conversation was created in database
        from sqlalchemy import select
        from uuid import UUID
        result = await db_session.execute(
            select(Conversation).where(Conversation.id == UUID(conversation_id))
        )
        conversation = result.scalar_one_or_none()

        assert conversation is not None
        assert conversation.use_rag is False
        assert conversation.use_scratchpad is False


@pytest.mark.asyncio
async def test_stream_chat_with_existing_conversation(
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
    db_session
):
    """Test streaming chat with an existing conversation."""
    # Create existing conversation
    conversation = Conversation(
        user_id=test_user.id,
        title="Test Conversation",
        use_rag=False,
        use_scratchpad=False
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Reply"))],
                usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            ),
        ])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "conversation_id": str(conversation.id),
                "messages": [{"role": "user", "content": "Follow-up"}],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200

        # Verify it used the existing conversation
        events = parse_sse_events(response.text)
        conv_id_event = next(e for e in events if e.get('type') == 'conversation_id')
        assert conv_id_event['conversation_id'] == str(conversation.id)


# =============================================================================
# RAG Context Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stream_chat_with_rag_context(
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
    db_session
):
    """Test chat with RAG context injection."""
    # Create knowledge pool
    from app.models.database import KnowledgePool
    pool = KnowledgePool(
        user_id=test_user.id,
        name="Test Pool",
        collection_name=f"test_collection_{uuid4().hex[:8]}"
    )
    db_session.add(pool)
    await db_session.commit()
    await db_session.refresh(pool)

    # Mock RAG search results
    from app.services.rag.protocols import SearchResult
    mock_results = [
        SearchResult(
            document_id=uuid4(),
            content="Test document content",
            score=0.95,
            metadata={"filename": "test.pdf"}
        )
    ]

    with patch('app.api.chat.rag_service') as mock_rag:
        mock_rag.search_multiple_pools_with_reranking = AsyncMock(return_value=mock_results)

        with patch('app.api.chat.client') as mock_openai:
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = iter([
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="Based on the document..."))],
                    usage=MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
                ),
            ])
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

            response = await client.post(
                "/api/chat/stream",
                headers=auth_headers,
                json={
                    "messages": [{"role": "user", "content": "What does the document say?"}],
                    "use_rag": True,
                    "use_scratchpad": False,
                    "knowledge_pool_ids": [str(pool.id)]
                }
            )

            assert response.status_code == 200

            # Verify RAG context was included
            events = parse_sse_events(response.text)

            # Check for metadata event with sources
            metadata_events = [e for e in events if e.get('type') == 'metadata']
            assert len(metadata_events) > 0

            metadata = metadata_events[0].get('metadata', {})
            assert 'rag_sources' in metadata
            assert len(metadata['rag_sources']) > 0
            assert metadata['rag_sources'][0]['filename'] == "test.pdf"


# =============================================================================
# Scratchpad Context Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stream_chat_with_scratchpad_context(
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
    db_session
):
    """Test chat with scratchpad context injection."""
    # Create scratchpad entries
    from app.models.database import ScratchpadEntry, ScratchpadEntryType

    todo = ScratchpadEntry(
        user_id=test_user.id,
        entry_type=ScratchpadEntryType.TODO,
        content="Write tests",
        is_completed=False
    )
    note = ScratchpadEntry(
        user_id=test_user.id,
        entry_type=ScratchpadEntryType.NOTE,
        content="Remember to focus on edge cases"
    )
    db_session.add(todo)
    db_session.add(note)
    await db_session.commit()

    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="I see you need to write tests..."))],
                usage=MagicMock(prompt_tokens=40, completion_tokens=15, total_tokens=55)
            ),
        ])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "content": "What should I work on?"}],
                "use_rag": False,
                "use_scratchpad": True
            }
        )

        assert response.status_code == 200

        # Verify scratchpad context was included
        events = parse_sse_events(response.text)
        metadata_events = [e for e in events if e.get('type') == 'metadata']

        assert len(metadata_events) > 0
        metadata = metadata_events[0].get('metadata', {})
        assert metadata.get('scratchpad_included') is True


# =============================================================================
# Combined Context Tests
# =============================================================================

@pytest.mark.asyncio
async def test_stream_chat_with_both_contexts(
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
    db_session
):
    """Test chat with both RAG and scratchpad context."""
    # Create knowledge pool
    from app.models.database import KnowledgePool, ScratchpadEntry, ScratchpadEntryType

    pool = KnowledgePool(
        user_id=test_user.id,
        name="Docs",
        collection_name=f"docs_{uuid4().hex[:8]}"
    )
    db_session.add(pool)

    # Create scratchpad entry
    todo = ScratchpadEntry(
        user_id=test_user.id,
        entry_type=ScratchpadEntryType.TODO,
        content="Review documentation",
        is_completed=False
    )
    db_session.add(todo)
    await db_session.commit()
    await db_session.refresh(pool)

    # Mock RAG results
    from app.services.rag.protocols import SearchResult
    mock_results = [
        SearchResult(
            document_id=uuid4(),
            content="Documentation content",
            score=0.9,
            metadata={"filename": "docs.pdf"}
        )
    ]

    with patch('app.api.chat.rag_service') as mock_rag:
        mock_rag.search_multiple_pools_with_reranking = AsyncMock(return_value=mock_results)

        with patch('app.api.chat.client') as mock_openai:
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = iter([
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="Combining contexts..."))],
                    usage=MagicMock(prompt_tokens=80, completion_tokens=30, total_tokens=110)
                ),
            ])
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

            response = await client.post(
                "/api/chat/stream",
                headers=auth_headers,
                json={
                    "messages": [{"role": "user", "content": "Help with my task"}],
                    "use_rag": True,
                    "use_scratchpad": True,
                    "knowledge_pool_ids": [str(pool.id)]
                }
            )

            assert response.status_code == 200

            # Verify both contexts were included
            events = parse_sse_events(response.text)
            metadata_events = [e for e in events if e.get('type') == 'metadata']

            assert len(metadata_events) > 0
            metadata = metadata_events[0].get('metadata', {})

            assert 'rag_sources' in metadata
            assert metadata.get('scratchpad_included') is True


# =============================================================================
# Message Persistence Tests
# =============================================================================

@pytest.mark.asyncio
async def test_messages_saved_to_database(
    client: AsyncClient,
    auth_headers: dict,
    test_user: User,
    db_session
):
    """Test that messages are persisted to the database."""
    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Test response"))],
                usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            ),
        ])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "content": "Test query"}],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200

        # Get conversation ID
        events = parse_sse_events(response.text)
        conv_id_event = next(e for e in events if e.get('type') == 'conversation_id')
        conversation_id = conv_id_event['conversation_id']

        # Check messages were saved
        from sqlalchemy import select
        from uuid import UUID
        result = await db_session.execute(
            select(Message).where(Message.conversation_id == UUID(conversation_id))
        )
        messages = result.scalars().all()

        # Should have user message and assistant message
        assert len(messages) >= 2

        user_messages = [m for m in messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]

        assert len(user_messages) >= 1
        assert len(assistant_messages) >= 1

        assert user_messages[0].content == "Test query"


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.asyncio
async def test_chat_without_authentication(client: AsyncClient):
    """Test that chat requires authentication."""
    response = await client.post(
        "/api/chat/stream",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "use_rag": False,
            "use_scratchpad": False
        }
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chat_with_invalid_conversation_id(
    client: AsyncClient,
    auth_headers: dict
):
    """Test chat with non-existent conversation ID."""
    fake_id = str(uuid4())

    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "conversation_id": fake_id,
                "messages": [{"role": "user", "content": "Test"}],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        # Should get error event in stream
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        error_events = [e for e in events if e.get('type') == 'error']
        assert len(error_events) > 0


@pytest.mark.asyncio
async def test_chat_with_empty_message(client: AsyncClient, auth_headers: dict):
    """Test chat with no messages."""
    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="Hello!"))],
                usage=MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            ),
        ])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        # API accepts empty messages (e.g., for starting new conversation)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_non_user_last_message(
    client: AsyncClient,
    auth_headers: dict
):
    """Test that last message must be from user."""
    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={
                "messages": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Response"}
                ],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        error_events = [e for e in events if e.get('type') == 'error']
        assert len(error_events) > 0
        assert "Last message must be from user" in error_events[0].get('error', '')


# =============================================================================
# Data Isolation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_users_cannot_access_other_conversations(
    client: AsyncClient,
    test_user: User,
    another_user: User,
    auth_headers: dict,
    db_session
):
    """Test that users cannot access conversations from other users."""
    # Create conversation for another_user
    conversation = Conversation(
        user_id=another_user.id,
        title="Private Conversation",
        use_rag=False,
        use_scratchpad=False
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    # Try to access it with test_user's token
    with patch('app.api.chat.client') as mock_openai:
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = iter([])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream)

        response = await client.post(
            "/api/chat/stream",
            headers=auth_headers,  # test_user's token
            json={
                "conversation_id": str(conversation.id),
                "messages": [{"role": "user", "content": "Trying to access"}],
                "use_rag": False,
                "use_scratchpad": False
            }
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        error_events = [e for e in events if e.get('type') == 'error']

        # Should get "Conversation not found" error
        assert len(error_events) > 0
        assert "not found" in error_events[0].get('error', '').lower()
