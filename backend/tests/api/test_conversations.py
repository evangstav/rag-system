"""
Tests for Conversation API endpoints.

Tests conversation CRUD operations, message retrieval, and data isolation.
"""

import pytest
from uuid import uuid4
from httpx import AsyncClient
from sqlalchemy import select

from app.models.database import Conversation, Message, User


# ============================================================================
# Conversation Creation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_conversation_success(client: AsyncClient, auth_headers: dict):
    """Test successfully creating a conversation."""
    response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={
            "title": "My First Chat",
            "use_rag": True,
            "use_scratchpad": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "My First Chat"
    assert data["use_rag"] is True
    assert data["use_scratchpad"] is False
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_create_conversation_defaults(client: AsyncClient, auth_headers: dict):
    """Test creating conversation with default values."""
    response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["use_rag"] is False  # Default value
    assert data["use_scratchpad"] is False  # Default value
    assert data["title"] is None  # Can be null


@pytest.mark.asyncio
async def test_create_conversation_minimal(client: AsyncClient, auth_headers: dict):
    """Test creating conversation with minimal data."""
    response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Quick Chat"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Quick Chat"


@pytest.mark.asyncio
async def test_create_multiple_conversations(client: AsyncClient, auth_headers: dict):
    """Test creating multiple conversations."""
    titles = ["Chat 1", "Chat 2", "Chat 3"]

    for title in titles:
        response = await client.post(
            "/api/conversations/",
            headers=auth_headers,
            json={"title": title}
        )
        assert response.status_code == 200

    # List conversations
    response = await client.get("/api/conversations/", headers=auth_headers)
    assert response.status_code == 200
    conversations = response.json()
    assert len(conversations) == 3


# ============================================================================
# List Conversations Tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_conversations_success(client: AsyncClient, auth_headers: dict):
    """Test listing user's conversations."""
    # Create conversations
    await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "First"}
    )
    await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Second"}
    )

    # List conversations
    response = await client.get("/api/conversations/", headers=auth_headers)

    assert response.status_code == 200
    conversations = response.json()
    assert len(conversations) == 2
    titles = [c["title"] for c in conversations]
    assert "First" in titles
    assert "Second" in titles


@pytest.mark.asyncio
async def test_list_conversations_empty(client: AsyncClient, auth_headers: dict):
    """Test listing conversations when user has none."""
    response = await client.get("/api/conversations/", headers=auth_headers)

    assert response.status_code == 200
    conversations = response.json()
    assert len(conversations) == 0


@pytest.mark.asyncio
async def test_list_conversations_ordered_by_updated_at(
    client: AsyncClient, auth_headers: dict, db_session
):
    """Test conversations are ordered by most recently updated."""
    # Create first conversation
    response1 = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "First"}
    )
    conv1_id = response1.json()["id"]

    # Create second conversation (more recent)
    response2 = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Second"}
    )
    conv2_id = response2.json()["id"]

    # Update first conversation to make it more recent
    await client.patch(
        f"/api/conversations/{conv1_id}",
        headers=auth_headers,
        json={"title": "First Updated"}
    )

    # List conversations
    response = await client.get("/api/conversations/", headers=auth_headers)
    conversations = response.json()

    # First (updated) should be first now (most recent)
    assert conversations[0]["id"] == conv1_id
    assert conversations[1]["id"] == conv2_id


@pytest.mark.asyncio
async def test_list_conversations_pagination(client: AsyncClient, auth_headers: dict):
    """Test conversation list pagination."""
    # Create 10 conversations
    for i in range(10):
        await client.post(
            "/api/conversations/",
            headers=auth_headers,
            json={"title": f"Chat {i}"}
        )

    # Get first 5
    response = await client.get(
        "/api/conversations/?limit=5&offset=0",
        headers=auth_headers
    )
    assert response.status_code == 200
    page1 = response.json()
    assert len(page1) == 5

    # Get next 5
    response = await client.get(
        "/api/conversations/?limit=5&offset=5",
        headers=auth_headers
    )
    assert response.status_code == 200
    page2 = response.json()
    assert len(page2) == 5

    # Verify no overlap
    page1_ids = [c["id"] for c in page1]
    page2_ids = [c["id"] for c in page2]
    assert len(set(page1_ids) & set(page2_ids)) == 0


# ============================================================================
# Get Conversation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_conversation_success(client: AsyncClient, auth_headers: dict):
    """Test getting a specific conversation."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Test Chat", "use_rag": True}
    )
    conversation_id = create_response.json()["id"]

    # Get conversation
    response = await client.get(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conversation_id
    assert data["title"] == "Test Chat"
    assert data["use_rag"] is True


@pytest.mark.asyncio
async def test_get_conversation_not_found(client: AsyncClient, auth_headers: dict):
    """Test getting non-existent conversation returns 404."""
    fake_id = str(uuid4())

    response = await client.get(
        f"/api/conversations/{fake_id}",
        headers=auth_headers
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_other_users_conversation(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test users cannot access other users' conversations."""
    # User 1 creates conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "User 1 Chat"}
    )
    conversation_id = create_response.json()["id"]

    # User 2 tries to access User 1's conversation
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.get(
        f"/api/conversations/{conversation_id}",
        headers=user2_headers
    )

    assert response.status_code == 404


# ============================================================================
# Update Conversation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_conversation_title(client: AsyncClient, auth_headers: dict):
    """Test updating conversation title."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Original Title"}
    )
    conversation_id = create_response.json()["id"]

    # Update title
    response = await client.patch(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers,
        json={"title": "Updated Title"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["id"] == conversation_id


@pytest.mark.asyncio
async def test_update_conversation_toggles(client: AsyncClient, auth_headers: dict):
    """Test toggling RAG and scratchpad settings."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Test", "use_rag": False, "use_scratchpad": False}
    )
    conversation_id = create_response.json()["id"]

    # Toggle RAG on
    response = await client.patch(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers,
        json={"use_rag": True}
    )
    assert response.status_code == 200
    assert response.json()["use_rag"] is True
    assert response.json()["use_scratchpad"] is False  # Unchanged

    # Toggle scratchpad on
    response = await client.patch(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers,
        json={"use_scratchpad": True}
    )
    assert response.status_code == 200
    assert response.json()["use_rag"] is True  # Still on
    assert response.json()["use_scratchpad"] is True


@pytest.mark.asyncio
async def test_update_conversation_partial(client: AsyncClient, auth_headers: dict):
    """Test partial updates only modify specified fields."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Original", "use_rag": True, "use_scratchpad": False}
    )
    conversation_id = create_response.json()["id"]

    # Update only title
    response = await client.patch(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers,
        json={"title": "New Title"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Title"
    assert data["use_rag"] is True  # Unchanged
    assert data["use_scratchpad"] is False  # Unchanged


@pytest.mark.asyncio
async def test_update_conversation_not_found(client: AsyncClient, auth_headers: dict):
    """Test updating non-existent conversation fails."""
    fake_id = str(uuid4())

    response = await client.patch(
        f"/api/conversations/{fake_id}",
        headers=auth_headers,
        json={"title": "New"}
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_other_users_conversation(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test users cannot update other users' conversations."""
    # User 1 creates conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "User 1 Chat"}
    )
    conversation_id = create_response.json()["id"]

    # User 2 tries to update
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.patch(
        f"/api/conversations/{conversation_id}",
        headers=user2_headers,
        json={"title": "Hacked"}
    )

    assert response.status_code == 404

    # Verify original title unchanged
    verify_response = await client.get(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers
    )
    assert verify_response.json()["title"] == "User 1 Chat"


# ============================================================================
# Delete Conversation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_delete_conversation_success(client: AsyncClient, auth_headers: dict):
    """Test successfully deleting a conversation."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "To Delete"}
    )
    conversation_id = create_response.json()["id"]

    # Delete conversation
    response = await client.delete(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers
    )

    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"].lower()
    assert response.json()["id"] == conversation_id

    # Verify conversation is gone
    get_response = await client.get(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation_not_found(client: AsyncClient, auth_headers: dict):
    """Test deleting non-existent conversation fails."""
    fake_id = str(uuid4())

    response = await client.delete(
        f"/api/conversations/{fake_id}",
        headers=auth_headers
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_other_users_conversation(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test users cannot delete other users' conversations."""
    # User 1 creates conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "User 1 Chat"}
    )
    conversation_id = create_response.json()["id"]

    # User 2 tries to delete
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.delete(
        f"/api/conversations/{conversation_id}",
        headers=user2_headers
    )

    assert response.status_code == 404

    # Verify conversation still exists
    verify_response = await client.get(
        f"/api/conversations/{conversation_id}",
        headers=auth_headers
    )
    assert verify_response.status_code == 200


# ============================================================================
# Messages Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_conversation_messages(
    client: AsyncClient, auth_headers: dict, db_session, test_user: User
):
    """Test getting messages for a conversation."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Chat with Messages"}
    )
    conversation_id = create_response.json()["id"]

    # Add messages directly to database (since we're testing retrieval, not creation)
    from app.models.database import Conversation
    from uuid import UUID

    result = await db_session.execute(
        select(Conversation).where(Conversation.id == UUID(conversation_id))
    )
    conversation = result.scalar_one()

    messages_data = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"}
    ]

    for msg_data in messages_data:
        message = Message(
            conversation_id=conversation.id,
            role=msg_data["role"],
            content=msg_data["content"]
        )
        db_session.add(message)

    await db_session.commit()

    # Get messages
    response = await client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers
    )

    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_get_messages_empty_conversation(client: AsyncClient, auth_headers: dict):
    """Test getting messages from conversation with no messages."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Empty Chat"}
    )
    conversation_id = create_response.json()["id"]

    # Get messages
    response = await client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers
    )

    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_get_messages_nonexistent_conversation(client: AsyncClient, auth_headers: dict):
    """Test getting messages from non-existent conversation fails."""
    fake_id = str(uuid4())

    response = await client.get(
        f"/api/conversations/{fake_id}/messages",
        headers=auth_headers
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_messages_other_users_conversation(
    client: AsyncClient, auth_headers: dict, test_user_2: User, db_session
):
    """Test users cannot get messages from other users' conversations."""
    # User 1 creates conversation with messages
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Private Chat"}
    )
    conversation_id = create_response.json()["id"]

    # User 2 tries to get messages
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.get(
        f"/api/conversations/{conversation_id}/messages",
        headers=user2_headers
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_messages_pagination(
    client: AsyncClient, auth_headers: dict, db_session
):
    """Test message list pagination."""
    # Create conversation
    create_response = await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "Long Chat"}
    )
    conversation_id = create_response.json()["id"]

    # Add many messages
    from app.models.database import Conversation
    from uuid import UUID

    result = await db_session.execute(
        select(Conversation).where(Conversation.id == UUID(conversation_id))
    )
    conversation = result.scalar_one()

    for i in range(20):
        message = Message(
            conversation_id=conversation.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i}"
        )
        db_session.add(message)

    await db_session.commit()

    # Get first 10 messages
    response = await client.get(
        f"/api/conversations/{conversation_id}/messages?limit=10&offset=0",
        headers=auth_headers
    )
    assert response.status_code == 200
    page1 = response.json()
    assert len(page1) == 10
    assert page1[0]["content"] == "Message 0"

    # Get next 10 messages
    response = await client.get(
        f"/api/conversations/{conversation_id}/messages?limit=10&offset=10",
        headers=auth_headers
    )
    assert response.status_code == 200
    page2 = response.json()
    assert len(page2) == 10
    assert page2[0]["content"] == "Message 10"


# ============================================================================
# Authorization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_conversation_endpoints_require_auth(client: AsyncClient):
    """Test that all conversation endpoints require authentication."""
    fake_id = str(uuid4())

    endpoints = [
        ("GET", "/api/conversations/"),
        ("POST", "/api/conversations/"),
        ("GET", f"/api/conversations/{fake_id}"),
        ("PATCH", f"/api/conversations/{fake_id}"),
        ("DELETE", f"/api/conversations/{fake_id}"),
        ("GET", f"/api/conversations/{fake_id}/messages"),
    ]

    for method, endpoint in endpoints:
        if method == "GET":
            response = await client.get(endpoint)
        elif method == "POST":
            response = await client.post(endpoint, json={})
        elif method == "PATCH":
            response = await client.patch(endpoint, json={})
        elif method == "DELETE":
            response = await client.delete(endpoint)

        # FastAPI returns 403 Forbidden for missing/invalid JWT tokens
        assert response.status_code in [401, 403], f"{method} {endpoint} should require auth"


# ============================================================================
# Data Isolation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_conversations_data_isolation(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test that users only see their own conversations."""
    # User 1 creates conversations
    await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "User 1 Chat 1"}
    )
    await client.post(
        "/api/conversations/",
        headers=auth_headers,
        json={"title": "User 1 Chat 2"}
    )

    # User 2 creates conversations
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    await client.post(
        "/api/conversations/",
        headers=user2_headers,
        json={"title": "User 2 Chat 1"}
    )

    # User 1 lists conversations (should only see their own)
    response = await client.get("/api/conversations/", headers=auth_headers)
    user1_conversations = response.json()
    assert len(user1_conversations) == 2
    for conv in user1_conversations:
        assert "User 1" in conv["title"]

    # User 2 lists conversations (should only see their own)
    response = await client.get("/api/conversations/", headers=user2_headers)
    user2_conversations = response.json()
    assert len(user2_conversations) == 1
    assert user2_conversations[0]["title"] == "User 2 Chat 1"
