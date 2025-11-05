"""
Tests for scratchpad API endpoints.

Tests cover:
- Get scratchpad data (todos, notes, journal)
- Save scratchpad data
- CRUD operations for different entry types
- Context formatting for chat integration
- Data isolation between users
"""

import pytest
from httpx import AsyncClient
from uuid import uuid4
from datetime import datetime, date

from app.models.database import User, ScratchpadEntry, ScratchpadEntryType


# =============================================================================
# Get Scratchpad Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_empty_scratchpad(client: AsyncClient, auth_headers: dict):
    """Test getting scratchpad when user has no entries."""
    response = await client.get("/api/scratchpad/", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()

    # Should return empty scratchpad
    assert data["todos"] == []
    assert data["notes"] == ""
    assert data["journal"] == ""


@pytest.mark.asyncio
async def test_get_scratchpad_with_data(client: AsyncClient, auth_headers: dict):
    """Test getting scratchpad with existing data."""
    # First create some scratchpad data
    scratchpad_data = {
        "todos": [
            {"id": str(uuid4()), "text": "Complete project", "completed": False},
            {"id": str(uuid4()), "text": "Review PR", "completed": True},
        ],
        "notes": "Important notes about the project",
        "journal": "Today I worked on testing",
    }

    # Save the data
    save_response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )
    assert save_response.status_code == 200

    # Get the data
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    assert get_response.status_code == 200

    data = get_response.json()

    # Verify all data is returned
    assert len(data["todos"]) == 2
    assert data["notes"] == "Important notes about the project"
    assert data["journal"] == "Today I worked on testing"

    # Verify todos content
    todo_texts = [todo["text"] for todo in data["todos"]]
    assert "Complete project" in todo_texts
    assert "Review PR" in todo_texts


@pytest.mark.asyncio
async def test_get_scratchpad_requires_auth(client: AsyncClient):
    """Test that getting scratchpad requires authentication."""
    response = await client.get("/api/scratchpad/")

    # Should be forbidden without auth
    assert response.status_code == 403


# =============================================================================
# Save Scratchpad Tests
# =============================================================================

@pytest.mark.asyncio
async def test_save_scratchpad_todos(client: AsyncClient, auth_headers: dict):
    """Test saving todos to scratchpad."""
    todo_id = str(uuid4())
    scratchpad_data = {
        "todos": [
            {"id": todo_id, "text": "Buy groceries", "completed": False},
        ],
        "notes": "",
        "journal": "",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    # Verify the todo was saved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert len(data["todos"]) == 1
    assert data["todos"][0]["id"] == todo_id
    assert data["todos"][0]["text"] == "Buy groceries"
    assert data["todos"][0]["completed"] is False


@pytest.mark.asyncio
async def test_save_scratchpad_notes(client: AsyncClient, auth_headers: dict):
    """Test saving notes to scratchpad."""
    scratchpad_data = {
        "todos": [],
        "notes": "This is an important note about my project.",
        "journal": "",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Verify the note was saved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["notes"] == "This is an important note about my project."


@pytest.mark.asyncio
async def test_save_scratchpad_journal(client: AsyncClient, auth_headers: dict):
    """Test saving journal entry to scratchpad."""
    scratchpad_data = {
        "todos": [],
        "notes": "",
        "journal": "Today was productive. Completed all tests.",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Verify the journal was saved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["journal"] == "Today was productive. Completed all tests."


@pytest.mark.asyncio
async def test_save_scratchpad_requires_auth(client: AsyncClient):
    """Test that saving scratchpad requires authentication."""
    response = await client.post(
        "/api/scratchpad/",
        json={"todos": [], "notes": "", "journal": ""},
    )

    # Should be forbidden without auth
    assert response.status_code == 403


# =============================================================================
# Todo CRUD Operations
# =============================================================================

@pytest.mark.asyncio
async def test_update_existing_todo(client: AsyncClient, auth_headers: dict):
    """Test updating an existing todo."""
    todo_id = str(uuid4())

    # Create initial todo
    initial_data = {
        "todos": [{"id": todo_id, "text": "Original text", "completed": False}],
        "notes": "",
        "journal": "",
    }
    await client.post("/api/scratchpad/", headers=auth_headers, json=initial_data)

    # Update the todo
    updated_data = {
        "todos": [{"id": todo_id, "text": "Updated text", "completed": True}],
        "notes": "",
        "journal": "",
    }
    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=updated_data,
    )

    assert response.status_code == 200

    # Verify the update
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert len(data["todos"]) == 1
    assert data["todos"][0]["id"] == todo_id
    assert data["todos"][0]["text"] == "Updated text"
    assert data["todos"][0]["completed"] is True


@pytest.mark.asyncio
async def test_delete_todo(client: AsyncClient, auth_headers: dict):
    """Test deleting a todo by not including it in save request."""
    todo_id_1 = str(uuid4())
    todo_id_2 = str(uuid4())

    # Create two todos
    initial_data = {
        "todos": [
            {"id": todo_id_1, "text": "Keep this", "completed": False},
            {"id": todo_id_2, "text": "Delete this", "completed": False},
        ],
        "notes": "",
        "journal": "",
    }
    await client.post("/api/scratchpad/", headers=auth_headers, json=initial_data)

    # Save with only one todo (deletes the other)
    updated_data = {
        "todos": [{"id": todo_id_1, "text": "Keep this", "completed": False}],
        "notes": "",
        "journal": "",
    }
    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=updated_data,
    )

    assert response.status_code == 200

    # Verify only one todo remains
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert len(data["todos"]) == 1
    assert data["todos"][0]["id"] == todo_id_1
    assert data["todos"][0]["text"] == "Keep this"


@pytest.mark.asyncio
async def test_add_multiple_todos(client: AsyncClient, auth_headers: dict):
    """Test adding multiple todos at once."""
    todos = [
        {"id": str(uuid4()), "text": f"Task {i}", "completed": False}
        for i in range(5)
    ]

    scratchpad_data = {"todos": todos, "notes": "", "journal": ""}

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Verify all todos were saved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert len(data["todos"]) == 5
    todo_texts = [todo["text"] for todo in data["todos"]]
    for i in range(5):
        assert f"Task {i}" in todo_texts


# =============================================================================
# Notes and Journal Operations
# =============================================================================

@pytest.mark.asyncio
async def test_update_notes(client: AsyncClient, auth_headers: dict):
    """Test updating notes replaces the old content."""
    # Create initial notes
    initial_data = {"todos": [], "notes": "Original notes", "journal": ""}
    await client.post("/api/scratchpad/", headers=auth_headers, json=initial_data)

    # Update notes
    updated_data = {"todos": [], "notes": "Updated notes", "journal": ""}
    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=updated_data,
    )

    assert response.status_code == 200

    # Verify notes were updated
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["notes"] == "Updated notes"


@pytest.mark.asyncio
async def test_clear_notes(client: AsyncClient, auth_headers: dict):
    """Test clearing notes by sending empty string."""
    # Create initial notes
    initial_data = {"todos": [], "notes": "Some notes", "journal": ""}
    await client.post("/api/scratchpad/", headers=auth_headers, json=initial_data)

    # Clear notes
    cleared_data = {"todos": [], "notes": "", "journal": ""}
    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=cleared_data,
    )

    assert response.status_code == 200

    # Verify notes were cleared
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["notes"] == ""


@pytest.mark.asyncio
async def test_journal_only_returns_today(client: AsyncClient, auth_headers: dict):
    """Test that journal entry is date-specific (only today's entry is returned)."""
    # Save journal entry
    scratchpad_data = {"todos": [], "notes": "", "journal": "Today's journal entry"}

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Get scratchpad (should return today's journal)
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["journal"] == "Today's journal entry"


# =============================================================================
# Data Isolation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_users_cannot_see_other_users_scratchpad(
    client: AsyncClient,
    test_user: User,
    another_user: User,
    auth_headers: dict,
):
    """Test that users can only see their own scratchpad data."""
    # test_user creates scratchpad data
    test_user_data = {
        "todos": [{"id": str(uuid4()), "text": "Test user's todo", "completed": False}],
        "notes": "Test user's notes",
        "journal": "Test user's journal",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,  # test_user's auth
        json=test_user_data,
    )
    assert response.status_code == 200

    # another_user logs in and checks scratchpad
    from app.auth import create_access_token

    another_token = create_access_token(data={"sub": str(another_user.id)})
    another_headers = {"Authorization": f"Bearer {another_token}"}

    response = await client.get("/api/scratchpad/", headers=another_headers)
    assert response.status_code == 200

    data = response.json()

    # another_user should see empty scratchpad
    assert data["todos"] == []
    assert data["notes"] == ""
    assert data["journal"] == ""


@pytest.mark.asyncio
async def test_users_can_have_different_scratchpad_data(
    client: AsyncClient,
    test_user: User,
    another_user: User,
    auth_headers: dict,
):
    """Test that different users can have different scratchpad data."""
    # test_user creates data
    test_user_data = {"todos": [{"id": str(uuid4()), "text": "User 1 todo", "completed": False}], "notes": "", "journal": ""}
    await client.post("/api/scratchpad/", headers=auth_headers, json=test_user_data)

    # another_user creates different data
    from app.auth import create_access_token

    another_token = create_access_token(data={"sub": str(another_user.id)})
    another_headers = {"Authorization": f"Bearer {another_token}"}

    another_user_data = {"todos": [{"id": str(uuid4()), "text": "User 2 todo", "completed": False}], "notes": "", "journal": ""}
    await client.post("/api/scratchpad/", headers=another_headers, json=another_user_data)

    # Verify test_user's data
    response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = response.json()
    assert data["todos"][0]["text"] == "User 1 todo"

    # Verify another_user's data
    response = await client.get("/api/scratchpad/", headers=another_headers)
    data = response.json()
    assert data["todos"][0]["text"] == "User 2 todo"


# =============================================================================
# Edge Cases and Validation
# =============================================================================

@pytest.mark.asyncio
async def test_save_empty_scratchpad(client: AsyncClient, auth_headers: dict):
    """Test saving completely empty scratchpad."""
    empty_data = {"todos": [], "notes": "", "journal": ""}

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=empty_data,
    )

    assert response.status_code == 200

    # Verify it was saved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["todos"] == []
    assert data["notes"] == ""
    assert data["journal"] == ""


@pytest.mark.asyncio
async def test_todo_with_long_text(client: AsyncClient, auth_headers: dict):
    """Test that todos can handle long text content."""
    long_text = "This is a very long todo item. " * 50  # ~1500 characters

    scratchpad_data = {
        "todos": [{"id": str(uuid4()), "text": long_text, "completed": False}],
        "notes": "",
        "journal": "",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Verify it was saved correctly
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert data["todos"][0]["text"] == long_text


@pytest.mark.asyncio
async def test_markdown_in_notes(client: AsyncClient, auth_headers: dict):
    """Test that notes can contain markdown formatting."""
    markdown_notes = """
# Header

- Bullet 1
- Bullet 2

**Bold text** and *italic text*

```python
def hello():
    print("Hello")
```
"""

    scratchpad_data = {"todos": [], "notes": markdown_notes, "journal": ""}

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=scratchpad_data,
    )

    assert response.status_code == 200

    # Verify markdown was preserved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert "# Header" in data["notes"]
    assert "**Bold text**" in data["notes"]
    assert "```python" in data["notes"]


@pytest.mark.asyncio
async def test_special_characters_in_scratchpad(client: AsyncClient, auth_headers: dict):
    """Test that special characters are handled correctly."""
    special_data = {
        "todos": [
            {"id": str(uuid4()), "text": "Todo with emoji 🎉", "completed": False}
        ],
        "notes": "Notes with special chars: @#$%^&*()[]{}",
        "journal": "Journal with unicode: café, naïve, 日本語",
    }

    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,
        json=special_data,
    )

    assert response.status_code == 200

    # Verify special characters were preserved
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    assert "emoji 🎉" in data["todos"][0]["text"]
    assert "@#$%^&*()" in data["notes"]
    assert "日本語" in data["journal"]


@pytest.mark.asyncio
async def test_rapid_successive_saves(client: AsyncClient, auth_headers: dict):
    """Test that rapid successive saves work correctly (like auto-save)."""
    # Simulate auto-save: save 5 times in rapid succession
    for i in range(5):
        scratchpad_data = {
            "todos": [{"id": str(uuid4()), "text": f"Todo version {i}", "completed": False}],
            "notes": f"Notes version {i}",
            "journal": f"Journal version {i}",
        }

        response = await client.post(
            "/api/scratchpad/",
            headers=auth_headers,
            json=scratchpad_data,
        )
        assert response.status_code == 200

    # Verify final state
    get_response = await client.get("/api/scratchpad/", headers=auth_headers)
    data = get_response.json()

    # Should have the last version
    assert "version 4" in data["notes"]
    assert "version 4" in data["journal"]
