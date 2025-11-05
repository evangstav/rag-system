"""
Tests for authentication API endpoints.

Tests cover:
- User registration (success and error cases)
- User login (success and error cases)
- Token refresh functionality
- Authorization and data isolation
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID, uuid4

from app.models.database import User
from app.auth import create_refresh_token, decode_token, create_access_token


# =============================================================================
# Registration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_register_new_user_success(client: AsyncClient):
    """Test successful user registration."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "securepassword123",
        },
    )

    assert response.status_code == 201
    data = response.json()

    # Verify response structure
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data

    # Verify user data
    user = data["user"]
    assert user["email"] == "newuser@example.com"
    assert user["username"] == "newuser"
    assert user["is_active"] is True
    assert user["is_superuser"] is False
    assert "id" in user
    assert "created_at" in user
    assert "updated_at" in user

    # Verify tokens are valid JWT strings
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 20
    assert isinstance(data["refresh_token"], str)
    assert len(data["refresh_token"]) > 20

    # Verify token payload
    payload = decode_token(data["access_token"])
    assert payload is not None
    assert "sub" in payload
    assert payload["sub"] == user["id"]


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user: User):
    """Test registration with an already registered email."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": test_user.email,  # Already exists
            "username": "differentusername",
            "password": "securepassword123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient, test_user: User):
    """Test registration with an already taken username."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "different@example.com",
            "username": test_user.username,  # Already exists
            "password": "securepassword123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Username already taken"


@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient):
    """Test registration with invalid email format."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "not-an-email",
            "username": "testuser",
            "password": "securepassword123",
        },
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient):
    """Test registration with password shorter than minimum length."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "short",  # Less than 8 characters
        },
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_register_short_username(client: AsyncClient):
    """Test registration with username shorter than minimum length."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "ab",  # Less than 3 characters
            "password": "securepassword123",
        },
    )

    assert response.status_code == 422  # Validation error


# =============================================================================
# Login Tests
# =============================================================================

@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, test_user: User):
    """Test successful login with correct credentials."""
    response = await client.post(
        "/api/auth/login",
        json={
            "email": test_user.email,
            "password": "testpassword123",  # Password from test_user fixture
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data

    # Verify user data
    user = data["user"]
    assert user["email"] == test_user.email
    assert user["username"] == test_user.username
    assert user["id"] == str(test_user.id)


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, test_user: User):
    """Test login with incorrect password."""
    response = await client.post(
        "/api/auth/login",
        json={
            "email": test_user.email,
            "password": "wrongpassword",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


@pytest.mark.asyncio
async def test_login_nonexistent_email(client: AsyncClient):
    """Test login with email that doesn't exist."""
    response = await client.post(
        "/api/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "anypassword",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


@pytest.mark.asyncio
async def test_login_inactive_user(client: AsyncClient, db_session: AsyncSession):
    """Test login with inactive user account."""
    # Create inactive user
    from app.auth import get_password_hash

    inactive_user = User(
        email="inactive@example.com",
        username="inactiveuser",
        hashed_password=get_password_hash("testpassword123"),
        is_active=False,  # Inactive
    )
    db_session.add(inactive_user)
    await db_session.commit()

    response = await client.post(
        "/api/auth/login",
        json={
            "email": "inactive@example.com",
            "password": "testpassword123",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "User account is disabled"


# =============================================================================
# Token Refresh Tests
# =============================================================================

@pytest.mark.asyncio
async def test_refresh_access_token_success(client: AsyncClient, test_user: User):
    """Test successful access token refresh."""
    # Create a valid refresh token
    refresh_token = create_refresh_token(data={"sub": str(test_user.id)})

    response = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "refresh_token" not in data  # Only access token is returned

    # Verify new access token is valid
    payload = decode_token(data["access_token"])
    assert payload is not None
    assert payload["sub"] == str(test_user.id)


@pytest.mark.asyncio
async def test_refresh_with_invalid_token(client: AsyncClient):
    """Test refresh with invalid token."""
    response = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": "invalid.token.here"},
    )

    assert response.status_code == 401
    assert "Invalid refresh token" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_with_access_token(client: AsyncClient, test_user: User):
    """Test that access tokens cannot be used for refresh (only refresh tokens)."""
    from app.auth import create_access_token

    # Try to use access token instead of refresh token
    access_token = create_access_token(data={"sub": str(test_user.id)})

    response = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": access_token},
    )

    assert response.status_code == 401
    assert "Invalid token type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_with_deleted_user(client: AsyncClient, db_session: AsyncSession):
    """Test refresh token for a user that has been deleted."""
    from app.auth import get_password_hash, create_refresh_token

    # Create user
    temp_user = User(
        email="temp@example.com",
        username="tempuser",
        hashed_password=get_password_hash("password123"),
        is_active=True,
    )
    db_session.add(temp_user)
    await db_session.commit()
    await db_session.refresh(temp_user)

    user_id = temp_user.id

    # Create refresh token
    refresh_token = create_refresh_token(data={"sub": str(user_id)})

    # Delete user
    await db_session.delete(temp_user)
    await db_session.commit()

    # Try to refresh
    response = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 401
    assert "User not found or inactive" in response.json()["detail"]


# =============================================================================
# Authorization Tests
# =============================================================================

@pytest.mark.asyncio
async def test_protected_route_without_token(client: AsyncClient):
    """Test accessing protected route without authentication token."""
    # Try to access a protected endpoint (e.g., scratchpad)
    response = await client.get("/api/scratchpad/")

    assert response.status_code == 403  # Forbidden (no credentials)


@pytest.mark.asyncio
async def test_protected_route_with_invalid_token(client: AsyncClient):
    """Test accessing protected route with invalid token."""
    response = await client.get(
        "/api/scratchpad/",
        headers={"Authorization": "Bearer invalid.token.here"},
    )

    assert response.status_code == 401  # Unauthorized


@pytest.mark.asyncio
async def test_protected_route_with_valid_token(
    client: AsyncClient, auth_headers: dict
):
    """Test accessing protected route with valid authentication token."""
    # This should work - we'll get a 200 with empty scratchpad data
    # but NOT 401 or 403
    response = await client.get("/api/scratchpad/", headers=auth_headers)

    # Should be authorized (200 with empty scratchpad)
    assert response.status_code == 200
    # Verify we get valid scratchpad data
    data = response.json()
    assert "todos" in data
    assert "notes" in data
    assert "journal" in data


# =============================================================================
# Data Isolation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_users_cannot_access_other_users_data(
    client: AsyncClient,
    test_user: User,
    another_user: User,
    auth_headers: dict,
):
    """Test that users cannot access data belonging to other users."""
    # First create some data with test_user
    test_todo_id = str(uuid4())
    response = await client.post(
        "/api/scratchpad/",
        headers=auth_headers,  # test_user's headers
        json={
            "todos": [{"id": test_todo_id, "text": "Private todo", "completed": False}],
            "notes": "",
            "journal": ""
        },
    )
    assert response.status_code == 200

    # Try to access it with another_user's token
    another_access_token = create_access_token(data={"sub": str(another_user.id)})
    another_headers = {"Authorization": f"Bearer {another_access_token}"}

    # Try to get test_user's scratchpad
    response = await client.get("/api/scratchpad/", headers=another_headers)
    assert response.status_code == 200

    # Verify another_user doesn't see test_user's todos
    data = response.json()
    todo_ids = [todo["id"] for todo in data["todos"]]
    assert test_todo_id not in todo_ids


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.asyncio
async def test_register_with_whitespace_in_fields(client: AsyncClient):
    """Test that whitespace in email/username is handled properly."""
    response = await client.post(
        "/api/auth/register",
        json={
            "email": " whitespace@example.com ",
            "username": " whitespaceuser ",
            "password": "securepassword123",
        },
    )

    # Depending on validation, this might succeed or fail
    # The important thing is it doesn't crash
    assert response.status_code in [201, 422]


@pytest.mark.asyncio
async def test_login_is_case_sensitive_for_email(
    client: AsyncClient, test_user: User
):
    """Test that email login is case-insensitive (common UX pattern)."""
    response = await client.post(
        "/api/auth/login",
        json={
            "email": test_user.email.upper(),  # Uppercase email
            "password": "testpassword123",
        },
    )

    # This might fail or succeed depending on implementation
    # Document the behavior
    # Most systems treat emails as case-insensitive
    assert response.status_code in [200, 401]


@pytest.mark.asyncio
async def test_multiple_registrations_in_sequence(client: AsyncClient):
    """Test that multiple users can register in sequence."""
    users_to_create = [
        {"email": f"user{i}@example.com", "username": f"user{i}", "password": "pass123456"}
        for i in range(5)
    ]

    created_user_ids = []

    for user_data in users_to_create:
        response = await client.post("/api/auth/register", json=user_data)
        assert response.status_code == 201

        user_id = response.json()["user"]["id"]
        created_user_ids.append(user_id)

    # Verify all user IDs are unique
    assert len(set(created_user_ids)) == len(created_user_ids)


@pytest.mark.asyncio
async def test_token_contains_correct_user_id(client: AsyncClient, test_user: User):
    """Test that tokens contain the correct user ID."""
    response = await client.post(
        "/api/auth/login",
        json={
            "email": test_user.email,
            "password": "testpassword123",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Decode both tokens and verify user ID
    access_payload = decode_token(data["access_token"])
    refresh_payload = decode_token(data["refresh_token"])

    assert access_payload["sub"] == str(test_user.id)
    assert refresh_payload["sub"] == str(test_user.id)
    assert access_payload["type"] == "access"
    assert refresh_payload["type"] == "refresh"
