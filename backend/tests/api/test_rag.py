"""
Tests for RAG API endpoints.

Tests knowledge pool management, document upload/deletion, and semantic search.
"""

import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from httpx import AsyncClient
from sqlalchemy import select

from app.models.database import KnowledgePool, Document as DBDocument, DocumentStatus, User


# ============================================================================
# Knowledge Pool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_knowledge_pool_success(client: AsyncClient, auth_headers: dict):
    """Test successfully creating a knowledge pool."""
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={
                "name": "My Research Papers",
                "description": "Collection of AI research papers"
            }
        )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Research Papers"
    assert data["description"] == "Collection of AI research papers"
    assert "id" in data
    assert "collection_name" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_knowledge_pool_duplicate_name(
    client: AsyncClient, auth_headers: dict, db_session
):
    """Test that creating a pool with duplicate collection name fails."""
    # Create first pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        response1 = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Research", "description": "First pool"}
        )
    assert response1.status_code == 201

    # Try to create second pool with same name (should fail)
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        response2 = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Research", "description": "Duplicate pool"}
        )
    assert response2.status_code == 400
    assert "already exists" in response2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_knowledge_pool_validation(client: AsyncClient, auth_headers: dict):
    """Test knowledge pool creation validation."""
    # Empty name
    response = await client.post(
        "/api/rag/knowledge-pools",
        headers=auth_headers,
        json={"name": "", "description": "Test"}
    )
    assert response.status_code == 422

    # Missing name
    response = await client.post(
        "/api/rag/knowledge-pools",
        headers=auth_headers,
        json={"description": "Test"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_knowledge_pools(client: AsyncClient, auth_headers: dict):
    """Test listing user's knowledge pools."""
    # Create multiple pools
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Pool 1", "description": "First pool"}
        )
        await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Pool 2", "description": "Second pool"}
        )

    # List pools
    response = await client.get("/api/rag/knowledge-pools", headers=auth_headers)

    assert response.status_code == 200
    pools = response.json()
    assert len(pools) == 2
    assert pools[0]["name"] in ["Pool 1", "Pool 2"]
    assert pools[1]["name"] in ["Pool 1", "Pool 2"]


@pytest.mark.asyncio
async def test_list_knowledge_pools_empty(client: AsyncClient, auth_headers: dict):
    """Test listing pools when user has none."""
    response = await client.get("/api/rag/knowledge-pools", headers=auth_headers)

    assert response.status_code == 200
    pools = response.json()
    assert len(pools) == 0


@pytest.mark.asyncio
async def test_delete_knowledge_pool_success(client: AsyncClient, auth_headers: dict):
    """Test successfully deleting a knowledge pool."""
    # Create pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        create_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "To Delete", "description": "Will be deleted"}
        )
    assert create_response.status_code == 201
    pool_id = create_response.json()["id"]

    # Delete pool
    with patch('app.api.rag.rag_service.delete_knowledge_pool', new=AsyncMock()):
        delete_response = await client.delete(
            f"/api/rag/knowledge-pools/{pool_id}",
            headers=auth_headers
        )

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    # Verify pool is gone
    list_response = await client.get("/api/rag/knowledge-pools", headers=auth_headers)
    pools = list_response.json()
    assert not any(p["id"] == pool_id for p in pools)


@pytest.mark.asyncio
async def test_delete_knowledge_pool_not_found(client: AsyncClient, auth_headers: dict):
    """Test deleting non-existent pool returns 404."""
    fake_pool_id = str(uuid4())

    response = await client.delete(
        f"/api/rag/knowledge-pools/{fake_pool_id}",
        headers=auth_headers
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_knowledge_pool_data_isolation(
    client: AsyncClient, auth_headers: dict, test_user_2: User, db_session
):
    """Test that users cannot access other users' knowledge pools."""
    # User 1 creates a pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "User 1 Pool", "description": "Private pool"}
        )
    assert response.status_code == 201
    pool_id = response.json()["id"]

    # User 2's auth headers
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    # User 2 tries to delete User 1's pool (should fail)
    with patch('app.api.rag.rag_service.delete_knowledge_pool', new=AsyncMock()):
        delete_response = await client.delete(
            f"/api/rag/knowledge-pools/{pool_id}",
            headers=user2_headers
        )

    assert delete_response.status_code == 404  # Not found (because it doesn't belong to user 2)


# ============================================================================
# Document Upload Tests
# ============================================================================


@pytest.mark.asyncio
async def test_upload_document_success(client: AsyncClient, auth_headers: dict):
    """Test successfully uploading a document."""
    # Create knowledge pool first
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Test Pool", "description": "For uploads"}
        )
    pool_id = pool_response.json()["id"]

    # Create a fake PDF file
    fake_file_content = b"%PDF-1.4 fake pdf content"
    files = {"file": ("test.pdf", io.BytesIO(fake_file_content), "application/pdf")}
    data = {"pool_id": pool_id}

    # Upload document
    response = await client.post(
        "/api/rag/upload",
        headers=auth_headers,
        files=files,
        data=data
    )

    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "test.pdf"
    assert result["status"] == "processing"
    assert "document_id" in result
    assert "message" in result


@pytest.mark.asyncio
async def test_upload_document_to_nonexistent_pool(client: AsyncClient, auth_headers: dict):
    """Test uploading to non-existent pool fails."""
    fake_pool_id = str(uuid4())
    fake_file_content = b"test content"
    files = {"file": ("test.txt", io.BytesIO(fake_file_content), "text/plain")}
    data = {"pool_id": fake_pool_id}

    response = await client.post(
        "/api/rag/upload",
        headers=auth_headers,
        files=files,
        data=data
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_document_to_other_users_pool(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test uploading to another user's pool fails."""
    # User 1 creates pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "User 1 Pool", "description": "Private"}
        )
    pool_id = pool_response.json()["id"]

    # User 2 tries to upload to User 1's pool
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    fake_file_content = b"test content"
    files = {"file": ("test.txt", io.BytesIO(fake_file_content), "text/plain")}
    data = {"pool_id": pool_id}

    response = await client.post(
        "/api/rag/upload",
        headers=user2_headers,
        files=files,
        data=data
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_multiple_document_types(client: AsyncClient, auth_headers: dict):
    """Test uploading various document types."""
    # Create pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Multi-format Pool", "description": "Test"}
        )
    pool_id = pool_response.json()["id"]

    # Test different file types
    test_files = [
        ("test.pdf", b"%PDF-1.4 content", "application/pdf"),
        ("test.txt", b"text content", "text/plain"),
        ("test.md", b"# Markdown\nContent", "text/markdown"),
    ]

    for filename, content, mime_type in test_files:
        files = {"file": (filename, io.BytesIO(content), mime_type)}
        data = {"pool_id": pool_id}

        response = await client.post(
            "/api/rag/upload",
            headers=auth_headers,
            files=files,
            data=data
        )

        assert response.status_code == 200, f"Failed for {filename}"
        assert response.json()["filename"] == filename


# ============================================================================
# Document Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_documents_in_pool(client: AsyncClient, auth_headers: dict, db_session):
    """Test listing documents in a knowledge pool."""
    # Create pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Doc Pool", "description": "Test"}
        )
    pool_id = pool_response.json()["id"]

    # Upload documents
    for i in range(3):
        files = {"file": (f"doc{i}.txt", io.BytesIO(b"content"), "text/plain")}
        data = {"pool_id": pool_id}
        await client.post("/api/rag/upload", headers=auth_headers, files=files, data=data)

    # List documents
    response = await client.get(
        f"/api/rag/knowledge-pools/{pool_id}/documents",
        headers=auth_headers
    )

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 3
    filenames = [doc["filename"] for doc in documents]
    assert "doc0.txt" in filenames
    assert "doc1.txt" in filenames
    assert "doc2.txt" in filenames


@pytest.mark.asyncio
async def test_list_documents_empty_pool(client: AsyncClient, auth_headers: dict):
    """Test listing documents in empty pool."""
    # Create pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Empty Pool", "description": "No docs"}
        )
    pool_id = pool_response.json()["id"]

    # List documents
    response = await client.get(
        f"/api/rag/knowledge-pools/{pool_id}/documents",
        headers=auth_headers
    )

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 0


@pytest.mark.asyncio
async def test_list_documents_nonexistent_pool(client: AsyncClient, auth_headers: dict):
    """Test listing documents in non-existent pool fails."""
    fake_pool_id = str(uuid4())

    response = await client.get(
        f"/api/rag/knowledge-pools/{fake_pool_id}/documents",
        headers=auth_headers
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_success(client: AsyncClient, auth_headers: dict):
    """Test successfully deleting a document."""
    # Create pool and upload document
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Test Pool", "description": "Test"}
        )
    pool_id = pool_response.json()["id"]

    files = {"file": ("to_delete.txt", io.BytesIO(b"content"), "text/plain")}
    data = {"pool_id": pool_id}
    upload_response = await client.post(
        "/api/rag/upload",
        headers=auth_headers,
        files=files,
        data=data
    )
    document_id = upload_response.json()["document_id"]

    # Delete document
    with patch('app.api.rag.rag_service.vector_store.delete_by_document_id', new=AsyncMock()):
        delete_response = await client.delete(
            f"/api/rag/documents/{document_id}",
            headers=auth_headers
        )

    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True

    # Verify document is gone
    list_response = await client.get(
        f"/api/rag/knowledge-pools/{pool_id}/documents",
        headers=auth_headers
    )
    documents = list_response.json()
    assert not any(doc["id"] == document_id for doc in documents)


@pytest.mark.asyncio
async def test_delete_document_not_found(client: AsyncClient, auth_headers: dict):
    """Test deleting non-existent document fails."""
    fake_document_id = str(uuid4())

    response = await client.delete(
        f"/api/rag/documents/{fake_document_id}",
        headers=auth_headers
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_other_users_document(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test users cannot delete other users' documents."""
    # User 1 uploads document
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "User 1 Pool", "description": "Test"}
        )
    pool_id = pool_response.json()["id"]

    files = {"file": ("user1_doc.txt", io.BytesIO(b"content"), "text/plain")}
    data = {"pool_id": pool_id}
    upload_response = await client.post(
        "/api/rag/upload",
        headers=auth_headers,
        files=files,
        data=data
    )
    document_id = upload_response.json()["document_id"]

    # User 2 tries to delete
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.delete(
        f"/api/rag/documents/{document_id}",
        headers=user2_headers
    )

    assert response.status_code == 404


# ============================================================================
# Search Tests
# ============================================================================


@pytest.mark.asyncio
async def test_search_in_specific_pools(client: AsyncClient, auth_headers: dict):
    """Test searching in specific knowledge pools."""
    # Create pools
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool1_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "AI Papers", "description": "Test"}
        )
        pool2_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "History Papers", "description": "Test"}
        )
    pool1_id = pool1_response.json()["id"]
    pool2_id = pool2_response.json()["id"]

    # Mock search results
    from app.models.schemas import RAGDocument
    mock_results = [
        MagicMock(
            document_id=uuid4(),
            filename="result1.pdf",
            content="Machine learning is...",
            score=0.95,
            metadata={"page": 1}
        ),
        MagicMock(
            document_id=uuid4(),
            filename="result2.pdf",
            content="Neural networks are...",
            score=0.87,
            metadata={"page": 3}
        )
    ]

    with patch('app.api.rag.rag_service.search_multiple_pools', new=AsyncMock(return_value=mock_results)):
        response = await client.post(
            "/api/rag/search",
            headers=auth_headers,
            json={
                "query": "What is machine learning?",
                "knowledge_pool_ids": [pool1_id],
                "limit": 5
            }
        )

    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "What is machine learning?"
    assert result["num_results"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["content"] == "Machine learning is..."
    assert result["results"][0]["score"] == 0.95


@pytest.mark.asyncio
async def test_search_across_all_pools(client: AsyncClient, auth_headers: dict):
    """Test searching across all user's pools."""
    # Create multiple pools
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Pool 1", "description": "Test"}
        )
        await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Pool 2", "description": "Test"}
        )

    # Mock search results
    mock_results = [
        MagicMock(
            document_id=uuid4(),
            filename="doc.pdf",
            content="Result content",
            score=0.9,
            metadata={}
        )
    ]

    with patch('app.api.rag.rag_service.search_multiple_pools', new=AsyncMock(return_value=mock_results)):
        # Search without specifying pool IDs (searches all)
        response = await client.post(
            "/api/rag/search",
            headers=auth_headers,
            json={
                "query": "test query",
                "limit": 10
            }
        )

    assert response.status_code == 200
    result = response.json()
    assert result["num_results"] == 1


@pytest.mark.asyncio
async def test_search_with_no_pools(client: AsyncClient, auth_headers: dict):
    """Test searching when user has no pools returns empty results."""
    response = await client.post(
        "/api/rag/search",
        headers=auth_headers,
        json={
            "query": "test query",
            "limit": 5
        }
    )

    assert response.status_code == 200
    result = response.json()
    assert result["num_results"] == 0
    assert result["results"] == []


@pytest.mark.asyncio
async def test_search_nonexistent_pools(client: AsyncClient, auth_headers: dict):
    """Test searching non-existent pools returns 404."""
    fake_pool_id = str(uuid4())

    response = await client.post(
        "/api/rag/search",
        headers=auth_headers,
        json={
            "query": "test query",
            "knowledge_pool_ids": [fake_pool_id],
            "limit": 5
        }
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_validation(client: AsyncClient, auth_headers: dict):
    """Test search request validation."""
    # Empty query
    response = await client.post(
        "/api/rag/search",
        headers=auth_headers,
        json={
            "query": "",
            "limit": 5
        }
    )
    assert response.status_code == 422

    # Limit too high
    response = await client.post(
        "/api/rag/search",
        headers=auth_headers,
        json={
            "query": "test",
            "limit": 100
        }
    )
    assert response.status_code == 422

    # Limit too low
    response = await client.post(
        "/api/rag/search",
        headers=auth_headers,
        json={
            "query": "test",
            "limit": 0
        }
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_data_isolation(
    client: AsyncClient, auth_headers: dict, test_user_2: User
):
    """Test users can only search their own pools."""
    # User 1 creates pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "User 1 Pool", "description": "Private"}
        )
    pool_id = pool_response.json()["id"]

    # User 2 tries to search User 1's pool
    from app.auth import create_access_token
    user2_token = create_access_token(data={"sub": str(test_user_2.id)})
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    response = await client.post(
        "/api/rag/search",
        headers=user2_headers,
        json={
            "query": "test",
            "knowledge_pool_ids": [pool_id],
            "limit": 5
        }
    )

    assert response.status_code == 404  # Pool not found (doesn't belong to user 2)


# ============================================================================
# Background Processing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_document_background_processing(
    client: AsyncClient, auth_headers: dict, db_session
):
    """Test document processing updates status correctly."""
    # Create pool
    with patch('app.api.rag.rag_service.create_knowledge_pool', new=AsyncMock()):
        pool_response = await client.post(
            "/api/rag/knowledge-pools",
            headers=auth_headers,
            json={"name": "Processing Pool", "description": "Test"}
        )
    pool_id = pool_response.json()["id"]

    # Upload document (mocking background task)
    files = {"file": ("test.txt", io.BytesIO(b"test content"), "text/plain")}
    data = {"pool_id": pool_id}

    response = await client.post(
        "/api/rag/upload",
        headers=auth_headers,
        files=files,
        data=data
    )

    assert response.status_code == 200
    document_id = response.json()["document_id"]

    # Verify document starts as PENDING
    result = await db_session.execute(
        select(DBDocument).where(DBDocument.id == UUID(document_id))
    )
    doc = result.scalar_one()
    assert doc.status == DocumentStatus.PENDING
    assert doc.filename == "test.txt"


@pytest.mark.asyncio
async def test_document_processing_failure_handling(db_session):
    """Test that document processing handles errors correctly."""
    from app.api.rag import process_document_background

    # This test verifies the background task doesn't crash on errors
    # Note: The background task creates its own DB session, so we can't verify
    # the document status in the test session. This test just ensures no exceptions
    # are raised to the caller.

    # Create a test document in test database
    pool = KnowledgePool(
        user_id=uuid4(),
        name="Test Pool",
        description="Test",
        collection_name="test_collection"
    )
    db_session.add(pool)
    await db_session.commit()
    await db_session.refresh(pool)

    doc = DBDocument(
        knowledge_pool_id=pool.id,
        filename="test.txt",
        file_path="/nonexistent/path.txt",
        file_size=100,
        mime_type="text/plain",
        source_type="upload",
        status=DocumentStatus.PENDING
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    # Process document - should handle error gracefully (no exception raised)
    # The background task won't find the document in its own session, but that's OK
    # We're just testing it doesn't crash
    try:
        await process_document_background(
            document_id=doc.id,
            file_path="/nonexistent/path.txt",
            collection_name="test_collection"
        )
        # Success - no exception raised
        assert True
    except Exception as e:
        # Background task should not raise exceptions
        pytest.fail(f"Background task raised exception: {e}")


# ============================================================================
# Authorization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rag_endpoints_require_auth(client: AsyncClient):
    """Test that all RAG endpoints require authentication."""
    endpoints = [
        ("GET", "/api/rag/knowledge-pools"),
        ("POST", "/api/rag/knowledge-pools"),
        ("POST", "/api/rag/search"),
    ]

    for method, endpoint in endpoints:
        if method == "GET":
            response = await client.get(endpoint)
        elif method == "POST":
            response = await client.post(endpoint, json={})

        # FastAPI returns 403 Forbidden for missing/invalid JWT tokens
        assert response.status_code in [401, 403], f"{method} {endpoint} should require auth"
