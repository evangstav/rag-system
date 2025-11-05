"""
Pytest configuration and shared fixtures for backend tests.

Provides:
- Test database setup and teardown
- Test HTTP client
- Authentication helpers
- Mock external services (OpenAI, Qdrant)
- Sample test data
"""

import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import StaticPool
from faker import Faker

from app.main import app
from app.models.database import Base, User
from app.database import get_session
from app.dependencies import get_db
from app.config import settings
from app.auth import create_access_token, get_password_hash

# Initialize Faker for generating test data
fake = Faker()

# =============================================================================
# Environment Configuration
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["TESTING"] = "1"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    # Disable observability in tests
    os.environ["OBSERVABILITY_ENABLED"] = "false"
    os.environ["LANGFUSE_ENABLED"] = "false"


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a test database engine using in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Share same connection across requests
        echo=False,  # Set to True for SQL debugging
    )

    # Create tables (replacing PostgreSQL types with SQLite-compatible ones)
    async with engine.begin() as conn:
        def create_tables_sync(connection):
            from sqlalchemy import JSON, String
            from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

            # Replace PostgreSQL-specific types with SQLite-compatible ones
            for table in Base.metadata.sorted_tables:
                # Skip PostgreSQL-specific tables
                if table.name in ['bm25_documents', 'bm25_stats']:
                    continue

                # Replace JSONB columns with JSON
                for column in table.columns:
                    if isinstance(column.type, JSONB):
                        column.type = JSON()
                    elif isinstance(column.type, type(TSVECTOR)):
                        column.type = String()

                # Create table
                table.create(connection, checkfirst=True)

        await conn.run_sync(create_tables_sync)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Create session factory
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()  # Rollback any uncommitted changes


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Create a test HTTP client with overridden database dependency.

    All API requests made with this client will use the test database.
    """
    # Override database dependency to use test session
    async def override_get_db():
        yield db_session

    # Override the get_db dependency that's actually used by the API
    app.dependency_overrides[get_db] = override_get_db

    # Create async HTTP client
    transport = ASGITransport(app=app)  # type: ignore
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver"
    ) as ac:
        yield ac

    # Clear overrides after test
    app.dependency_overrides.clear()


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user in the database."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword123"),
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture(scope="function")
async def auth_headers(test_user: User) -> Dict[str, str]:
    """Get authentication headers with valid JWT token for test user."""
    access_token = create_access_token(data={"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {access_token}"}


@pytest_asyncio.fixture(scope="function")
async def another_user(db_session: AsyncSession) -> User:
    """Create a second test user for testing data isolation."""
    user = User(
        email="another@example.com",
        username="anotheruser",
        hashed_password=get_password_hash("anotherpassword123"),
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# =============================================================================
# Mock External Services
# =============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for chat completions."""
    class MockChoice:
        def __init__(self, content: str):
            self.message = type('obj', (object,), {'content': content})()
            self.finish_reason = "stop"

    class MockCompletion:
        def __init__(self, content: str):
            self.choices = [MockChoice(content)]
            self.id = fake.uuid4()
            self.model = "gpt-4"
            self.usage = type('obj', (object,), {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            })()

    return lambda content="Test response": MockCompletion(content)


@pytest.fixture
def mock_openai_stream():
    """Mock OpenAI streaming response."""
    async def stream_generator(content: str = "Test streaming response"):
        words = content.split()
        for word in words:
            chunk = type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'delta': type('obj', (object,), {'content': word + " "})(),
                    'finish_reason': None
                })()]
            })()
            yield chunk
        # Final chunk
        final_chunk = type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'delta': type('obj', (object,), {})(),
                'finish_reason': 'stop'
            })()]
        })()
        yield final_chunk

    return stream_generator


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings API response."""
    def generate_embedding(text: str, dimension: int = 1536):
        # Generate deterministic fake embedding based on text
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Simple pseudo-random embedding
        embedding = [(hash_val >> (i % 32)) % 100 / 100.0 for i in range(dimension)]
        return embedding

    class MockEmbedding:
        def __init__(self, text: str):
            self.embedding = generate_embedding(text)

    class MockEmbeddingResponse:
        def __init__(self, texts: list[str]):
            self.data = [MockEmbedding(text) for text in texts]
            self.usage = type('obj', (object,), {'total_tokens': sum(len(t.split()) for t in texts)})()

    return lambda texts: MockEmbeddingResponse(texts if isinstance(texts, list) else [texts])


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector operations."""
    class MockQdrantClient:
        def __init__(self):
            self.collections = {}
            self.points = {}

        async def create_collection(self, collection_name: str, **kwargs):
            self.collections[collection_name] = {
                "name": collection_name,
                "config": kwargs
            }
            self.points[collection_name] = []
            return True

        async def upsert(self, collection_name: str, points: list):
            if collection_name not in self.points:
                self.points[collection_name] = []
            self.points[collection_name].extend(points)
            return {"status": "ok"}

        async def search(self, collection_name: str, query_vector: list, limit: int = 10, **kwargs):
            # Return mock search results
            results = []
            for i, point in enumerate(self.points.get(collection_name, [])[:limit]):
                results.append({
                    "id": point.get("id", i),
                    "score": 0.9 - (i * 0.1),  # Decreasing scores
                    "payload": point.get("payload", {})
                })
            return results

        async def delete(self, collection_name: str, points_selector: dict):
            if collection_name in self.points:
                # Simplified deletion
                self.points[collection_name] = []
            return {"status": "ok"}

        async def get_collection(self, collection_name: str):
            return self.collections.get(collection_name)

    return MockQdrantClient()


# =============================================================================
# Test Data Factories
# =============================================================================

@pytest.fixture
def sample_document_content():
    """Generate sample document content for testing."""
    return """
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Types of Learning

1. **Supervised Learning**: Uses labeled data for training
2. **Unsupervised Learning**: Finds patterns in unlabeled data
3. **Reinforcement Learning**: Learns through trial and error

## Applications

Machine learning is used in various domains including:
- Natural Language Processing
- Computer Vision
- Recommendation Systems
- Fraud Detection

## Key Algorithms

Common algorithms include linear regression, decision trees, neural networks, and support vector machines.
"""


@pytest.fixture
def sample_pdf_upload():
    """Generate a mock PDF file upload for testing."""
    import io

    # Create a minimal PDF-like bytes object
    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>%%EOF"

    return {
        "file": ("test_document.pdf", io.BytesIO(pdf_content), "application/pdf")
    }


@pytest.fixture
def sample_chat_messages():
    """Generate sample chat messages for testing."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Tell me more about supervised learning."},
    ]


@pytest.fixture
def sample_scratchpad_data():
    """Generate sample scratchpad data for testing."""
    return {
        "todos": [
            {"content": "Review ML paper", "completed": False, "priority": "high"},
            {"content": "Implement RAG system", "completed": False, "priority": "medium"},
        ],
        "notes": [
            {"content": "Important: Focus on semantic search quality", "tags": ["rag", "search"]},
        ],
        "journal_entry": {
            "content": "Today I learned about hybrid search combining BM25 and semantic search."
        }
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def assert_valid_uuid():
    """Helper to assert that a string is a valid UUID."""
    from uuid import UUID
    def validator(value: str):
        try:
            UUID(value)
            return True
        except ValueError:
            return False
    return validator


@pytest.fixture
def assert_valid_iso_datetime():
    """Helper to assert that a string is a valid ISO datetime."""
    from datetime import datetime
    def validator(value: str):
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False
    return validator


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_after_test():
    """Auto-cleanup after each test."""
    yield
    # Any cleanup logic here
    # For example, clear caches, reset singletons, etc.
