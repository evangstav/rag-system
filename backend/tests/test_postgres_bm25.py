"""
Tests for PostgreSQL-backed BM25 implementation.

Tests cover:
- PostgreSQL BM25 indexing and search
- Full-text search functionality
- Integration with RAGService
- Database persistence
"""

import pytest
import pytest_asyncio
import uuid
from typing import List
from sqlalchemy import select, func, text

from app.models.database import BM25Document
from app.services.rag.protocols import DocumentChunk
from app.services.rag.postgres_bm25 import PostgresBM25Index
from app.database import AsyncSessionLocal


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            content="Python is a high-level programming language",
            metadata={"document_id": "doc1", "source": "test.pdf"},
            chunk_index=0,
        ),
        DocumentChunk(
            content="JavaScript is widely used for web development",
            metadata={"document_id": "doc1", "source": "test.pdf"},
            chunk_index=1,
        ),
        DocumentChunk(
            content="Machine learning uses Python extensively",
            metadata={"document_id": "doc2", "source": "ml.pdf"},
            chunk_index=0,
        ),
        DocumentChunk(
            content="TypeScript is a superset of JavaScript",
            metadata={"document_id": "doc2", "source": "ml.pdf"},
            chunk_index=1,
        ),
        DocumentChunk(
            content="Neural networks are fundamental to deep learning",
            metadata={"document_id": "doc3", "source": "ai.pdf"},
            chunk_index=0,
        ),
    ]


@pytest_asyncio.fixture
async def db_session():
    """Create a test database session with cleanup."""
    async with AsyncSessionLocal() as session:
        yield session
        # Cleanup: delete all test data
        try:
            await session.execute(text("DELETE FROM bm25_documents"))
            await session.commit()
        except Exception:
            await session.rollback()


@pytest_asyncio.fixture
async def collection_name():
    """Generate a unique collection name for each test."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def postgres_bm25(db_session, sample_chunks, collection_name):
    """Create and populate PostgreSQL BM25 index with unique collection."""
    index = PostgresBM25Index(db_session)
    await index.index_documents(collection_name, sample_chunks)
    yield index
    # Cleanup this specific collection
    try:
        await index.delete_collection(collection_name)
    except Exception:
        pass


class TestPostgresBM25Index:
    """Tests for PostgreSQL BM25 indexing."""

    @pytest.mark.asyncio
    async def test_index_documents(self, db_session, sample_chunks, collection_name):
        """Test indexing documents in PostgreSQL."""
        index = PostgresBM25Index(db_session)

        # Index documents
        await index.index_documents(collection_name, sample_chunks)

        # Verify documents were inserted
        result = await db_session.execute(
            select(func.count(BM25Document.id)).where(
                BM25Document.collection_name == collection_name
            )
        )
        count = result.scalar_one()
        assert count == len(sample_chunks)

        # Cleanup
        await index.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_tsvector_trigger(self, db_session, sample_chunks, collection_name):
        """Test that tsvector is automatically populated by trigger."""
        index = PostgresBM25Index(db_session)
        await index.index_documents(collection_name, sample_chunks)

        # Query to check tsvector was created
        result = await db_session.execute(
            text("""
                SELECT content_tsv IS NOT NULL as has_tsv
                FROM bm25_documents
                WHERE collection_name = :collection_name
                LIMIT 1
            """),
            {"collection_name": collection_name}
        )
        row = result.fetchone()
        assert row is not None
        assert row.has_tsv is True

        # Cleanup
        await index.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_search_exact_match(self, postgres_bm25, collection_name):
        """Test BM25 search with exact keyword match."""
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="Python programming",
            limit=5,
        )

        # Should find documents mentioning Python
        assert len(results) > 0

        # First result should be the Python document
        top_result, score = results[0]
        assert "Python" in top_result["content"]
        assert score > 0

    @pytest.mark.asyncio
    async def test_search_keyword_relevance(self, postgres_bm25, collection_name):
        """Test that BM25 ranks by keyword relevance."""
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="JavaScript TypeScript",
            limit=5,
        )

        # Should find both JavaScript documents
        assert len(results) >= 2

        # Results should contain JavaScript or TypeScript
        for doc, _ in results:
            content = doc["content"]
            assert "JavaScript" in content or "TypeScript" in content

    @pytest.mark.asyncio
    async def test_search_no_results(self, postgres_bm25, collection_name):
        """Test search with no matching keywords."""
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="quantum computing blockchain",
            limit=5,
        )

        # Should return empty (none of these terms in sample data)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_append_documents(self, postgres_bm25, collection_name):
        """Test appending new documents to existing index."""
        initial_count = await postgres_bm25.get_collection_size(collection_name)

        # Create new chunks
        new_chunks = [
            DocumentChunk(
                content="Rust is a systems programming language",
                metadata={"document_id": "doc4", "source": "rust.pdf"},
                chunk_index=0,
            ),
        ]

        # Append to index
        await postgres_bm25.append_documents(collection_name, new_chunks)

        # Verify size increased
        new_count = await postgres_bm25.get_collection_size(collection_name)
        assert new_count == initial_count + 1

        # Verify new document is searchable
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="Rust systems",
            limit=5,
        )
        assert len(results) > 0
        assert "Rust" in results[0][0]["content"]

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, postgres_bm25, collection_name):
        """Test deleting documents by document ID."""
        # Delete all chunks from doc1
        deleted_count = await postgres_bm25.delete_by_document_id(
            collection_name=collection_name,
            document_id="doc1",
        )

        assert deleted_count == 2  # doc1 has 2 chunks

        # Verify chunks are gone
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="Python JavaScript",
            limit=10,
        )

        # Should not find doc1 chunks anymore
        for doc, _ in results:
            assert doc["document_id"] != "doc1"

    @pytest.mark.asyncio
    async def test_delete_collection(self, postgres_bm25, collection_name, db_session):
        """Test deleting entire collection."""
        await postgres_bm25.delete_collection(collection_name)

        # Verify collection is empty
        assert not await postgres_bm25.has_collection(collection_name)
        assert await postgres_bm25.get_collection_size(collection_name) == 0

        # Verify database is empty for this collection
        result = await db_session.execute(
            select(func.count(BM25Document.id)).where(
                BM25Document.collection_name == collection_name
            )
        )
        assert result.scalar_one() == 0

    @pytest.mark.asyncio
    async def test_phrase_search(self, postgres_bm25, collection_name):
        """Test phrase search with PostgreSQL FTS."""
        # Phrase search should match exact phrases
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query='"programming language"',
            limit=5,
        )

        # Should find documents with this exact phrase
        assert len(results) > 0
        assert "programming language" in results[0][0]["content"].lower()

    @pytest.mark.asyncio
    async def test_empty_query(self, postgres_bm25, collection_name):
        """Test search with empty query."""
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="",
            limit=5,
        )

        # Empty query should return empty results
        assert results == []

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, postgres_bm25, collection_name):
        """Test query with special characters."""
        results = await postgres_bm25.search(
            collection_name=collection_name,
            query="Python!@#$%^&*() programming",
            limit=5,
        )

        # Should still find Python-related documents
        # PostgreSQL FTS handles special characters gracefully
        assert len(results) > 0


class TestPostgresBM25Stats:
    """Tests for BM25 statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_stats(self, postgres_bm25, collection_name):
        """Test getting collection statistics."""
        stats = await postgres_bm25.get_stats(collection_name)

        assert stats["collection_name"] == collection_name
        assert stats["document_count"] == 5
        assert stats["avg_content_length"] > 0
        assert stats["backend"] == "postgresql_fts"
        assert "table_size" in stats

    @pytest.mark.asyncio
    async def test_has_collection(self, postgres_bm25, collection_name):
        """Test checking if collection exists."""
        # Existing collection
        assert await postgres_bm25.has_collection(collection_name)

        # Non-existing collection
        assert not await postgres_bm25.has_collection("nonexistent_collection_xyz")

    @pytest.mark.asyncio
    async def test_get_collection_size(self, postgres_bm25, collection_name):
        """Test getting collection size."""
        size = await postgres_bm25.get_collection_size(collection_name)
        assert size == 5

        # Non-existing collection
        size = await postgres_bm25.get_collection_size("nonexistent_xyz")
        assert size == 0


class TestPostgresBM25Persistence:
    """Tests for PostgreSQL BM25 persistence."""

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, sample_chunks):
        """Test that indexed documents persist across sessions."""
        collection_name = f"persist_{uuid.uuid4().hex[:8]}"

        # Index in first session
        async with AsyncSessionLocal() as session1:
            index1 = PostgresBM25Index(session1)
            await index1.index_documents(collection_name, sample_chunks)

        # Create new session and index
        async with AsyncSessionLocal() as session2:
            index2 = PostgresBM25Index(session2)

            # Verify documents are still there
            assert await index2.has_collection(collection_name)
            assert await index2.get_collection_size(collection_name) == 5

            # Verify search works
            results = await index2.search(collection_name, "Python", limit=5)
            assert len(results) > 0

            # Cleanup
            await index2.delete_collection(collection_name)
