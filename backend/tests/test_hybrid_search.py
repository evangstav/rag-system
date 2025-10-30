"""
Tests for hybrid search functionality (BM25 + semantic with RRF).

Tests cover:
- BM25 indexing and search
- Reciprocal Rank Fusion
- Hybrid search integration with RAGService
- Configuration validation
"""

import pytest
from typing import List, Dict, Any
from pydantic import ValidationError

from app.services.rag.protocols import DocumentChunk, SearchResult
from app.services.rag.bm25_index import BM25Index
from app.services.rag.hybrid_search import HybridSearchService


# Fixtures


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


@pytest.fixture
async def bm25_index(sample_chunks) -> BM25Index:
    """Create and populate BM25 index."""
    index = BM25Index()
    await index.index_documents("test_collection", sample_chunks)
    return index


# BM25Index Tests


class TestBM25Index:
    """Tests for BM25 keyword indexing."""

    @pytest.mark.asyncio
    async def test_index_documents(self, sample_chunks):
        """Test indexing documents in BM25."""
        index = BM25Index()

        # Index documents
        await index.index_documents("test_collection", sample_chunks)

        # Verify index was created
        assert index.has_collection("test_collection")
        assert index.get_collection_size("test_collection") == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_search_exact_match(self, bm25_index):
        """Test BM25 search with exact keyword match."""
        results = await bm25_index.search(
            collection_name="test_collection",
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
    async def test_search_keyword_relevance(self, bm25_index):
        """Test that BM25 ranks by keyword relevance."""
        results = await bm25_index.search(
            collection_name="test_collection",
            query="JavaScript TypeScript",
            limit=5,
        )

        # Should find both JavaScript documents
        assert len(results) >= 2

        # Results should contain JavaScript or TypeScript
        for doc, score in results:
            content = doc["content"]
            assert "JavaScript" in content or "TypeScript" in content

    @pytest.mark.asyncio
    async def test_search_no_results(self, bm25_index):
        """Test search with no matching keywords."""
        results = await bm25_index.search(
            collection_name="test_collection",
            query="quantum computing blockchain",
            limit=5,
        )

        # Should return empty (none of these terms in sample data)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_append_documents(self, bm25_index, sample_chunks):
        """Test appending new documents to existing index."""
        initial_size = bm25_index.get_collection_size("test_collection")

        # Create new chunks
        new_chunks = [
            DocumentChunk(
                content="Rust is a systems programming language",
                metadata={"document_id": "doc4", "source": "rust.pdf"},
                chunk_index=0,
            ),
        ]

        # Append to index
        await bm25_index.append_documents("test_collection", new_chunks)

        # Verify size increased
        assert bm25_index.get_collection_size("test_collection") == initial_size + 1

        # Verify new document is searchable
        results = await bm25_index.search(
            collection_name="test_collection",
            query="Rust systems",
            limit=5,
        )
        assert len(results) > 0
        assert "Rust" in results[0][0]["content"]

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, bm25_index):
        """Test deleting documents by document ID."""
        # Delete all chunks from doc1
        deleted_count = await bm25_index.delete_by_document_id(
            collection_name="test_collection",
            document_id="doc1",
        )

        assert deleted_count == 2  # doc1 has 2 chunks

        # Verify chunks are gone
        results = await bm25_index.search(
            collection_name="test_collection",
            query="Python JavaScript",
            limit=10,
        )

        # Should not find doc1 chunks anymore
        for doc, score in results:
            assert doc["document_id"] != "doc1"

    @pytest.mark.asyncio
    async def test_delete_collection(self, bm25_index):
        """Test deleting entire collection."""
        await bm25_index.delete_collection("test_collection")

        assert not bm25_index.has_collection("test_collection")
        assert bm25_index.get_collection_size("test_collection") == 0

    @pytest.mark.asyncio
    async def test_tokenization(self):
        """Test BM25 tokenization."""
        index = BM25Index()

        # Test various text inputs
        tokens = index._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "," not in tokens  # Punctuation removed

        # Test numbers
        tokens = index._tokenize("Python 3.9 version 2024")
        assert "python" in tokens
        assert "version" in tokens

        # Test minimum length filtering (single chars removed)
        tokens = index._tokenize("a b cd efg")
        assert "a" not in tokens  # Too short
        assert "b" not in tokens  # Too short
        assert "cd" in tokens
        assert "efg" in tokens


# HybridSearchService Tests


class TestHybridSearchService:
    """Tests for hybrid search combining semantic + BM25."""

    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self):
        """Test RRF score calculation."""
        # Mock semantic results
        semantic_results = [
            SearchResult(
                content="Result A",
                score=0.95,
                metadata={},
                document_id="doc1",
                chunk_index=0,
            ),
            SearchResult(
                content="Result B",
                score=0.85,
                metadata={},
                document_id="doc2",
                chunk_index=0,
            ),
        ]

        # Mock keyword results (metadata, bm25_score)
        keyword_results = [
            (
                {
                    "content": "Result B",
                    "document_id": "doc2",
                    "chunk_index": 0,
                },
                12.5,
            ),
            (
                {
                    "content": "Result C",
                    "document_id": "doc3",
                    "chunk_index": 0,
                },
                10.0,
            ),
        ]

        # Create hybrid search service (with mocks)
        class MockEmbeddingProvider:
            dimensions = 1536
            model_name = "test-model"

            async def embed_text(self, text):
                return [0.1] * self.dimensions

            async def embed_batch(self, texts):
                return [[0.1] * self.dimensions] * len(texts)

        class MockVectorStore:
            async def search(self, **kwargs):
                return semantic_results

        hybrid_service = HybridSearchService(
            embedding_provider=MockEmbeddingProvider(),
            vector_store=MockVectorStore(),
        )

        # Test RRF fusion
        fused = hybrid_service._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=0.5,
            keyword_weight=0.5,
            rrf_k=60,
        )

        # Result B should be ranked highest (appears in both)
        assert fused[0].document_id == "doc2"

        # All unique results should be included
        assert len(fused) == 3

        # Scores should be RRF scores (not original scores)
        assert all(0 < result.score < 1 for result in fused)

    @pytest.mark.asyncio
    async def test_rrf_weighting(self):
        """Test that RRF weights affect ranking."""
        semantic_results = [
            SearchResult(
                content="Semantic winner",
                score=0.99,
                metadata={},
                document_id="doc1",
                chunk_index=0,
            ),
        ]

        keyword_results = [
            (
                {
                    "content": "Keyword winner",
                    "document_id": "doc2",
                    "chunk_index": 0,
                },
                100.0,
            ),
        ]

        class MockEmbeddingProvider:
            dimensions = 1536

            async def embed_text(self, text):
                return [0.1] * self.dimensions

        class MockVectorStore:
            async def search(self, **kwargs):
                return semantic_results

        hybrid_service = HybridSearchService(
            embedding_provider=MockEmbeddingProvider(),
            vector_store=MockVectorStore(),
        )

        # Test with semantic weight higher
        fused_semantic = hybrid_service._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=0.9,
            keyword_weight=0.1,
            rrf_k=60,
        )

        # Test with keyword weight higher
        fused_keyword = hybrid_service._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=0.1,
            keyword_weight=0.9,
            rrf_k=60,
        )

        # Semantic doc should score higher with semantic weight
        semantic_score = next(
            r.score for r in fused_semantic if r.document_id == "doc1"
        )
        keyword_biased_semantic_score = next(
            r.score for r in fused_keyword if r.document_id == "doc1"
        )

        assert semantic_score > keyword_biased_semantic_score


# Integration Tests


class TestHybridSearchIntegration:
    """Integration tests for hybrid search in RAGService."""

    @pytest.mark.asyncio
    async def test_hybrid_search_disabled_by_default(self):
        """Test that hybrid search is disabled by default in config."""
        from app.config import settings

        # Default should be False (semantic only)
        # This ensures backward compatibility
        assert settings.enable_hybrid_search is False

    @pytest.mark.asyncio
    async def test_rag_service_hybrid_initialization(self):
        """Test RAGService initializes hybrid search when enabled."""
        from app.services.rag_service import RAGService

        # Test with hybrid search disabled
        rag_service_disabled = RAGService(enable_hybrid_search=False)
        assert rag_service_disabled.enable_hybrid_search is False
        assert rag_service_disabled.bm25_index is None
        assert rag_service_disabled.hybrid_search is None

        # Test with hybrid search enabled
        rag_service_enabled = RAGService(enable_hybrid_search=True)
        assert rag_service_enabled.enable_hybrid_search is True
        assert rag_service_enabled.bm25_index is not None
        assert rag_service_enabled.hybrid_search is not None


# Edge Cases


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_collection_search(self):
        """Test searching empty BM25 index."""
        index = BM25Index()

        results = await index.search(
            collection_name="nonexistent_collection",
            query="test query",
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query(self, bm25_index):
        """Test search with empty query."""
        results = await bm25_index.search(
            collection_name="test_collection",
            query="",
            limit=5,
        )

        # Empty query should return empty results
        assert results == []

    @pytest.mark.asyncio
    async def test_index_empty_documents(self):
        """Test indexing empty document list."""
        index = BM25Index()

        await index.index_documents("test_collection", [])

        # Should handle gracefully
        assert not index.has_collection("test_collection")

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, bm25_index):
        """Test query with special characters."""
        results = await bm25_index.search(
            collection_name="test_collection",
            query="Python!@#$%^&*() programming",
            limit=5,
        )

        # Should still find Python-related documents
        assert len(results) > 0


# Configuration Validation Tests


class TestConfigurationValidation:
    """Test configuration validation for hybrid search weights."""

    def test_valid_weights(self):
        """Test that valid weights are accepted."""
        from app.config import Settings

        # Valid: both weights between 0 and 1
        settings = Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
            HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
        )
        assert settings.hybrid_search_semantic_weight == 0.5
        assert settings.hybrid_search_keyword_weight == 0.5

    def test_invalid_semantic_weight_negative(self):
        """Test that negative semantic weight is rejected."""
        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                OPENAI_API_KEY="test-key",
                HYBRID_SEARCH_SEMANTIC_WEIGHT=-0.1,
                HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
            )

        assert "hybrid_search_semantic_weight" in str(exc_info.value)
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_invalid_semantic_weight_too_high(self):
        """Test that semantic weight > 1 is rejected."""
        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                OPENAI_API_KEY="test-key",
                HYBRID_SEARCH_SEMANTIC_WEIGHT=1.5,
                HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
            )

        assert "hybrid_search_semantic_weight" in str(exc_info.value)
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_invalid_keyword_weight_negative(self):
        """Test that negative keyword weight is rejected."""
        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                OPENAI_API_KEY="test-key",
                HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
                HYBRID_SEARCH_KEYWORD_WEIGHT=-0.2,
            )

        assert "hybrid_search_keyword_weight" in str(exc_info.value)
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_invalid_keyword_weight_too_high(self):
        """Test that keyword weight > 1 is rejected."""
        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                OPENAI_API_KEY="test-key",
                HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
                HYBRID_SEARCH_KEYWORD_WEIGHT=2.0,
            )

        assert "hybrid_search_keyword_weight" in str(exc_info.value)
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_boundary_values(self):
        """Test that boundary values (0 and 1) are accepted."""
        from app.config import Settings

        # All semantic (keyword = 0)
        settings = Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=1.0,
            HYBRID_SEARCH_KEYWORD_WEIGHT=0.0,
        )
        assert settings.hybrid_search_semantic_weight == 1.0
        assert settings.hybrid_search_keyword_weight == 0.0

        # All keyword (semantic = 0)
        settings = Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=0.0,
            HYBRID_SEARCH_KEYWORD_WEIGHT=1.0,
        )
        assert settings.hybrid_search_semantic_weight == 0.0
        assert settings.hybrid_search_keyword_weight == 1.0

    def test_weights_sum_warning(self):
        """Test that warning is issued when weights don't sum to 1.0."""
        from app.config import Settings
        import warnings

        # Weights that don't sum to 1.0 should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            settings = Settings(
                OPENAI_API_KEY="test-key",
                HYBRID_SEARCH_SEMANTIC_WEIGHT=0.3,
                HYBRID_SEARCH_KEYWORD_WEIGHT=0.3,  # Sum = 0.6
            )

            # Check that a warning was issued
            assert len(w) == 1
            assert "sum to" in str(w[0].message)
            assert "0.60" in str(w[0].message)
