"""
Adapters for different RAG implementations.

Wraps various retrieval strategies to conform to the RAGRetriever protocol.
"""

from typing import List, Dict, Any

from app.services.rag_service import RAGService
from app.evaluation.protocols import RAGRetriever, RetrievalResult


class BaselineRAGAdapter:
    """Adapter for baseline RAG service (current implementation)."""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.name = "Baseline (Semantic Search Only)"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve using baseline semantic search."""
        search_results = await self.rag_service.search(
            query=query,
            collection_name=collection_name,
            limit=limit,
        )

        # Convert to RetrievalResult
        return [
            RetrievalResult(
                document_id=str(r.document_id) if r.document_id else f"unknown_{i}",
                content=r.content,
                score=r.score,
                metadata=r.metadata or {},
                chunk_index=r.chunk_index,
            )
            for i, r in enumerate(search_results)
        ]

    async def get_name(self) -> str:
        return self.name


class HybridSearchAdapter:
    """
    Adapter for hybrid search (semantic + keyword).

    Placeholder for future hybrid search implementation.
    """

    def __init__(self, rag_service: RAGService, semantic_weight: float = 0.7):
        self.rag_service = rag_service
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.name = f"Hybrid Search (α={semantic_weight:.1f})"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid search.

        TODO: Implement actual hybrid search with BM25.
        For now, this is a placeholder that uses semantic search.
        """
        # Placeholder: Use semantic search for now
        # Future: Combine semantic + BM25 results
        search_results = await self.rag_service.search(
            query=query,
            collection_name=collection_name,
            limit=limit * 2,  # Retrieve more for reranking
        )

        # Convert and return top-k
        results = [
            RetrievalResult(
                document_id=str(r.document_id) if r.document_id else f"unknown_{i}",
                content=r.content,
                score=r.score,
                metadata=r.metadata or {},
                chunk_index=r.chunk_index,
            )
            for i, r in enumerate(search_results)
        ]

        return results[:limit]

    async def get_name(self) -> str:
        return self.name


class RerankedAdapter:
    """
    Adapter that adds reranking on top of another retriever.

    Retrieves more candidates, then reranks them with a cross-encoder.
    """

    def __init__(
        self,
        base_retriever: RAGRetriever,
        rerank_top_k: int = 20,
        final_k: int = 5,
    ):
        self.base_retriever = base_retriever
        self.rerank_top_k = rerank_top_k
        self.final_k = final_k
        self.name = f"Reranked ({base_retriever.__class__.__name__})"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve with reranking.

        TODO: Implement actual cross-encoder reranking.
        For now, this is a placeholder.
        """
        # Get more candidates than needed
        candidates = await self.base_retriever.retrieve(
            query=query,
            collection_name=collection_name,
            limit=self.rerank_top_k,
        )

        # Placeholder: Actual reranking would go here
        # Future: Use cross-encoder to rerank candidates
        # reranked = cross_encoder.rerank(query, candidates)

        # For now, just return top limit results
        return candidates[:limit]

    async def get_name(self) -> str:
        base_name = await self.base_retriever.get_name()
        return f"Reranked + {base_name}"


class ConfigurableAdapter:
    """
    Adapter that allows configuring different RAG parameters.

    Useful for testing different configurations without code changes.
    """

    def __init__(
        self,
        rag_service: RAGService,
        config: Dict[str, Any],
        name: str = None,
    ):
        """
        Initialize configurable adapter.

        Args:
            rag_service: Base RAG service
            config: Configuration dict with parameters
            name: Optional name for this configuration
        """
        self.rag_service = rag_service
        self.config = config
        self._name = name or f"Custom ({', '.join(f'{k}={v}' for k, v in config.items())})"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve with custom configuration."""
        # Extract config parameters
        score_threshold = self.config.get("score_threshold", None)
        limit_override = self.config.get("limit", limit)

        search_results = await self.rag_service.search(
            query=query,
            collection_name=collection_name,
            limit=limit_override,
            score_threshold=score_threshold,
        )

        results = [
            RetrievalResult(
                document_id=str(r.document_id) if r.document_id else f"unknown_{i}",
                content=r.content,
                score=r.score,
                metadata=r.metadata or {},
                chunk_index=r.chunk_index,
            )
            for i, r in enumerate(search_results)
        ]

        return results[:limit]

    async def get_name(self) -> str:
        return self._name
