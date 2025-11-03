"""
Reranking implementations for improving RAG retrieval quality.

Cross-encoder rerankers jointly process [query, document] pairs to provide
more accurate relevance scores than bi-encoder vector similarity alone.
"""

from typing import List, Protocol
from dataclasses import dataclass
import asyncio

from sentence_transformers import CrossEncoder

from app.services.rag.protocols import SearchResult
from app.config import Settings


class Reranker(Protocol):
    """Protocol for reranking implementations."""

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Rerank search results based on query relevance."""
        ...


@dataclass
class RerankResult:
    """Result from reranking with detailed metadata."""

    original_result: SearchResult
    rerank_score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Significantly more accurate than pure vector similarity because it
    processes [query, document] pairs jointly instead of separately.

    Model options:
    - mixedbread-ai/mxbai-rerank-large-v1 (recommended, NDCG@10: 0.869)
    - BAAI/bge-reranker-v2-m3 (multilingual, NDCG@10: 0.851)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, NDCG@10: 0.77)

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> results = await reranker.rerank(
        ...     query="How to reset password?",
        ...     results=vector_search_results,
        ...     top_k=5
        ... )
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        """
        Initialize reranker with specified model.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self._model = None  # Lazy load to avoid blocking startup

    @property
    def model(self) -> CrossEncoder:
        """Lazy load model on first use."""
        if self._model is None:
            # Model download happens here (first call only, ~1-2GB)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: User query
            results: Initial search results from vector search
            top_k: Number of results to return after reranking

        Returns:
            Reranked results with updated scores (sorted by reranker score)

        Example:
            >>> results = await reranker.rerank(
            ...     query="authentication issues",
            ...     results=initial_results,
            ...     top_k=10
            ... )
            >>> print(results[0].score)  # Reranker score (0-1)
            >>> print(results[0].metadata["original_score"])  # Vector similarity
        """
        if not results:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, result.content) for result in results]

        # Run cross-encoder in thread pool (CPU-bound operation)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.model.predict, pairs)

        # Create reranked results with new scores
        reranked = []
        for result, score in zip(results, scores):
            # Create new SearchResult with updated score
            reranked_result = SearchResult(
                content=result.content,
                score=float(score),  # Replace vector similarity with reranker score
                metadata={
                    **result.metadata,
                    "original_score": result.score,  # Preserve original for debugging
                    "rerank_score": float(score),
                },
                document_id=result.document_id,
                chunk_index=result.chunk_index,
            )
            reranked.append(reranked_result)

        # Sort by reranker score (descending)
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked[:top_k]

    async def rerank_with_metadata(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank with detailed metadata about rank changes.

        Useful for evaluation and debugging to see how rankings change.

        Args:
            query: User query
            results: Initial search results
            top_k: Number of results to return

        Returns:
            List of RerankResult objects with rank metadata
        """
        if not results:
            return []

        pairs = [(query, result.content) for result in results]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.model.predict, pairs)

        # Create results with rank metadata
        rerank_results = [
            RerankResult(
                original_result=result,
                rerank_score=float(score),
                original_rank=idx,
                new_rank=-1,  # Set after sorting
            )
            for idx, (result, score) in enumerate(zip(results, scores))
        ]

        # Sort by reranker score
        rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update new ranks
        for idx, result in enumerate(rerank_results[:top_k]):
            result.new_rank = idx

        return rerank_results[:top_k]


class CohereReranker:
    """
    Cohere Rerank API v3 (highest accuracy but paid).

    Advantages:
    - Best-in-class NDCG@10: 0.875
    - No model hosting required
    - Multilingual support

    Cost: $1/1000 searches
    Requires: COHERE_API_KEY environment variable

    Example:
        >>> reranker = CohereReranker(api_key=settings.cohere_api_key)
        >>> results = await reranker.rerank(query, results, top_k=10)
    """

    def __init__(self, api_key: str):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key

        Raises:
            ImportError: If cohere package not installed
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere SDK not installed. Install with: pip install cohere"
            )

        self.client = cohere.Client(api_key)

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Rerank using Cohere API."""
        if not results:
            return []

        # Prepare documents
        documents = [result.content for result in results]

        # Call Cohere Rerank API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0",
            ),
        )

        # Map back to SearchResult objects
        reranked = []
        for item in response.results:
            original_result = results[item.index]
            reranked_result = SearchResult(
                content=original_result.content,
                score=item.relevance_score,
                metadata={
                    **original_result.metadata,
                    "original_score": original_result.score,
                    "rerank_score": item.relevance_score,
                },
                document_id=original_result.document_id,
                chunk_index=original_result.chunk_index,
            )
            reranked.append(reranked_result)

        return reranked


class FlashRankReranker:
    """
    Lightweight approximate reranker (10-20ms latency).

    Good for high-volume, cost-sensitive applications.
    NDCG@10: ~0.75-0.80 (15% lower than full cross-encoders)

    Trade-off: Faster but less accurate than full cross-encoders.

    Example:
        >>> reranker = FlashRankReranker()
        >>> results = await reranker.rerank(query, results, top_k=10)
    """

    def __init__(self):
        """
        Initialize FlashRank reranker.

        Raises:
            ImportError: If flashrank package not installed
        """
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError(
                "FlashRank not installed. Install with: pip install flashrank"
            )

        self.ranker = Ranker()

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Fast approximate reranking."""
        if not results:
            return []

        # Prepare passages
        passages = [
            {"id": idx, "text": result.content} for idx, result in enumerate(results)
        ]

        # Rerank
        loop = asyncio.get_event_loop()
        reranked_passages = await loop.run_in_executor(
            None, lambda: self.ranker.rerank(query, passages)
        )

        # Map back to SearchResult
        reranked = []
        for passage in reranked_passages[:top_k]:
            original_result = results[passage["id"]]
            reranked_result = SearchResult(
                content=original_result.content,
                score=passage["score"],
                metadata={
                    **original_result.metadata,
                    "original_score": original_result.score,
                    "rerank_score": passage["score"],
                },
                document_id=original_result.document_id,
                chunk_index=original_result.chunk_index,
            )
            reranked.append(reranked_result)

        return reranked


def get_reranker(settings: Settings) -> Reranker:
    """
    Factory function for reranker.

    Selects appropriate reranker based on settings.

    Args:
        settings: Application settings

    Returns:
        Configured reranker instance

    Example:
        >>> from app.config import settings
        >>> reranker = get_reranker(settings)
        >>> results = await reranker.rerank(query, results, top_k=10)

    Note:
        This function creates a new reranker instance on each call.
        The CrossEncoderReranker uses lazy loading, so the model is only
        loaded on first use, not on instantiation.
    """
    model_name = settings.reranker_model.lower()

    if "cohere" in model_name:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY required for Cohere reranker")
        return CohereReranker(settings.cohere_api_key)
    elif "flashrank" in model_name:
        return FlashRankReranker()
    else:
        # Default: Cross-encoder (lazy loaded on first use)
        return CrossEncoderReranker(settings.reranker_model)
