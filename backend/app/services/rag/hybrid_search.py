"""
Hybrid search combining semantic vector search with BM25 keyword search.

Combines the strengths of both approaches:
- Semantic search: Understands meaning and context
- BM25 keyword search: Captures exact term matches

Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from app.services.rag.protocols import SearchResult, EmbeddingProvider, VectorStore
from app.services.rag.bm25_index import BM25Index
from app.config import settings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid search service combining semantic and keyword-based retrieval.

    Architecture:
        Query → [Semantic Search] + [BM25 Search] → RRF Fusion → Top Results

    Benefits:
    - Better recall: Finds documents through multiple pathways
    - Exact match: BM25 catches specific terms, IDs, acronyms
    - Semantic understanding: Embeddings handle synonyms and context

    Research shows 15-25% improvement in recall vs semantic-only search.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
    ):
        """
        Initialize hybrid search service.

        Args:
            embedding_provider: Embedding provider for semantic search
            vector_store: Vector store for semantic search
            bm25_index: BM25 index for keyword search (created if None)
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()

    async def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        retrieval_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            collection_name: Collection to search
            limit: Number of final results to return
            semantic_weight: Weight for semantic search (0-1, default from config)
            keyword_weight: Weight for BM25 search (0-1, default from config)
            retrieval_k: Number of candidates to retrieve from each method
            rrf_k: RRF constant (controls how much to discount lower ranks)
            score_threshold: Minimum similarity score for semantic search
            filter_conditions: Metadata filters for semantic search

        Returns:
            Top results ranked by combined score
        """
        # Use config defaults if not provided
        semantic_weight = semantic_weight or settings.hybrid_search_semantic_weight
        keyword_weight = keyword_weight or settings.hybrid_search_keyword_weight
        retrieval_k = retrieval_k or settings.hybrid_search_retrieval_k
        rrf_k = rrf_k or settings.hybrid_search_rrf_k

        # 1. Semantic vector search
        semantic_results = await self._semantic_search(
            query=query,
            collection_name=collection_name,
            limit=retrieval_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        # 2. BM25 keyword search
        keyword_results = await self._keyword_search(
            query=query,
            collection_name=collection_name,
            limit=retrieval_k,
        )

        # 3. Combine using Reciprocal Rank Fusion
        combined_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            rrf_k=rrf_k,
        )

        logger.info(
            f"Hybrid search completed: "
            f"{len(semantic_results)} semantic + {len(keyword_results)} keyword → "
            f"{len(combined_results)} fused results (top {limit} returned)"
        )

        # Return top-k results
        return combined_results[:limit]

    async def _semantic_search(
        self,
        query: str,
        collection_name: str,
        limit: int,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform semantic vector search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed_text(query)

            # Search vector store
            results = await self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

            logger.debug(f"Semantic search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        collection_name: str,
        limit: int,
    ) -> List[tuple[Dict[str, Any], float]]:
        """Perform BM25 keyword search."""
        try:
            # Check if BM25 index exists for this collection
            if not self.bm25_index.has_collection(collection_name):
                logger.warning(
                    f"BM25 index not found for collection '{collection_name}'. "
                    "Hybrid search will use semantic results only."
                )
                return []

            # Search BM25 index
            results = await self.bm25_index.search(
                collection_name=collection_name,
                query=query,
                limit=limit,
            )

            logger.debug(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[tuple[Dict[str, Any], float]],
        semantic_weight: float,
        keyword_weight: float,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF Formula:
            RRF_score(d) = Σ w_i / (k + rank_i(d))

        Where:
        - d is a document
        - w_i is the weight for search method i
        - k is a constant (typically 60)
        - rank_i(d) is the rank of document d in method i (0-indexed)

        RRF is more robust than score normalization since scores from
        different systems (embeddings vs BM25) aren't directly comparable.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from BM25 search
            semantic_weight: Weight for semantic results
            keyword_weight: Weight for keyword results
            rrf_k: RRF constant (higher = less penalty for lower ranks)

        Returns:
            Fused results sorted by RRF score
        """
        # Build rank maps
        # Semantic: use (document_id, chunk_index) as unique identifier
        semantic_ranks = {}
        for rank, result in enumerate(semantic_results):
            doc_id = str(result.document_id) if result.document_id else "unknown"
            chunk_idx = result.chunk_index
            key = (doc_id, chunk_idx)
            semantic_ranks[key] = rank

        # Keyword: same identifier
        keyword_ranks = {}
        for rank, (metadata, score) in enumerate(keyword_results):
            doc_id = str(metadata.get("document_id", "unknown"))
            chunk_idx = metadata.get("chunk_index", 0)
            key = (doc_id, chunk_idx)
            keyword_ranks[key] = rank

        # Get all unique document keys
        all_keys = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Calculate RRF scores
        rrf_scores: Dict[tuple, float] = {}

        for key in all_keys:
            score = 0.0

            # Add semantic contribution
            if key in semantic_ranks:
                score += semantic_weight / (rrf_k + semantic_ranks[key])

            # Add keyword contribution
            if key in keyword_ranks:
                score += keyword_weight / (rrf_k + keyword_ranks[key])

            rrf_scores[key] = score

        # Build result objects with RRF scores
        # We need to merge the original SearchResult objects
        results_map: Dict[tuple, SearchResult] = {}

        # Add semantic results
        for result in semantic_results:
            doc_id = str(result.document_id) if result.document_id else "unknown"
            key = (doc_id, result.chunk_index)
            results_map[key] = result

        # Add keyword-only results (not in semantic results)
        for metadata, bm25_score in keyword_results:
            doc_id = str(metadata.get("document_id", "unknown"))
            chunk_idx = metadata.get("chunk_index", 0)
            key = (doc_id, chunk_idx)

            if key not in results_map:
                # Create SearchResult from BM25 metadata
                results_map[key] = SearchResult(
                    content=metadata.get("content", ""),
                    score=bm25_score,  # Temporary, will be replaced with RRF
                    metadata={k: v for k, v in metadata.items() if k != "content"},
                    document_id=metadata.get("document_id"),
                    chunk_index=chunk_idx,
                )

        # Update all results with RRF scores and sort
        final_results = []
        for key, rrf_score in rrf_scores.items():
            result = results_map[key]
            # Create new SearchResult with RRF score
            final_results.append(
                SearchResult(
                    content=result.content,
                    score=rrf_score,  # Replace with RRF score
                    metadata=result.metadata,
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                )
            )

        # Sort by RRF score (descending)
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results

    async def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about hybrid search indexes for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with stats about vector and BM25 indexes
        """
        stats = {
            "collection_name": collection_name,
            "bm25_indexed": self.bm25_index.has_collection(collection_name),
        }

        if stats["bm25_indexed"]:
            stats["bm25_document_count"] = self.bm25_index.get_collection_size(
                collection_name
            )

        try:
            vector_stats = await self.vector_store.get_collection_stats(collection_name)
            stats["vector_stats"] = vector_stats
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {e}")
            stats["vector_stats"] = None

        return stats
