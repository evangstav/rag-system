"""
Deduplication strategies for RAG retrieval results.

Removes redundant chunks while preserving diversity to avoid repeating
the same information multiple times in the context.
"""

from typing import List, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.rag.protocols import SearchResult


class MMRDeduplicator:
    """
    Maximal Marginal Relevance (MMR) for result diversification.

    Balances relevance and diversity to avoid redundant chunks.
    Industry-standard approach for deduplication in RAG systems.

    Formula:
        MMR = λ * Relevance(chunk, query) - (1-λ) * max(Similarity(chunk, selected))

    Parameters:
        λ=1.0: Pure relevance (no deduplication)
        λ=0.7: Balanced (recommended, slight preference for relevance)
        λ=0.5: Equal weight to relevance and diversity
        λ=0.3: High diversity (aggressive deduplication)
        λ=0.0: Pure diversity (ignores relevance)

    Example:
        >>> deduplicator = MMRDeduplicator(lambda_param=0.7)
        >>> diverse_results = await deduplicator.deduplicate(
        ...     results=reranked_results,
        ...     top_k=8,
        ...     embeddings=embeddings
        ... )
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR deduplicator.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)

        Raises:
            ValueError: If lambda_param not in [0, 1]
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
        self.lambda_param = lambda_param

    async def deduplicate(
        self,
        results: List[SearchResult],
        top_k: int,
        embeddings: List[List[float]],
    ) -> List[SearchResult]:
        """
        Apply MMR to select diverse results.

        Greedy iterative algorithm:
        1. Start with highest-scoring result
        2. For each remaining result, calculate MMR score
        3. Select result with highest MMR
        4. Repeat until top_k selected

        Args:
            results: Search results (must be sorted by relevance score)
            top_k: Number of results to select
            embeddings: Embeddings for each result (for similarity calculation)

        Returns:
            Diverse subset of results, ordered by selection (most relevant first)

        Raises:
            ValueError: If results and embeddings lengths don't match

        Example:
            >>> # After reranking
            >>> embeddings = await embedding_provider.embed_batch(
            ...     [r.content for r in reranked_results]
            ... )
            >>> diverse = await deduplicator.deduplicate(
            ...     results=reranked_results,
            ...     top_k=8,
            ...     embeddings=embeddings
            ... )
        """
        if not results or not embeddings:
            return []

        if len(results) != len(embeddings):
            raise ValueError("results and embeddings must have same length")

        # Handle case where we have fewer results than requested
        if len(results) <= top_k:
            return results

        # Convert to numpy for vectorized operations
        embeddings_np = np.array(embeddings)
        scores = np.array([r.score for r in results])

        # Normalize scores to [0, 1] for consistent weighting
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = scores

        selected_indices: List[int] = []
        remaining_indices = set(range(len(results)))

        # Start with highest-scoring result
        first_idx = 0
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select results with high MMR
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = normalized_scores[idx]

                # Diversity component: max similarity to already selected
                similarities = cosine_similarity(
                    embeddings_np[idx].reshape(1, -1), embeddings_np[selected_indices]
                )[0]
                max_similarity = similarities.max()

                # MMR formula
                mmr = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * max_similarity
                )
                mmr_scores.append((idx, mmr))

            # Select result with highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected results in order of selection
        return [results[idx] for idx in selected_indices]


class TokenDeduplicator:
    """
    Fast n-gram based deduplication.

    Removes chunks with high token overlap (e.g., from overlapping chunks).
    Much faster than MMR but less sophisticated - uses Jaccard similarity
    on n-grams instead of semantic embeddings.

    Good for:
    - High-volume applications where speed is critical
    - When embeddings are not available
    - Quick filtering before more expensive operations

    Example:
        >>> deduplicator = TokenDeduplicator(ngram_size=5, overlap_threshold=0.6)
        >>> unique_results = await deduplicator.deduplicate(
        ...     results=search_results,
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        ngram_size: int = 5,
        overlap_threshold: float = 0.6,
    ):
        """
        Initialize token deduplicator.

        Args:
            ngram_size: Size of n-grams to compare (5-10 recommended)
            overlap_threshold: Fraction of n-grams that must match (0.6 = 60%)

        Raises:
            ValueError: If parameters out of valid range
        """
        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")
        if not 0 < overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")

        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold

    def _get_ngrams(self, text: str) -> Set[str]:
        """
        Extract n-grams from text.

        Args:
            text: Input text

        Returns:
            Set of n-gram strings
        """
        tokens = text.lower().split()
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i : i + self.ngram_size])
            ngrams.add(ngram)
        return ngrams

    def _calculate_overlap(self, ngrams1: Set[str], ngrams2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between n-gram sets.

        Args:
            ngrams1: First n-gram set
            ngrams2: Second n-gram set

        Returns:
            Jaccard similarity [0, 1]
        """
        if not ngrams1 or not ngrams2:
            return 0.0
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0

    async def deduplicate(
        self,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Remove results with high n-gram overlap.

        Greedy algorithm: Keep highest-scoring results, skip similar ones.

        Args:
            results: Search results (should be sorted by relevance)
            top_k: Number of results to return

        Returns:
            Deduplicated results (up to top_k)

        Example:
            >>> results = await deduplicator.deduplicate(
            ...     results=search_results,
            ...     top_k=10
            ... )
            >>> # Results with >60% n-gram overlap are removed
        """
        if not results:
            return []

        selected: List[SearchResult] = []
        selected_ngrams: List[Set[str]] = []

        for result in results:
            if len(selected) >= top_k:
                break

            # Check overlap with already selected results
            current_ngrams = self._get_ngrams(result.content)
            is_duplicate = False

            for prev_ngrams in selected_ngrams:
                overlap = self._calculate_overlap(current_ngrams, prev_ngrams)
                if overlap >= self.overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(result)
                selected_ngrams.append(current_ngrams)

        return selected


class DocumentDeduplicator:
    """
    Document-level deduplication (simplest approach).

    If multiple chunks from same document are retrieved, keep only
    the highest-scoring one. Useful when you want at most one chunk
    per source document.

    Trade-off: Simple and fast, but may discard relevant chunks
    from the same document that cover different topics.

    Example:
        >>> deduplicator = DocumentDeduplicator()
        >>> one_per_doc = await deduplicator.deduplicate(
        ...     results=search_results,
        ...     top_k=10
        ... )
    """

    async def deduplicate(
        self,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Keep only highest-scoring chunk per document.

        Args:
            results: Search results (should be sorted by relevance)
            top_k: Number of results to return

        Returns:
            Deduplicated results with at most one chunk per document

        Example:
            >>> # If results contain 3 chunks from doc1 and 2 from doc2,
            >>> # only the highest-scoring chunk from each will be kept
            >>> results = await deduplicator.deduplicate(results, top_k=10)
        """
        if not results:
            return []

        seen_docs: Set[str] = set()
        deduplicated: List[SearchResult] = []

        for result in results:
            # Use document_id if available, otherwise fall back to filename
            doc_id = str(result.document_id) if result.document_id else result.filename

            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                deduplicated.append(result)

                if len(deduplicated) >= top_k:
                    break

        return deduplicated
