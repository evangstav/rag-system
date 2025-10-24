"""
Protocols for RAG evaluation framework.

Defines interfaces that any RAG implementation must satisfy to be testable.
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from uuid import UUID


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_index: Optional[int] = None


class RAGRetriever(Protocol):
    """
    Protocol for RAG retrieval implementations.

    Any retrieval strategy must implement this interface to be testable.
    """

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            collection_name: Knowledge pool to search
            limit: Maximum number of results
            **kwargs: Implementation-specific parameters

        Returns:
            List of retrieval results sorted by relevance
        """
        ...

    async def get_name(self) -> str:
        """Return a descriptive name for this retriever."""
        ...


@dataclass
class TestQuery:
    """Test case for RAG evaluation."""
    id: str
    query: str
    relevant_document_ids: List[str]
    collection_name: str
    query_type: str = "general"  # factual, how-to, comparison, etc.
    difficulty: str = "medium"  # easy, medium, hard
    description: Optional[str] = None
    expected_answer_keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationMetrics:
    """Metrics for a single query evaluation."""
    # Retrieval metrics
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    map_score: float  # Mean Average Precision

    # Additional info
    num_retrieved: int
    num_relevant: int
    top_score: float

    # Per-position relevance (for analysis)
    relevance_at_position: List[bool]


@dataclass
class QueryEvaluation:
    """Complete evaluation result for a single query."""
    test_query: TestQuery
    retrieved_results: List[RetrievalResult]
    metrics: EvaluationMetrics
    latency_ms: float
    timestamp: str


@dataclass
class SuiteResults:
    """Results for entire test suite on one retriever."""
    retriever_name: str
    config: Dict[str, Any]
    query_evaluations: List[QueryEvaluation]
    aggregate_metrics: Dict[str, float]
    breakdowns: Dict[str, Dict[str, float]]
    total_latency_ms: float
    timestamp: str
