"""
Retrieval metrics implementations.

Standard metrics for evaluating information retrieval quality.
"""

from typing import List, Set
import math


def precision_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5
) -> float:
    """
    Precision@K: What fraction of retrieved docs are relevant?

    Formula: (# relevant docs in top K) / K

    Example:
        Retrieved: [doc1, doc2, doc3, doc4, doc5]
        Relevant:  {doc1, doc3}
        Precision@5 = 2/5 = 0.4
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
    return relevant_retrieved / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Recall@K: What fraction of relevant docs were retrieved?

    Formula: (# relevant docs in top K) / (total # relevant docs)

    Example:
        Retrieved: [doc1, doc2, doc3, doc4, doc5]
        Relevant:  {doc1, doc3, doc6}
        Recall@5 = 2/3 = 0.67 (missed doc6)
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    MRR: How high is the first relevant document ranked?

    Formula: 1 / (rank of first relevant doc)

    Example:
        Retrieved: [doc1, doc2, doc3, doc4, doc5]
        Relevant:  {doc3, doc6}
        First relevant doc is doc3 at rank 3
        MRR = 1/3 = 0.333
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Discounted Cumulative Gain at K.

    Formula: sum(relevance / log2(rank + 1))

    Relevance is binary here (1 if relevant, 0 otherwise).
    """
    score = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        relevance = 1.0 if doc_id in relevant_ids else 0.0
        # Use log2(i + 1) for position discount
        score += relevance / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Rewards:
    - Relevant docs ranked higher
    - Multiple relevant docs retrieved

    Formula: DCG / IDCG (ideal DCG)
    """
    if not relevant_ids:
        return 0.0

    # Actual DCG
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Ideal DCG (perfect ranking: all relevant docs first)
    ideal_ranking = list(relevant_ids)[:k] + [f"dummy_{i}" for i in range(k)]
    ideal_dcg = dcg_at_k(ideal_ranking, relevant_ids, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def mean_average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    MAP: Mean Average Precision across all relevant documents.

    Average of precision values at each relevant document position.

    Example:
        Retrieved: [doc1, doc2, doc3, doc4, doc5]
        Relevant:  {doc1, doc3}

        Precision at doc1 (rank 1): 1/1 = 1.0
        Precision at doc3 (rank 3): 2/3 = 0.667
        MAP = (1.0 + 0.667) / 2 = 0.833
    """
    if not relevant_ids:
        return 0.0

    precisions = []
    relevant_count = 0

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            relevant_count += 1
            precisions.append(relevant_count / i)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant_ids)


def compute_relevance_at_positions(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5
) -> List[bool]:
    """
    Return boolean list indicating relevance at each position.

    Useful for analysis and visualization.
    """
    return [doc_id in relevant_ids for doc_id in retrieved_ids[:k]]
