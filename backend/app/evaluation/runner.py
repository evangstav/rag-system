"""
Evaluation runner for RAG test suites.

Executes test queries against any RAG implementation and produces metrics.
"""

import time
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

from app.evaluation.protocols import (
    RAGRetriever,
    TestQuery,
    RetrievalResult,
    QueryEvaluation,
    EvaluationMetrics,
    SuiteResults,
)
from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    mean_average_precision,
    compute_relevance_at_positions,
)


class EvaluationRunner:
    """
    Run evaluation test suites against RAG implementations.

    This runner is implementation-agnostic - it works with any
    retriever that conforms to the RAGRetriever protocol.
    """

    def __init__(self, k: int = 5):
        """
        Initialize evaluation runner.

        Args:
            k: Number of results to evaluate (default: 5)
        """
        self.k = k

    async def evaluate_query(
        self,
        test_query: TestQuery,
        retriever: RAGRetriever,
    ) -> QueryEvaluation:
        """
        Evaluate a single query against a retriever.

        Args:
            test_query: Test case to evaluate
            retriever: RAG retriever implementation

        Returns:
            Complete evaluation results for this query
        """
        # Measure retrieval latency
        start_time = time.perf_counter()

        results = await retriever.retrieve(
            query=test_query.query,
            collection_name=test_query.collection_name,
            limit=self.k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract retrieved document IDs
        retrieved_ids = [r.document_id for r in results]
        relevant_ids = set(test_query.relevant_document_ids)

        # Compute metrics
        metrics = self._compute_metrics(retrieved_ids, relevant_ids, results)

        return QueryEvaluation(
            test_query=test_query,
            retrieved_results=results,
            metrics=metrics,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _compute_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids: set[str],
        results: List[RetrievalResult],
    ) -> EvaluationMetrics:
        """Compute all retrieval metrics for a query."""
        # Deduplicate document IDs while preserving order for ranking metrics
        # This ensures we evaluate at document-level, not chunk-level
        seen = set()
        unique_retrieved_ids = []
        for doc_id in retrieved_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_retrieved_ids.append(doc_id)

        return EvaluationMetrics(
            precision_at_k=precision_at_k(unique_retrieved_ids, relevant_ids, self.k),
            recall_at_k=recall_at_k(unique_retrieved_ids, relevant_ids, self.k),
            mrr=mean_reciprocal_rank(unique_retrieved_ids, relevant_ids),
            ndcg_at_k=ndcg_at_k(unique_retrieved_ids, relevant_ids, self.k),
            map_score=mean_average_precision(unique_retrieved_ids, relevant_ids),
            num_retrieved=len(unique_retrieved_ids),
            num_relevant=len(relevant_ids),
            top_score=results[0].score if results else 0.0,
            relevance_at_position=compute_relevance_at_positions(
                unique_retrieved_ids, relevant_ids, self.k
            ),
        )

    async def run_suite(
        self,
        test_queries: List[TestQuery],
        retriever: RAGRetriever,
        config: Dict[str, Any] = None,
    ) -> SuiteResults:
        """
        Run complete test suite against a retriever.

        Args:
            test_queries: List of test cases
            retriever: RAG retriever implementation
            config: Configuration used for this run (for documentation)

        Returns:
            Complete results including per-query and aggregate metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Running evaluation suite")
        print(f"Retriever: {await retriever.get_name()}")
        print(f"Test queries: {len(test_queries)}")
        print(f"k={self.k}")
        print(f"{'=' * 60}\n")

        # Evaluate each query
        query_evaluations = []
        total_latency = 0.0

        for i, test_query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Evaluating: {test_query.query[:60]}...")

            evaluation = await self.evaluate_query(test_query, retriever)
            query_evaluations.append(evaluation)
            total_latency += evaluation.latency_ms

            # Print quick summary
            m = evaluation.metrics
            print(f"    P@{self.k}={m.precision_at_k:.2f} | "
                  f"R@{self.k}={m.recall_at_k:.2f} | "
                  f"NDCG@{self.k}={m.ndcg_at_k:.2f} | "
                  f"Latency={evaluation.latency_ms:.0f}ms")

        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(query_evaluations)

        # Compute breakdowns
        breakdowns = self._compute_breakdowns(query_evaluations)

        return SuiteResults(
            retriever_name=await retriever.get_name(),
            config=config or {},
            query_evaluations=query_evaluations,
            aggregate_metrics=aggregate_metrics,
            breakdowns=breakdowns,
            total_latency_ms=total_latency,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _compute_aggregate_metrics(
        self,
        evaluations: List[QueryEvaluation]
    ) -> Dict[str, float]:
        """Compute average metrics across all queries."""
        if not evaluations:
            return {}

        metrics_to_aggregate = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "ndcg_at_k",
            "map_score",
        ]

        aggregates = {}
        for metric_name in metrics_to_aggregate:
            values = [getattr(e.metrics, metric_name) for e in evaluations]
            aggregates[metric_name] = sum(values) / len(values)

        # Additional aggregates
        aggregates["avg_latency_ms"] = sum(e.latency_ms for e in evaluations) / len(evaluations)
        aggregates["p50_latency_ms"] = self._percentile([e.latency_ms for e in evaluations], 50)
        aggregates["p95_latency_ms"] = self._percentile([e.latency_ms for e in evaluations], 95)
        aggregates["avg_top_score"] = sum(e.metrics.top_score for e in evaluations) / len(evaluations)

        return aggregates

    def _compute_breakdowns(
        self,
        evaluations: List[QueryEvaluation]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by query type and difficulty."""
        by_type = defaultdict(list)
        by_difficulty = defaultdict(list)

        for eval in evaluations:
            query_type = eval.test_query.query_type
            difficulty = eval.test_query.difficulty
            ndcg = eval.metrics.ndcg_at_k

            by_type[query_type].append(ndcg)
            by_difficulty[difficulty].append(ndcg)

        return {
            "by_query_type": {
                qtype: sum(scores) / len(scores)
                for qtype, scores in by_type.items()
            },
            "by_difficulty": {
                diff: sum(scores) / len(scores)
                for diff, scores in by_difficulty.items()
            },
        }

    def _percentile(self, values: List[float], p: int) -> float:
        """Compute percentile of a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * p / 100
        f = int(k)
        c = int(k) + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return d0 + d1

    def print_summary(self, results: SuiteResults):
        """Print human-readable summary of results."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Retriever: {results.retriever_name}")
        print(f"Queries evaluated: {len(results.query_evaluations)}")
        print(f"Timestamp: {results.timestamp}")

        print(f"\nAggregate Metrics (k={self.k}):")
        print(f"{'  Metric':<30} {'Value':>10}")
        print(f"  {'-' * 40}")
        for metric, value in results.aggregate_metrics.items():
            if "latency" in metric:
                print(f"  {metric:<30} {value:>8.1f}ms")
            else:
                print(f"  {metric:<30} {value:>10.3f}")

        print(f"\nBreakdown by Query Type:")
        for qtype, score in results.breakdowns["by_query_type"].items():
            print(f"  {qtype:<30} {score:>10.3f}")

        print(f"\nBreakdown by Difficulty:")
        for diff, score in results.breakdowns["by_difficulty"].items():
            print(f"  {diff:<30} {score:>10.3f}")

        print(f"\n{'=' * 60}\n")
