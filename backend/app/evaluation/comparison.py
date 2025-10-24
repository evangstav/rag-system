"""
Comparison tools for evaluating multiple RAG implementations.

Compare different retrievers and generate comparative reports.
"""

from typing import List, Dict
from dataclasses import asdict
import json
from pathlib import Path

from app.evaluation.protocols import SuiteResults


class ResultsComparator:
    """Compare multiple RAG evaluation results."""

    @staticmethod
    def compare(
        baseline: SuiteResults,
        comparisons: List[SuiteResults],
    ) -> Dict:
        """
        Compare multiple implementations against a baseline.

        Args:
            baseline: Baseline results to compare against
            comparisons: List of alternative implementations

        Returns:
            Comparison report with improvements/regressions
        """
        report = {
            "baseline": {
                "name": baseline.retriever_name,
                "metrics": baseline.aggregate_metrics,
            },
            "comparisons": [],
        }

        for comp in comparisons:
            comparison = {
                "name": comp.retriever_name,
                "metrics": comp.aggregate_metrics,
                "improvements": {},
            }

            # Compute improvements for each metric
            for metric in baseline.aggregate_metrics:
                baseline_val = baseline.aggregate_metrics[metric]
                comp_val = comp.aggregate_metrics[metric]

                if baseline_val > 0:
                    pct_change = ((comp_val - baseline_val) / baseline_val) * 100
                else:
                    pct_change = 0.0

                comparison["improvements"][metric] = {
                    "baseline": baseline_val,
                    "comparison": comp_val,
                    "absolute_change": comp_val - baseline_val,
                    "percent_change": pct_change,
                }

            report["comparisons"].append(comparison)

        return report

    @staticmethod
    def print_comparison(report: Dict):
        """Print human-readable comparison report."""
        print(f"\n{'=' * 80}")
        print(f"COMPARISON REPORT")
        print(f"{'=' * 80}\n")

        baseline = report["baseline"]
        print(f"Baseline: {baseline['name']}")

        for comp in report["comparisons"]:
            print(f"\n{'─' * 80}")
            print(f"Comparing: {comp['name']}")
            print(f"{'─' * 80}\n")

            print(f"{'Metric':<25} {'Baseline':>12} {'New':>12} {'Change':>12} {'%':>8}")
            print(f"{'-' * 80}")

            for metric, improvement in comp["improvements"].items():
                baseline_val = improvement["baseline"]
                comp_val = improvement["comparison"]
                abs_change = improvement["absolute_change"]
                pct_change = improvement["percent_change"]

                # Color coding for terminal (optional)
                change_indicator = "↑" if abs_change > 0 else "↓" if abs_change < 0 else "="

                if "latency" in metric:
                    # For latency, lower is better
                    print(f"{metric:<25} {baseline_val:>10.1f}ms {comp_val:>10.1f}ms "
                          f"{abs_change:>+10.1f}ms {pct_change:>+7.1f}%")
                else:
                    print(f"{metric:<25} {baseline_val:>12.3f} {comp_val:>12.3f} "
                          f"{abs_change:>+12.3f} {pct_change:>+7.1f}%")

        print(f"\n{'=' * 80}\n")

    @staticmethod
    def export_comparison(report: Dict, output_path: str | Path):
        """Export comparison report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Comparison report saved to: {output_path}")


class ResultsExporter:
    """Export/import evaluation results."""

    @staticmethod
    def export_results(results: SuiteResults, output_path: str | Path):
        """Export evaluation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            "retriever_name": results.retriever_name,
            "config": results.config,
            "aggregate_metrics": results.aggregate_metrics,
            "breakdowns": results.breakdowns,
            "total_latency_ms": results.total_latency_ms,
            "timestamp": results.timestamp,
            "num_queries": len(results.query_evaluations),
            # Optionally include per-query details
            "query_evaluations": [
                {
                    "query_id": e.test_query.id,
                    "query": e.test_query.query,
                    "query_type": e.test_query.query_type,
                    "difficulty": e.test_query.difficulty,
                    "metrics": asdict(e.metrics),
                    "latency_ms": e.latency_ms,
                    "num_retrieved": len(e.retrieved_results),
                    "retrieved_ids": [r.document_id for r in e.retrieved_results],
                }
                for e in results.query_evaluations
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results exported to: {output_path}")

    @staticmethod
    def import_results(input_path: str | Path) -> Dict:
        """Import evaluation results from JSON file."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def create_summary_table(results_files: List[str | Path]) -> str:
        """
        Create a markdown summary table comparing multiple results.

        Useful for documentation and reports.
        """
        all_results = []
        for file_path in results_files:
            data = ResultsExporter.import_results(file_path)
            all_results.append(data)

        # Build markdown table
        lines = []
        lines.append("# RAG Evaluation Results\n")
        lines.append(f"Compared {len(all_results)} implementations\n")
        lines.append("## Summary\n")

        # Table header
        lines.append("| Implementation | Precision@5 | Recall@5 | NDCG@5 | MRR | Avg Latency |")
        lines.append("|---------------|-------------|----------|--------|-----|-------------|")

        # Table rows
        for result in all_results:
            metrics = result["aggregate_metrics"]
            lines.append(
                f"| {result['retriever_name']:<25} "
                f"| {metrics['precision_at_k']:.3f} "
                f"| {metrics['recall_at_k']:.3f} "
                f"| {metrics['ndcg_at_k']:.3f} "
                f"| {metrics['mrr']:.3f} "
                f"| {metrics['avg_latency_ms']:.0f}ms |"
            )

        return "\n".join(lines)
