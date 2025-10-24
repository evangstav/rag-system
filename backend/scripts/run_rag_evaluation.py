#!/usr/bin/env python3
"""
RAG Evaluation CLI

Run evaluation test suites against different RAG implementations.

Usage:
    # Create example test suite
    python scripts/run_rag_evaluation.py create-example --output tests/data/example_suite.json

    # Run single implementation
    python scripts/run_rag_evaluation.py run \
        --test-suite tests/data/example_suite.json \
        --implementation baseline \
        --output results/baseline.json

    # Compare multiple implementations
    python scripts/run_rag_evaluation.py compare \
        --baseline results/baseline.json \
        --comparisons results/hybrid.json results/reranked.json \
        --output results/comparison.json
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.evaluation.runner import EvaluationRunner
from app.evaluation.loader import TestSuiteLoader
from app.evaluation.adapters import (
    BaselineRAGAdapter,
    HybridSearchAdapter,
    RerankedAdapter,
    ConfigurableAdapter,
)
from app.evaluation.comparison import ResultsComparator, ResultsExporter


async def create_example_suite(args):
    """Create an example test suite file."""
    output_path = Path(args.output)
    TestSuiteLoader.create_example_suite(output_path)


async def run_evaluation(args):
    """Run evaluation on a single implementation."""
    print(f"Loading test suite from: {args.test_suite}")
    test_queries = TestSuiteLoader.load_from_json(args.test_suite)
    print(f"Loaded {len(test_queries)} test queries\n")

    # Initialize RAG service
    rag_service = RAGService()

    # Select implementation
    implementation = args.implementation.lower()

    if implementation == "baseline":
        retriever = BaselineRAGAdapter(rag_service)
        config = {"type": "baseline"}

    elif implementation == "hybrid":
        alpha = args.hybrid_alpha if hasattr(args, 'hybrid_alpha') else 0.7
        retriever = HybridSearchAdapter(rag_service, semantic_weight=alpha)
        config = {"type": "hybrid", "semantic_weight": alpha}

    elif implementation == "reranked":
        base = BaselineRAGAdapter(rag_service)
        retriever = RerankedAdapter(base, rerank_top_k=20, final_k=args.k)
        config = {"type": "reranked", "rerank_top_k": 20}

    elif implementation == "custom":
        if not args.config:
            print("ERROR: --config required for custom implementation")
            sys.exit(1)

        import json
        with open(args.config) as f:
            custom_config = json.load(f)

        retriever = ConfigurableAdapter(
            rag_service,
            config=custom_config,
            name=args.name or "Custom"
        )
        config = custom_config

    else:
        print(f"ERROR: Unknown implementation: {implementation}")
        print("Available: baseline, hybrid, reranked, custom")
        sys.exit(1)

    # Run evaluation
    runner = EvaluationRunner(k=args.k)
    results = await runner.run_suite(test_queries, retriever, config)

    # Print summary
    runner.print_summary(results)

    # Export results
    if args.output:
        ResultsExporter.export_results(results, args.output)

    return results


async def compare_results(args):
    """Compare multiple evaluation results."""
    print("Loading baseline results...")
    baseline_data = ResultsExporter.import_results(args.baseline)

    print("Loading comparison results...")
    comparison_data = []
    for comp_path in args.comparisons:
        data = ResultsExporter.import_results(comp_path)
        comparison_data.append(data)

    # Convert to SuiteResults-like dicts for comparison
    from app.evaluation.protocols import SuiteResults

    def dict_to_suite_results(data: dict) -> SuiteResults:
        """Convert dict to SuiteResults (simplified)."""
        return SuiteResults(
            retriever_name=data["retriever_name"],
            config=data["config"],
            query_evaluations=[],  # Not needed for comparison
            aggregate_metrics=data["aggregate_metrics"],
            breakdowns=data["breakdowns"],
            total_latency_ms=data["total_latency_ms"],
            timestamp=data["timestamp"],
        )

    baseline = dict_to_suite_results(baseline_data)
    comparisons = [dict_to_suite_results(d) for d in comparison_data]

    # Generate comparison report
    report = ResultsComparator.compare(baseline, comparisons)

    # Print report
    ResultsComparator.print_comparison(report)

    # Export report
    if args.output:
        ResultsComparator.export_comparison(report, args.output)


async def generate_summary(args):
    """Generate markdown summary from multiple result files."""
    markdown = ResultsExporter.create_summary_table(args.results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"Summary saved to: {args.output}")
    else:
        print(markdown)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create example suite
    parser_create = subparsers.add_parser(
        'create-example',
        help='Create an example test suite'
    )
    parser_create.add_argument(
        '--output',
        required=True,
        help='Output path for example suite'
    )

    # Run evaluation
    parser_run = subparsers.add_parser(
        'run',
        help='Run evaluation on an implementation'
    )
    parser_run.add_argument(
        '--test-suite',
        required=True,
        help='Path to test suite JSON file'
    )
    parser_run.add_argument(
        '--implementation',
        choices=['baseline', 'hybrid', 'reranked', 'custom'],
        default='baseline',
        help='Implementation to evaluate'
    )
    parser_run.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results to evaluate (default: 5)'
    )
    parser_run.add_argument(
        '--config',
        help='Config file for custom implementation'
    )
    parser_run.add_argument(
        '--name',
        help='Name for custom implementation'
    )
    parser_run.add_argument(
        '--output',
        help='Output path for results JSON'
    )
    parser_run.add_argument(
        '--hybrid-alpha',
        type=float,
        default=0.7,
        help='Semantic weight for hybrid search (default: 0.7)'
    )

    # Compare results
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare multiple evaluation results'
    )
    parser_compare.add_argument(
        '--baseline',
        required=True,
        help='Baseline results JSON file'
    )
    parser_compare.add_argument(
        '--comparisons',
        nargs='+',
        required=True,
        help='Comparison results JSON files'
    )
    parser_compare.add_argument(
        '--output',
        help='Output path for comparison report'
    )

    # Generate summary
    parser_summary = subparsers.add_parser(
        'summary',
        help='Generate markdown summary table'
    )
    parser_summary.add_argument(
        '--results',
        nargs='+',
        required=True,
        help='Results JSON files to include'
    )
    parser_summary.add_argument(
        '--output',
        help='Output path for markdown file'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run appropriate command
    if args.command == 'create-example':
        asyncio.run(create_example_suite(args))
    elif args.command == 'run':
        asyncio.run(run_evaluation(args))
    elif args.command == 'compare':
        asyncio.run(compare_results(args))
    elif args.command == 'summary':
        asyncio.run(generate_summary(args))


if __name__ == '__main__':
    main()
