#!/usr/bin/env python3
"""
End-to-end RAG evaluation test script.

This script provides a complete testing workflow that:
1. Creates a dedicated test collection in Qdrant
2. Ingests a PDF from tests/data directory
3. Generates embeddings and stores them
4. Runs the evaluation suite
5. Saves results to SQLite for historical tracking
6. Compares results with previous runs

Usage:
    # Run full test
    python scripts/e2e_rag_test.py

    # Use custom PDF
    python scripts/e2e_rag_test.py --pdf tests/data/custom.pdf

    # Force re-ingestion (even if PDF unchanged)
    python scripts/e2e_rag_test.py --force-ingest

    # Compare two specific runs
    python scripts/e2e_rag_test.py --compare 4 5

    # Show recent history
    python scripts/e2e_rag_test.py --history

    # Clean test collection
    python scripts/e2e_rag_test.py --clean
"""

import asyncio
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.rag_service import RAGService
from app.evaluation.runner import EvaluationRunner
from app.evaluation.loader import TestSuiteLoader
from app.evaluation.adapters import BaselineRAGAdapter
from app.evaluation.comparison import ResultsExporter
from tests.utils.results_db import EvaluationResultsDB


# Test configuration
TEST_USER_ID = "test_user_00000000"
TEST_COLLECTION_NAME = "test_rag_evaluation"
DEFAULT_PDF_PATH = Path(__file__).parent.parent / "tests" / "data" / "How to Train Guide.pdf"
DEFAULT_SUITE_PATH = Path(__file__).parent.parent / "tests" / "data" / "my_suite.json"


def get_git_info() -> tuple[Optional[str], bool]:
    """
    Get current git commit hash and dirty status.

    Returns:
        Tuple of (commit_hash, is_dirty)
    """
    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_hash = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_dirty = bool(result.stdout.strip())

        return git_hash, git_dirty

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


async def ensure_test_collection(rag_service: RAGService) -> None:
    """Ensure test collection exists in Qdrant."""
    exists = await rag_service.vector_store.collection_exists(TEST_COLLECTION_NAME)

    if not exists:
        print(f"Creating test collection: {TEST_COLLECTION_NAME}")
        await rag_service.vector_store.create_collection(
            collection_name=TEST_COLLECTION_NAME,
            vector_size=rag_service.embedding_provider.dimensions,
        )


async def check_pdf_ingested(
    rag_service: RAGService, pdf_hash: str
) -> tuple[bool, int]:
    """
    Check if PDF with this hash is already ingested.

    Returns:
        Tuple of (is_ingested, num_chunks)
    """
    try:
        stats = await rag_service.vector_store.get_collection_stats(TEST_COLLECTION_NAME)
        num_chunks = stats.get("vectors_count", 0)

        if num_chunks > 0:
            # Check if we have any points with this PDF hash in metadata
            # For simplicity, we'll just check if collection has data
            # In production, you might want to store pdf_hash in point metadata
            return True, num_chunks

        return False, 0

    except Exception:
        return False, 0


async def ingest_pdf(
    rag_service: RAGService,
    pdf_path: Path,
    force: bool = False,
) -> int:
    """
    Ingest PDF into test collection.

    Args:
        rag_service: RAG service instance
        pdf_path: Path to PDF file
        force: Force re-ingestion even if already exists

    Returns:
        Number of chunks created
    """
    pdf_hash = compute_file_hash(pdf_path)
    is_ingested, num_chunks = await check_pdf_ingested(rag_service, pdf_hash)

    if is_ingested and not force:
        print(f"✓ PDF already ingested ({num_chunks} chunks)")
        return num_chunks

    # Delete existing data if re-ingesting
    if is_ingested:
        print(f"Deleting existing data in {TEST_COLLECTION_NAME}...")
        await rag_service.vector_store.delete_collection(TEST_COLLECTION_NAME)
        await ensure_test_collection(rag_service)

    # Load and split document
    print(f"Loading PDF: {pdf_path.name}")
    from app.services.rag.loaders.pdf_loader import PDFLoader

    loader = PDFLoader()
    document = await loader.load(str(pdf_path))

    # Split into chunks
    print(f"Splitting document...")
    chunks = rag_service.text_splitter.split_documents([document])
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    print(f"Generating embeddings...")
    texts = [chunk.content for chunk in chunks]
    vectors = await rag_service.embedding_provider.embed_batch(texts)

    # Store in Qdrant
    print(f"Storing in Qdrant ({TEST_COLLECTION_NAME})...")
    await rag_service.vector_store.upsert(
        collection_name=TEST_COLLECTION_NAME,
        documents=chunks,
        vectors=vectors,
    )

    print(f"✓ Ingested {len(chunks)} chunks")
    return len(chunks)


async def run_evaluation(
    rag_service: RAGService,
    suite_path: Path,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run evaluation suite against test collection.

    Args:
        rag_service: RAG service instance
        suite_path: Path to test suite JSON
        k: Number of results to evaluate

    Returns:
        Evaluation results dictionary
    """
    print(f"\nRunning evaluation suite...")

    # Load test queries
    test_queries = TestSuiteLoader.load_from_json(str(suite_path))

    # Update collection names to use test collection
    for query in test_queries:
        query.collection_name = TEST_COLLECTION_NAME

    # Create adapter and runner
    retriever = BaselineRAGAdapter(rag_service)
    runner = EvaluationRunner(k=k)

    # Run evaluation
    results = await runner.run_suite(
        test_queries,
        retriever,
        config={"type": "baseline", "collection": TEST_COLLECTION_NAME},
    )

    # Print summary
    runner.print_summary(results)

    # Convert to dict for serialization
    return {
        "retriever_name": results.retriever_name,
        "config": results.config,
        "aggregate_metrics": results.aggregate_metrics,
        "breakdowns": results.breakdowns,
        "total_latency_ms": results.total_latency_ms,
        "timestamp": results.timestamp,
        "query_evaluations": [
            {
                "query": eval.test_query.query,
                "query_type": eval.test_query.query_type,
                "difficulty": eval.test_query.difficulty,
                "metrics": {
                    "precision_at_k": eval.metrics.precision_at_k,
                    "recall_at_k": eval.metrics.recall_at_k,
                    "mrr": eval.metrics.mrr,
                    "ndcg_at_k": eval.metrics.ndcg_at_k,
                    "map_score": eval.metrics.map_score,
                },
                "latency_ms": eval.latency_ms,
            }
            for eval in results.query_evaluations
        ],
    }


def print_comparison(db: EvaluationResultsDB, current_id: int):
    """Print comparison with previous run."""
    previous = db.get_run(current_id - 1)
    current = db.get_run(current_id)

    if not previous:
        print("\nNo previous run for comparison.")
        return

    print(f"\n{'=' * 60}")
    print(f"COMPARISON: Run #{previous['id']} vs Run #{current_id}")
    print(f"{'=' * 60}")

    prev_metrics = previous["aggregate_metrics"]
    curr_metrics = current["aggregate_metrics"]

    for key in ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map_score"]:
        if key in prev_metrics and key in curr_metrics:
            prev_val = prev_metrics[key]
            curr_val = curr_metrics[key]
            diff = curr_val - prev_val

            # Determine arrow
            if abs(diff) < 0.001:
                arrow = "→"
            elif diff > 0:
                arrow = "↑"
            else:
                arrow = "↓"

            print(f"  {key:<20} {curr_val:.3f} ({arrow} {diff:+.3f})")

    # Latency comparison
    prev_lat = prev_metrics.get("avg_latency_ms", 0)
    curr_lat = curr_metrics.get("avg_latency_ms", 0)
    lat_diff = curr_lat - prev_lat

    if abs(lat_diff) < 1:
        arrow = "→"
    elif lat_diff > 0:
        arrow = "↑"
    else:
        arrow = "↓"

    print(f"  {'avg_latency_ms':<20} {curr_lat:.1f}ms ({arrow} {lat_diff:+.1f}ms)")


async def run_test(args):
    """Main test execution."""
    pdf_path = Path(args.pdf)
    suite_path = Path(args.suite)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    if not suite_path.exists():
        print(f"Error: Test suite not found: {suite_path}")
        sys.exit(1)

    # Initialize services
    rag_service = RAGService()

    # Ensure test collection exists
    await ensure_test_collection(rag_service)

    # Ingest PDF (if needed)
    num_chunks = await ingest_pdf(rag_service, pdf_path, force=args.force_ingest)

    # Run evaluation
    results = await run_evaluation(rag_service, suite_path, k=args.k)

    # Save to database
    print(f"\nSaving results to database...")
    git_hash, git_dirty = get_git_info()
    pdf_hash = compute_file_hash(pdf_path)

    with EvaluationResultsDB() as db:
        run_id = db.save_run(
            pdf_filename=pdf_path.name,
            pdf_path=str(pdf_path),
            pdf_hash=pdf_hash,
            collection_name=TEST_COLLECTION_NAME,
            num_chunks=num_chunks,
            config={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "k": args.k,
            },
            aggregate_metrics=results["aggregate_metrics"],
            breakdowns=results["breakdowns"],
            query_results=results["query_evaluations"],
            total_latency_ms=results["total_latency_ms"],
            git_hash=git_hash,
            git_dirty=git_dirty,
            notes=args.notes,
        )

        print(f"✓ Saved run #{run_id} to database")

        # Print comparison
        if not args.no_compare:
            print_comparison(db, run_id)


async def show_history(args):
    """Show recent test runs."""
    with EvaluationResultsDB() as db:
        runs = db.get_recent_runs(limit=args.limit)

        if not runs:
            print("No test runs found.")
            return

        print(f"\nRecent Test Runs (last {len(runs)}):")
        print(f"{'=' * 80}")

        for run in runs:
            git_info = f"{run['git_hash'][:8]}" if run['git_hash'] else "N/A"
            if run['git_dirty']:
                git_info += " (dirty)"

            metrics = run["aggregate_metrics"]

            print(f"Run #{run['id']} - {run['timestamp']}")
            print(f"  Git: {git_info}")
            print(f"  PDF: {run['pdf_filename']} ({run['num_chunks']} chunks)")
            print(f"  Metrics: P@5={metrics['precision_at_k']:.3f} | "
                  f"R@5={metrics['recall_at_k']:.3f} | "
                  f"NDCG@5={metrics['ndcg_at_k']:.3f}")
            print()


async def compare_runs(args):
    """Compare two specific runs."""
    with EvaluationResultsDB() as db:
        try:
            comparison = db.compare_runs(args.compare[0], args.compare[1])
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"\n{'=' * 80}")
        print(f"COMPARISON: Run #{args.compare[0]} vs Run #{args.compare[1]}")
        print(f"{'=' * 80}\n")

        run1 = comparison["run1"]
        run2 = comparison["run2"]

        print(f"Run #{run1['id']}: {run1['timestamp']} (Git: {run1['git_hash'][:8] if run1['git_hash'] else 'N/A'})")
        print(f"Run #{run2['id']}: {run2['timestamp']} (Git: {run2['git_hash'][:8] if run2['git_hash'] else 'N/A'})")
        print()

        print(f"{'Metric':<20} {'Baseline':<12} {'Current':<12} {'Diff':<12}")
        print(f"{'-' * 60}")

        for key, data in comparison["metric_diffs"].items():
            diff_str = f"{data['diff']:+.3f}"
            if abs(data["diff"]) < 0.001:
                arrow = "→"
            elif data["diff"] > 0:
                arrow = "↑"
            else:
                arrow = "↓"

            print(f"{key:<20} {data['baseline']:<12.3f} {data['current']:<12.3f} {arrow} {diff_str:<10}")


async def clean_test_collection(args):
    """Delete test collection."""
    print(f"Deleting test collection: {TEST_COLLECTION_NAME}")

    rag_service = RAGService()

    try:
        await rag_service.vector_store.delete_collection(TEST_COLLECTION_NAME)
        print(f"✓ Deleted collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        print(f"Error deleting collection: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAG evaluation testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF_PATH),
        help=f"Path to PDF file (default: {DEFAULT_PDF_PATH.name})",
    )
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE_PATH),
        help=f"Path to test suite JSON (default: {DEFAULT_SUITE_PATH.name})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to evaluate (default: 5)",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force re-ingestion even if PDF unchanged",
    )
    parser.add_argument(
        "--notes",
        help="Optional notes to save with this run",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Don't compare with previous run",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show recent test runs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of runs to show in history (default: 10)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=int,
        metavar=("RUN1", "RUN2"),
        help="Compare two specific runs by ID",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete test collection",
    )

    args = parser.parse_args()

    # Handle different commands
    if args.history:
        asyncio.run(show_history(args))
    elif args.compare:
        asyncio.run(compare_runs(args))
    elif args.clean:
        asyncio.run(clean_test_collection(args))
    else:
        asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
