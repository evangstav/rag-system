# End-to-End RAG Testing System

Automated testing framework for evaluating RAG (Retrieval-Augmented Generation) performance over time.

## Overview

This system provides:
- **Automated PDF ingestion** into a dedicated test collection
- **Evaluation suite** execution with standard IR metrics
- **SQLite tracking** of all test runs with git correlation
- **Comparison tools** to measure impact of changes
- **Simple Make commands** for easy execution

## Architecture

```
backend/
├── scripts/
│   └── e2e_rag_test.py          # Main orchestrator
├── tests/
│   ├── data/
│   │   ├── my_suite.json        # Test queries (12 queries)
│   │   ├── *.pdf                # Test documents (gitignored)
│   │   └── README.md
│   ├── utils/
│   │   ├── results_db.py        # SQLite wrapper
│   │   └── __init__.py
│   └── results/
│       ├── schema.sql           # DB schema
│       ├── evaluation_history.db # Results database (gitignored)
│       └── .gitignore
└── Makefile                     # Convenient commands
```

## Quick Start

### 1. Setup Test Data

```bash
# Add your PDF to test data directory
cp /path/to/document.pdf backend/tests/data/How\ to\ Train\ Guide.pdf
```

### 2. Run Test

```bash
cd backend
make test-rag
```

**Output:**
```
Creating test collection: test_rag_evaluation
Loading PDF: How to Train Guide.pdf
Splitting document...
Created 148 chunks
Generating embeddings...
Storing in Qdrant (test_rag_evaluation)...
✓ Ingested 148 chunks

Running evaluation suite...
============================================================
Running evaluation suite
Retriever: Baseline RAG Retriever
Test queries: 12
k=5
============================================================

[1/12] Evaluating: What are the core principles of effective...
    P@5=0.80 | R@5=0.80 | NDCG@5=0.92 | Latency=45ms
...

✓ Saved run #1 to database

============================================================
COMPARISON: Run #0 vs Run #1
============================================================
  precision_at_k       0.850 (→ +0.000)
  recall_at_k          0.900 (→ +0.000)
  ndcg_at_k            0.880 (→ +0.000)
```

## Commands

### Run Test
```bash
# Default (uses tests/data/How to Train Guide.pdf)
make test-rag

# Custom PDF
make test-rag PDF=tests/data/my-document.pdf

# Force re-ingestion (even if PDF unchanged)
make test-rag-force

# With custom k value
make test-rag K=10
```

### View History
```bash
# Show last 10 runs
make test-rag-history

# Output:
# Recent Test Runs (last 10):
# ============================================================
# Run #5 - 2025-10-24T15:30:00
#   Git: d3709c5 (dirty)
#   PDF: How to Train Guide.pdf (148 chunks)
#   Metrics: P@5=0.850 | R@5=0.900 | NDCG@5=0.880
```

### Compare Runs
```bash
# Compare two specific runs
make test-rag-compare RUNS='4 5'

# Output shows metric changes:
# Metric               Baseline     Current      Diff
# ------------------------------------------------------------
# precision_at_k       0.800        0.850        ↑ +0.050
# recall_at_k          0.900        0.900        → +0.000
# ndcg_at_k            0.900        0.880        ↓ -0.020
```

### Clean Up
```bash
# Delete test collection from Qdrant
make test-rag-clean
```

## Advanced Usage

### Direct Script Usage

```bash
# Run with custom parameters
python scripts/e2e_rag_test.py \
    --pdf tests/data/my-doc.pdf \
    --suite tests/data/my_suite.json \
    --k 10 \
    --notes "Testing new chunking strategy"

# Show history with more runs
python scripts/e2e_rag_test.py --history --limit 20

# Compare specific runs
python scripts/e2e_rag_test.py --compare 4 5

# Force re-ingestion
python scripts/e2e_rag_test.py --force-ingest
```

### Query Results Database

```python
from tests.utils.results_db import EvaluationResultsDB

with EvaluationResultsDB() as db:
    # Get latest run
    latest = db.get_latest_run()
    print(latest['aggregate_metrics'])

    # Get runs for specific commit
    runs = db.get_runs_by_git_hash('d3709c5')

    # Compare two runs
    comparison = db.compare_runs(4, 5)
    print(comparison['metric_diffs'])
```

## Test Collection

- **Collection Name:** `test_rag_evaluation`
- **User ID:** `test_user_00000000`
- **Isolation:** Completely separate from production data
- **Persistence:** Collection persists between runs
- **Reingestion:** Only occurs if PDF changes or `--force-ingest` used

## Metrics Tracked

### Aggregate Metrics
- **Precision@K**: Fraction of retrieved docs that are relevant
- **Recall@K**: Fraction of relevant docs that were retrieved
- **MRR**: Mean Reciprocal Rank of first relevant doc
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Latency**: Query response time (avg, p50, p95)

### Breakdowns
- **By Query Type**: factual, technical, procedural, optimization, etc.
- **By Difficulty**: easy, medium, hard

### Per-Query Results
- Individual metrics for each test query
- Retrieved documents
- Latency per query

## Database Schema

```sql
evaluation_runs:
  - id, timestamp, git_hash, git_dirty
  - pdf_filename, pdf_path, pdf_hash
  - collection_name, num_chunks
  - config (JSON): chunk_size, model, etc.
  - aggregate_metrics (JSON)
  - breakdowns (JSON)
  - query_results (JSON)
  - total_latency_ms, avg_latency_ms
  - notes
```

## Typical Workflow

### 1. Baseline
```bash
# Establish baseline on main branch
git checkout main
make test-rag
# Run #1: P@5=0.850, R@5=0.900, NDCG@5=0.880
```

### 2. Experiment
```bash
# Make changes (e.g., adjust chunk size in config.py)
vim app/config.py  # Change CHUNK_SIZE from 1000 to 500

# Re-test (will force re-ingestion since config changed)
make test-rag-force
# Run #2: P@5=0.870 ↑, R@5=0.920 ↑, NDCG@5=0.900 ↑
```

### 3. Compare
```bash
# Compare the two runs
make test-rag-compare RUNS='1 2'

# Analyze improvements:
# - Precision improved by +2.0%
# - Recall improved by +2.2%
# - NDCG improved by +2.3%
```

### 4. Commit
```bash
# If results improved, commit the change
git add app/config.py
git commit -m "Reduce chunk size to 500 for better retrieval"
```

## Benefits

✅ **Reproducible**: Same PDF + same code = same results
✅ **Fast**: Reuses embeddings if PDF unchanged
✅ **Isolated**: Won't interfere with production data
✅ **Trackable**: Every run linked to git commit
✅ **Comparable**: Instantly see impact of changes
✅ **Automated**: No manual UI interaction needed

## Troubleshooting

### PDF Not Found
```bash
Error: PDF not found: tests/data/How to Train Guide.pdf
```
**Solution:** Add your PDF to `backend/tests/data/`

### Collection Already Exists
If you want to start fresh:
```bash
make test-rag-clean
make test-rag
```

### Import Errors
Make sure you're running from the backend directory:
```bash
cd backend
make test-rag
```

### Qdrant Connection Error
Ensure Qdrant is running:
```bash
docker ps | grep qdrant
# If not running:
docker-compose up -d qdrant
```

## Future Enhancements

Potential additions:
- [ ] Support for multiple PDFs in a single test
- [ ] Parallel evaluation across different configurations
- [ ] Automatic regression detection with alerts
- [ ] Export results to CSV/HTML reports
- [ ] Integration with CI/CD pipelines
- [ ] Support for custom metrics/scorers
- [ ] Web dashboard for visualizing trends

## See Also

- `tests/data/my_suite.json` - Test query definitions
- `app/evaluation/` - Evaluation framework
- `app/services/rag_service.py` - RAG implementation
