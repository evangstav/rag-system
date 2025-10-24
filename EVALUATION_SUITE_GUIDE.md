# RAG Evaluation Suite Guide

A modular, implementation-agnostic framework for evaluating and comparing RAG systems.

## 🎯 Overview

This evaluation suite allows you to:
- **Define test cases once**, run them against any implementation
- **Compare different approaches** (baseline vs hybrid vs reranked, etc.)
- **Track improvements** with quantitative metrics
- **Iterate quickly** on RAG improvements

## 📁 Architecture

```
backend/
├── app/evaluation/
│   ├── protocols.py      # Interfaces (RAGRetriever, TestQuery, etc.)
│   ├── metrics.py        # Retrieval metrics (precision, recall, NDCG, etc.)
│   ├── runner.py         # Test suite execution engine
│   ├── adapters.py       # Wrappers for different implementations
│   ├── loader.py         # Load/save test suites
│   └── comparison.py     # Compare multiple implementations
├── scripts/
│   └── run_rag_evaluation.py  # CLI tool
└── tests/data/
    └── test_suites/      # Your test query files
```

### Key Concepts

**1. Protocol-Based Design**
- Any retriever that implements `RAGRetriever` protocol can be evaluated
- No coupling to specific implementations
- Easy to add new strategies

**2. Adapters**
- Wrap your RAG implementations to conform to protocol
- E.g., `BaselineRAGAdapter`, `HybridSearchAdapter`, `RerankedAdapter`

**3. Test Suites**
- JSON files defining test queries and expected relevant docs
- Reusable across all implementations

**4. Metrics**
- Precision@K, Recall@K, MRR, NDCG@K, MAP
- Latency (avg, p50, p95)
- Breakdowns by query type and difficulty

---

## 🚀 Quick Start

### Step 1: Create a Test Suite

```bash
# Create example test suite
python scripts/run_rag_evaluation.py create-example \
  --output tests/data/my_test_suite.json
```

This creates a JSON file like:

```json
{
  "name": "Example Test Suite",
  "description": "Sample test suite for RAG evaluation",
  "queries": [
    {
      "id": "q1",
      "query": "How do I reset my password?",
      "relevant_document_ids": ["doc_password_reset", "doc_account_settings"],
      "collection_name": "support_docs",
      "query_type": "factual",
      "difficulty": "easy"
    },
    {
      "id": "q2",
      "query": "What are the differences between Pro and Enterprise plans?",
      "relevant_document_ids": ["doc_pricing"],
      "collection_name": "support_docs",
      "query_type": "comparison",
      "difficulty": "medium"
    }
  ]
}
```

### Step 2: Upload Test Documents

Make sure your test documents are uploaded to the knowledge pools referenced in the test suite.

```python
# Upload your test documents
await rag_service.ingest_document(
    source="path/to/password_reset_guide.pdf",
    collection_name="support_docs",
    document_id=UUID("doc_password_reset"),
)
```

### Step 3: Run Baseline Evaluation

```bash
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/my_test_suite.json \
  --implementation baseline \
  --output results/baseline_20250124.json
```

Output:
```
============================================================
Running evaluation suite
Retriever: Baseline (Semantic Search Only)
Test queries: 3
k=5
============================================================

[1/3] Evaluating: How do I reset my password?...
    P@5=0.40 | R@5=1.00 | NDCG@5=0.63 | Latency=145ms
[2/3] Evaluating: What are the differences between Pro and ...
    P@5=0.20 | R@5=1.00 | NDCG@5=0.43 | Latency=132ms
[3/3] Evaluating: How does JWT authentication work in the API?...
    P@5=0.40 | R@5=1.00 | NDCG@5=0.61 | Latency=158ms

============================================================
EVALUATION SUMMARY
============================================================
Retriever: Baseline (Semantic Search Only)
Queries evaluated: 3

Aggregate Metrics (k=5):
  Metric                              Value
  ----------------------------------------
  precision_at_k                      0.333
  recall_at_k                         1.000
  mrr                                 0.778
  ndcg_at_k                           0.557
  map_score                           0.778
  avg_latency_ms                    145.0ms
  p50_latency_ms                    145.0ms
  p95_latency_ms                    158.0ms
  avg_top_score                       0.823

Breakdown by Query Type:
  factual                             0.630
  comparison                          0.430
  technical                           0.610

Breakdown by Difficulty:
  easy                                0.630
  medium                              0.430
  hard                                0.610
============================================================

Results exported to: results/baseline_20250124.json
```

---

## 🔬 Evaluating Different Implementations

### Test Hybrid Search

```bash
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/my_test_suite.json \
  --implementation hybrid \
  --hybrid-alpha 0.7 \
  --output results/hybrid_20250124.json
```

### Test with Reranking

```bash
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/my_test_suite.json \
  --implementation reranked \
  --output results/reranked_20250124.json
```

### Test Custom Configuration

Create a config file `configs/custom.json`:

```json
{
  "score_threshold": 0.6,
  "limit": 10
}
```

Run evaluation:

```bash
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/my_test_suite.json \
  --implementation custom \
  --config configs/custom.json \
  --name "High Threshold (0.6)" \
  --output results/custom_20250124.json
```

---

## 📊 Comparing Implementations

After running multiple evaluations, compare them:

```bash
python scripts/run_rag_evaluation.py compare \
  --baseline results/baseline_20250124.json \
  --comparisons results/hybrid_20250124.json results/reranked_20250124.json \
  --output results/comparison_report.json
```

Output:
```
================================================================================
COMPARISON REPORT
================================================================================

Baseline: Baseline (Semantic Search Only)

────────────────────────────────────────────────────────────────────────────────
Comparing: Hybrid Search (α=0.7)
────────────────────────────────────────────────────────────────────────────────

Metric                    Baseline          New       Change        %
────────────────────────────────────────────────────────────────────────────────
precision_at_k               0.333        0.467       +0.133   +40.0%
recall_at_k                  1.000        1.000       +0.000    +0.0%
mrr                          0.778        0.889       +0.111   +14.3%
ndcg_at_k                    0.557        0.689       +0.132   +23.7%
map_score                    0.778        0.889       +0.111   +14.3%
avg_latency_ms             145.0ms      178.0ms      +33.0ms   +22.8%
p50_latency_ms             145.0ms      175.0ms      +30.0ms   +20.7%
p95_latency_ms             158.0ms      189.0ms      +31.0ms   +19.6%
avg_top_score                0.823        0.891       +0.068    +8.3%

────────────────────────────────────────────────────────────────────────────────
Comparing: Reranked + Baseline (Semantic Search Only)
────────────────────────────────────────────────────────────────────────────────

Metric                    Baseline          New       Change        %
────────────────────────────────────────────────────────────────────────────────
precision_at_k               0.333        0.533       +0.200   +60.0%
recall_at_k                  1.000        1.000       +0.000    +0.0%
mrr                          0.778        0.944       +0.167   +21.4%
ndcg_at_k                    0.557        0.751       +0.194   +34.8%
map_score                    0.778        0.944       +0.167   +21.4%
avg_latency_ms             145.0ms      267.0ms     +122.0ms   +84.1%
p50_latency_ms             145.0ms      265.0ms     +120.0ms   +82.8%
p95_latency_ms             158.0ms      289.0ms     +131.0ms   +82.9%
avg_top_score                0.823        0.912       +0.089   +10.8%

================================================================================
```

### Generate Summary Table

Create a markdown summary of all results:

```bash
python scripts/run_rag_evaluation.py summary \
  --results results/*.json \
  --output EVALUATION_RESULTS.md
```

Output (`EVALUATION_RESULTS.md`):

```markdown
# RAG Evaluation Results

Compared 3 implementations

## Summary

| Implementation | Precision@5 | Recall@5 | NDCG@5 | MRR | Avg Latency |
|---------------|-------------|----------|--------|-----|-------------|
| Baseline (Semantic Search Only) | 0.333 | 1.000 | 0.557 | 0.778 | 145ms |
| Hybrid Search (α=0.7)           | 0.467 | 1.000 | 0.689 | 0.889 | 178ms |
| Reranked + Baseline             | 0.533 | 1.000 | 0.751 | 0.944 | 267ms |
```

---

## 🧪 Creating Custom Test Suites

### Manual Creation

Create `tests/data/support_queries.json`:

```json
{
  "name": "Support Documentation Test Suite",
  "description": "Tests retrieval quality for customer support queries",
  "queries": [
    {
      "id": "support_001",
      "query": "My account was hacked, what should I do?",
      "relevant_document_ids": ["doc_security_breach", "doc_account_recovery"],
      "collection_name": "support_docs",
      "query_type": "urgent",
      "difficulty": "hard",
      "description": "Security incident requiring multiple steps",
      "expected_answer_keywords": ["password", "2FA", "contact support", "secure"]
    },
    {
      "id": "support_002",
      "query": "How do I export my data?",
      "relevant_document_ids": ["doc_data_export", "doc_gdpr"],
      "collection_name": "support_docs",
      "query_type": "factual",
      "difficulty": "easy",
      "expected_answer_keywords": ["export", "download", "settings"]
    }
  ]
}
```

### Programmatic Creation

```python
from app.evaluation.protocols import TestQuery
from app.evaluation.loader import TestSuiteLoader

# Create test queries programmatically
test_queries = [
    TestQuery(
        id="custom_001",
        query="How does authentication work?",
        relevant_document_ids=["doc_auth_guide"],
        collection_name="tech_docs",
        query_type="technical",
        difficulty="medium",
    ),
    # Add more...
]

# Save to file
TestSuiteLoader.save_to_json(
    test_queries,
    "tests/data/my_custom_suite.json",
    metadata={"name": "Custom Suite", "description": "My test suite"}
)
```

---

## 🔧 Creating Custom Adapters

Want to test a completely new retrieval strategy? Create an adapter:

```python
# app/evaluation/adapters.py

class MyCustomRetriever:
    """Your custom retrieval implementation."""

    def __init__(self, rag_service: RAGService, my_param: float):
        self.rag_service = rag_service
        self.my_param = my_param
        self.name = f"My Custom Retriever (param={my_param})"

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Implement your custom retrieval logic here."""

        # Your custom logic
        # ...

        # Return RetrievalResult objects
        return results

    async def get_name(self) -> str:
        return self.name
```

Then use it:

```python
# In scripts/run_rag_evaluation.py, add to the implementation choices

elif implementation == "my_custom":
    retriever = MyCustomRetriever(rag_service, my_param=0.5)
    config = {"type": "my_custom", "my_param": 0.5}
```

---

## 📈 Understanding Metrics

### Precision@K
**What**: Fraction of retrieved documents that are relevant

**Formula**: (# relevant in top-K) / K

**Interpretation**:
- 1.0 = Perfect (all results relevant)
- 0.5 = Half of results are relevant
- <0.3 = Poor precision, too much noise

**When to optimize**: When users complain about irrelevant results

---

### Recall@K
**What**: Fraction of relevant documents that were retrieved

**Formula**: (# relevant in top-K) / (total # relevant)

**Interpretation**:
- 1.0 = Found all relevant docs
- 0.5 = Found half of relevant docs
- <0.3 = Missing most relevant docs

**When to optimize**: When users say "it didn't find what I was looking for"

---

### NDCG@K (Best Overall Metric)
**What**: Ranking quality that rewards relevant docs at top positions

**Interpretation**:
- 1.0 = Perfect ranking
- 0.7-0.9 = Good ranking
- 0.5-0.7 = Acceptable
- <0.5 = Poor ranking

**When to optimize**: Always - this is your primary metric

---

### MRR (Mean Reciprocal Rank)
**What**: How high is the first relevant result?

**Formula**: 1 / (rank of first relevant doc)

**Interpretation**:
- 1.0 = First result is relevant
- 0.5 = Relevant doc at position 2
- 0.33 = Relevant doc at position 3

**When to optimize**: For "quick answer" use cases

---

## 🎯 Workflow Examples

### Example 1: Baseline → Reranking

```bash
# 1. Establish baseline
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/suite.json \
  --implementation baseline \
  --output results/baseline.json

# Baseline NDCG@5: 0.557

# 2. Implement reranking (in code)
# ... implement RerankedAdapter ...

# 3. Re-evaluate
python scripts/run_rag_evaluation.py run \
  --test-suite tests/data/suite.json \
  --implementation reranked \
  --output results/reranked.json

# Reranked NDCG@5: 0.751 (+34.8%)

# 4. Compare
python scripts/run_rag_evaluation.py compare \
  --baseline results/baseline.json \
  --comparisons results/reranked.json

# Decision: Ship reranking! 35% improvement worth the latency cost
```

### Example 2: Testing Different Chunk Sizes

```bash
# Create configs for different chunk sizes
echo '{"chunk_size": 512}' > configs/chunk_512.json
echo '{"chunk_size": 1024}' > configs/chunk_1024.json
echo '{"chunk_size": 2048}' > configs/chunk_2048.json

# Run evaluations
for size in 512 1024 2048; do
  python scripts/run_rag_evaluation.py run \
    --test-suite tests/data/suite.json \
    --implementation custom \
    --config configs/chunk_${size}.json \
    --name "Chunk Size ${size}" \
    --output results/chunk_${size}.json
done

# Compare
python scripts/run_rag_evaluation.py summary \
  --results results/chunk_*.json \
  --output CHUNK_SIZE_COMPARISON.md

# Result: chunk_size=1024 performs best (NDCG@5: 0.689)
```

### Example 3: A/B Testing in Production

```python
# After implementing improvements, run A/B test

# Record which implementation each user sees
user_implementations = {
    "user_123": "baseline",
    "user_456": "reranked",
    # ...
}

# Track metrics
for user_id, implementation in user_implementations.items():
    # Log query results
    # Track thumbs up/down
    # Measure task completion

# After 1 week, analyze
# If reranked shows 20% higher satisfaction → roll out to 100%
```

---

## 🐛 Troubleshooting

### Issue: "No relevant documents found"
**Cause**: Test queries reference document IDs that don't exist

**Fix**: Verify document IDs in your test suite match actual uploaded documents

```bash
# Check what documents exist in collection
python -c "
from app.services.rag_service import RAGService
import asyncio

async def check():
    rag = RAGService()
    stats = await rag.get_collection_stats('support_docs')
    print(stats)

asyncio.run(check())
"
```

---

### Issue: All queries score 0.0
**Cause**: Collection name mismatch

**Fix**: Ensure collection names in test suite match actual knowledge pools

---

### Issue: Evaluation runs but no improvement shown
**Cause**: Implementation isn't actually different from baseline

**Fix**: Add logging to your adapter to verify it's using the new logic

---

## 📚 Best Practices

1. **Start small**: Begin with 10-15 test queries, expand to 30-50
2. **Mix difficulties**: Include easy, medium, hard queries
3. **Cover query types**: Factual, how-to, comparison, troubleshooting
4. **Version results**: Include dates in filenames (results/baseline_20250124.json)
5. **Document configurations**: Save configs used for each run
6. **Track over time**: Run baseline evaluation regularly to detect regressions
7. **Real queries**: Use actual user queries when possible
8. **Iterate**: Measure → Improve → Measure again

---

## 🚀 Next Steps

1. **Create your first test suite** (10 queries)
2. **Run baseline evaluation**
3. **Implement one improvement** (reranking recommended)
4. **Re-evaluate and compare**
5. **Repeat** for other improvements

**Goal**: Build a culture of measurement-driven RAG improvement!

---

## 📖 Additional Resources

- [RAG_IMPROVEMENTS.md](./RAG_IMPROVEMENTS.md) - Detailed improvement recommendations
- [COMPLETE_DEVELOPMENT_GUIDE.md](./COMPLETE_DEVELOPMENT_GUIDE.md) - Full system architecture
- [PROJECT_STATUS.md](./PROJECT_STATUS.md) - Current implementation status
