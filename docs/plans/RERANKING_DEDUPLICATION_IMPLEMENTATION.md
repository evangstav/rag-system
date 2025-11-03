# Reranking & Deduplication Implementation Plan

**Status:** Ready to implement after hybrid search PR is merged
**Estimated Time:** 3-4 days
**Expected Improvement:** +40-50% in Precision@5 and NDCG@5
**Last Updated:** 2025-10-30 (merged with main branch updates)

> **Note:** The main branch has been updated with token-based chunking configuration (`chunk_size_tokens`, `chunk_overlap_tokens`, `tokenizer`) that was originally identified as a critical issue in the proposal. This is now ✅ **already fixed**, so Phase 1.1 tasks can be simplified to focus only on reranking settings.

---

## Overview

This plan implements **Phase 1** from the RAG Retrieval Improvement Proposal:
- **Cross-Encoder Reranking** - Re-scores initial candidates for better relevance
- **MMR Deduplication** - Removes redundant chunks while preserving diversity
- **Feature Flags** - Gradual rollout with ability to disable if needed

**Why This Order:**
1. ✅ Independent of hybrid search work (no conflicts)
2. ✅ Biggest quality improvement for effort (~40-50% gain)
3. ✅ Can be tested and deployed separately
4. ✅ Provides foundation for future query enhancement features

---

## Architecture Diagram

```
Current Pipeline:
Query → Embed → Vector Search (top 5) → Format → LLM

New Pipeline:
Query → Embed → Vector Search (top 40) → Rerank (top 15) → MMR Dedupe (top 5-8) → Format → LLM
         ↓              ↓                     ↓                  ↓
      100ms          80ms                 150ms              30ms

Total Latency: 120ms → 360ms (+240ms, acceptable)
```

---

## Implementation Sequence

### Phase 1: Configuration & Dependencies (Day 1, Morning)

#### 1.1 Update Configuration Settings

**File:** `backend/app/config.py`

**Add these settings:**
```python
class Settings(BaseSettings):
    # ... existing settings ...

    # ===== RERANKING SETTINGS =====
    enable_reranking: bool = Field(
        default=True,
        env="ENABLE_RERANKING",
        description="Enable cross-encoder reranking for improved relevance"
    )

    reranker_model: str = Field(
        default="mixedbread-ai/mxbai-rerank-large-v1",
        env="RERANKER_MODEL",
        description="Model for reranking. Options: mixedbread-ai/mxbai-rerank-large-v1, "
                    "BAAI/bge-reranker-v2-m3, flashrank, cohere-rerank-v3"
    )

    initial_retrieval_k: int = Field(
        default=40,
        env="INITIAL_RETRIEVAL_K",
        description="Number of candidates to retrieve before reranking (20-50 recommended)"
    )

    rerank_top_k: int = Field(
        default=15,
        env="RERANK_TOP_K",
        description="Number of results to keep after reranking (10-20 recommended)"
    )

    # ===== DEDUPLICATION SETTINGS =====
    enable_mmr: bool = Field(
        default=True,
        env="ENABLE_MMR",
        description="Enable MMR (Maximal Marginal Relevance) for result diversification"
    )

    mmr_lambda: float = Field(
        default=0.7,
        env="MMR_LAMBDA",
        description="MMR lambda parameter. 1.0=pure relevance, 0.0=pure diversity. "
                    "0.7 recommended for balanced results"
    )

    final_retrieval_k: int = Field(
        default=8,
        env="FINAL_RETRIEVAL_K",
        description="Final number of results to return after all post-processing (5-10 recommended)"
    )

    # ===== OPTIONAL: API-BASED RERANKERS =====
    cohere_api_key: Optional[str] = Field(
        default=None,
        env="COHERE_API_KEY",
        description="Cohere API key for Cohere Rerank v3 (optional, paid service)"
    )

    # ===== LEGACY SETTING (keep for backward compatibility) =====
    max_rag_results: int = Field(
        default=5,
        env="MAX_RAG_RESULTS",
        description="DEPRECATED: Use final_retrieval_k instead"
    )
```

**Validation:**
```python
@validator("mmr_lambda")
def validate_mmr_lambda(cls, v):
    if not 0 <= v <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1")
    return v

@validator("initial_retrieval_k")
def validate_initial_retrieval_k(cls, v):
    if v < 5:
        raise ValueError("initial_retrieval_k must be at least 5")
    if v > 100:
        raise ValueError("initial_retrieval_k should not exceed 100 (performance impact)")
    return v
```

#### 1.2 Add Dependencies

**File:** `backend/pyproject.toml`

**Add to dependencies:**
```toml
[tool.poetry.dependencies]
# ... existing dependencies ...

# Reranking
sentence-transformers = "^3.0.0"  # Cross-encoder models

# Deduplication (MMR)
scikit-learn = "^1.4.0"  # For cosine similarity calculations

# Optional: Fast reranking
# flashrank = "^0.2.0"  # Uncomment for FlashRank support

# Optional: API-based reranking
# cohere = "^5.0.0"  # Uncomment for Cohere Rerank API
```

**Install:**
```bash
cd backend
poetry add sentence-transformers scikit-learn
# OR with pip:
pip install sentence-transformers scikit-learn
```

---

### Phase 2: Core Implementation (Day 1, Afternoon)

#### 2.1 Create Reranker Module

**File:** `backend/app/services/rag/reranker.py`

**Implementation checklist:**
- [ ] Define `Reranker` protocol interface
- [ ] Implement `CrossEncoderReranker` class
  - Lazy model loading (avoid startup delay)
  - Async execution with `run_in_executor` (CPU-bound)
  - Batch processing support
  - Score normalization
- [ ] Implement `get_reranker()` factory function
- [ ] Add comprehensive docstrings with examples
- [ ] Include `RerankResult` dataclass for debugging

**Key implementation details:**

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        self.model_name = model_name
        self._model = None  # Lazy load to avoid blocking startup

    @property
    def model(self) -> CrossEncoder:
        """Lazy load model on first use"""
        if self._model is None:
            # Download happens here (first call only)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Main reranking method"""
        # 1. Prepare pairs: [(query, doc1), (query, doc2), ...]
        # 2. Run cross-encoder in thread pool (CPU-bound)
        # 3. Create new SearchResult objects with updated scores
        # 4. Sort by reranker score (NOT original similarity)
        # 5. Return top_k
```

**Testing strategy:**
```python
# Manual test
query = "How to reset password?"
mock_results = [...]
reranker = CrossEncoderReranker()
reranked = await reranker.rerank(query, mock_results, top_k=5)

# Verify:
# - Scores are between 0 and 1
# - Results are sorted descending
# - Original scores preserved in metadata
```

#### 2.2 Create Deduplication Module

**File:** `backend/app/services/rag/deduplication.py`

**Implementation checklist:**
- [ ] Implement `MMRDeduplicator` class
  - Greedy iterative selection algorithm
  - Vectorized cosine similarity calculations
  - Score normalization
  - Lambda parameter tuning
- [ ] Implement `TokenDeduplicator` (simpler, faster alternative)
- [ ] Implement `DocumentDeduplicator` (fallback option)
- [ ] Add comprehensive docstrings with MMR formula

**Key implementation details:**

```python
class MMRDeduplicator:
    def __init__(self, lambda_param: float = 0.7):
        """
        MMR Formula:
        score = λ * relevance - (1-λ) * max_similarity_to_selected

        λ=1.0: Pure relevance (no deduplication)
        λ=0.7: Balanced (recommended)
        λ=0.5: Equal weight
        λ=0.3: High diversity
        """
        self.lambda_param = lambda_param

    async def deduplicate(
        self,
        results: List[SearchResult],
        top_k: int,
        embeddings: List[List[float]],
    ) -> List[SearchResult]:
        """
        Greedy MMR algorithm:
        1. Start with highest-scoring result
        2. For remaining results, calculate MMR score
        3. Select result with highest MMR
        4. Repeat until top_k selected
        """
```

**Performance optimization:**
```python
# Use numpy for vectorized operations (10x faster)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings_np = np.array(embeddings)
similarities = cosine_similarity(
    embeddings_np[idx].reshape(1, -1),
    embeddings_np[selected_indices]
)
```

---

### Phase 3: Integration (Day 2, Morning)

#### 3.1 Update RAGService

**File:** `backend/app/services/rag_service.py`

**Changes:**

1. **Add reranker and deduplicator to __init__:**
```python
class RAGService:
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        # ... existing initialization ...

        # NEW: Initialize reranker and deduplicator
        if self.settings.enable_reranking:
            from app.services.rag.reranker import get_reranker
            self.reranker = get_reranker(self.settings)
        else:
            self.reranker = None

        if self.settings.enable_mmr:
            from app.services.rag.deduplication import MMRDeduplicator
            self.deduplicator = MMRDeduplicator(
                lambda_param=self.settings.mmr_lambda
            )
        else:
            self.deduplicator = None
```

2. **Implement search_with_reranking:**
```python
async def search_with_reranking(
    self,
    query: str,
    collection_name: str,
    final_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    """
    Enhanced search with reranking and deduplication.

    Pipeline:
    1. Retrieve initial_retrieval_k candidates (default 40)
    2. Rerank → top rerank_top_k (default 15)
    3. Deduplicate with MMR → final_k results

    Args:
        query: Search query
        collection_name: Vector collection to search
        final_k: Final number of results (default: settings.final_retrieval_k)
        score_threshold: Minimum similarity score
        filter_conditions: Metadata filters

    Returns:
        List of SearchResult objects, reranked and deduplicated
    """
    final_k = final_k or self.settings.final_retrieval_k

    # Stage 1: Initial retrieval (larger candidate set)
    candidates = await self.search(
        query=query,
        collection_name=collection_name,
        limit=self.settings.initial_retrieval_k,
        score_threshold=score_threshold,
        filter_conditions=filter_conditions,
    )

    if not candidates:
        return []

    # Stage 2: Reranking (if enabled)
    if self.reranker:
        reranked = await self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=self.settings.rerank_top_k,
        )
    else:
        reranked = candidates[:self.settings.rerank_top_k]

    # Stage 3: Deduplication (if enabled)
    if self.deduplicator and len(reranked) > final_k:
        # Get embeddings for MMR
        texts = [r.content for r in reranked]
        embeddings = await self.embedding_provider.embed_batch(texts)

        final_results = await self.deduplicator.deduplicate(
            results=reranked,
            top_k=final_k,
            embeddings=embeddings,
        )
    else:
        final_results = reranked[:final_k]

    return final_results
```

3. **Implement search_multiple_pools_with_reranking:**
```python
async def search_multiple_pools_with_reranking(
    self,
    query: str,
    collection_names: List[str],
    final_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """
    Multi-pool search with reranking and deduplication.

    Strategy:
    1. Search all pools in parallel
    2. Merge and sort by original score
    3. Apply reranking to top candidates
    4. Apply deduplication for final results
    """
    final_k = final_k or self.settings.final_retrieval_k

    # Distribute retrieval budget across pools
    per_pool_limit = max(5, self.settings.initial_retrieval_k // len(collection_names))

    # Search all pools in parallel
    search_tasks = [
        self.search(
            query=query,
            collection_name=collection,
            limit=per_pool_limit,
            score_threshold=score_threshold,
        )
        for collection in collection_names
    ]

    pool_results = await asyncio.gather(*search_tasks)

    # Merge all results
    all_results = []
    for results in pool_results:
        all_results.extend(results)

    # Sort by original score
    all_results.sort(key=lambda x: x.score, reverse=True)

    # Take top candidates for reranking
    candidates = all_results[:self.settings.initial_retrieval_k]

    if not candidates:
        return []

    # Apply reranking
    if self.reranker:
        reranked = await self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=self.settings.rerank_top_k,
        )
    else:
        reranked = candidates[:self.settings.rerank_top_k]

    # Apply deduplication
    if self.deduplicator and len(reranked) > final_k:
        texts = [r.content for r in reranked]
        embeddings = await self.embedding_provider.embed_batch(texts)

        final_results = await self.deduplicator.deduplicate(
            results=reranked,
            top_k=final_k,
            embeddings=embeddings,
        )
    else:
        final_results = reranked[:final_k]

    return final_results
```

#### 3.2 Update Chat API

**File:** `backend/app/api/chat.py`

**Changes in get_rag_context():**

```python
async def get_rag_context(
    query: str,
    user: User,
    db: AsyncSession,
    rag_service: RAGService,
    knowledge_pool_ids: Optional[List[UUID]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """Get RAG context for query"""

    # ... existing code to get collection_names ...

    # REPLACE THIS SECTION:
    # OLD:
    # if knowledge_pool_ids:
    #     results = await rag_service.search_multiple_pools(...)
    # else:
    #     results = await rag_service.search(...)

    # NEW:
    settings = get_settings()

    if len(collection_names) > 1:
        # Multiple pools - use multi-pool reranking
        results = await rag_service.search_multiple_pools_with_reranking(
            query=query,
            collection_names=collection_names,
            final_k=settings.final_retrieval_k,
        )
    elif collection_names:
        # Single pool - use single-pool reranking
        results = await rag_service.search_with_reranking(
            query=query,
            collection_name=collection_names[0],
            final_k=settings.final_retrieval_k,
        )
    else:
        # No pools available
        return "", []

    # ... rest of the function remains the same ...
```

---

### Phase 4: Testing (Day 2, Afternoon)

#### 4.1 Unit Tests

**File:** `backend/tests/services/rag/test_reranker.py`

```python
import pytest
from app.services.rag.reranker import CrossEncoderReranker
from app.services.rag.protocols import SearchResult

@pytest.mark.asyncio
async def test_cross_encoder_reranker():
    """Test basic reranking functionality"""
    reranker = CrossEncoderReranker()

    query = "How to reset password?"

    # Create mock results (intentionally out of order)
    results = [
        SearchResult(
            content="To change your theme, go to Settings > Appearance",
            score=0.75,
            metadata={"source": "doc1"},
            document_id=None,
            chunk_index=0,
        ),
        SearchResult(
            content="To reset your password, navigate to Account Settings and click 'Change Password'",
            score=0.70,  # Lower original score but more relevant
            metadata={"source": "doc2"},
            document_id=None,
            chunk_index=0,
        ),
        SearchResult(
            content="The weather today is sunny",
            score=0.65,
            metadata={"source": "doc3"},
            document_id=None,
            chunk_index=0,
        ),
    ]

    # Rerank
    reranked = await reranker.rerank(query, results, top_k=2)

    # Assertions
    assert len(reranked) == 2
    assert reranked[0].content.startswith("To reset your password")  # Should be first
    assert reranked[0].score > 0.5  # Cross-encoder score should be high
    assert "original_score" in reranked[0].metadata  # Should preserve original


@pytest.mark.asyncio
async def test_reranker_empty_input():
    """Test reranker with empty input"""
    reranker = CrossEncoderReranker()
    result = await reranker.rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_reranker_score_ordering():
    """Verify results are sorted by reranker score"""
    reranker = CrossEncoderReranker()

    query = "Python programming"
    results = [
        SearchResult(f"Content about Python {i}", 0.5, {}, None, i)
        for i in range(10)
    ]

    reranked = await reranker.rerank(query, results, top_k=5)

    # Verify descending order
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)
```

**File:** `backend/tests/services/rag/test_deduplication.py`

```python
import pytest
import numpy as np
from app.services.rag.deduplication import MMRDeduplicator
from app.services.rag.protocols import SearchResult

@pytest.mark.asyncio
async def test_mmr_deduplicator():
    """Test MMR deduplication"""
    deduplicator = MMRDeduplicator(lambda_param=0.7)

    # Create mock results
    results = [
        SearchResult("First document about cats", 0.9, {}, None, 0),
        SearchResult("Second document about cats and felines", 0.85, {}, None, 1),  # Similar to first
        SearchResult("Document about dogs", 0.80, {}, None, 2),  # Different topic
        SearchResult("Another cat document", 0.75, {}, None, 3),  # Similar to first
    ]

    # Create mock embeddings (simulate similarity)
    embeddings = [
        [1.0, 0.8, 0.1, 0.9],  # Cat doc 1
        [0.9, 1.0, 0.2, 0.85], # Cat doc 2 (similar to 1)
        [0.1, 0.2, 1.0, 0.15], # Dog doc (different)
        [0.95, 0.9, 0.1, 1.0], # Cat doc 3 (similar to 1)
    ]

    deduplicated = await deduplicator.deduplicate(
        results=results,
        top_k=2,
        embeddings=embeddings,
    )

    # Assertions
    assert len(deduplicated) == 2
    # Should select first (highest score) and third (most diverse)
    assert "cats" in deduplicated[0].content
    assert "dogs" in deduplicated[1].content


@pytest.mark.asyncio
async def test_mmr_lambda_extremes():
    """Test MMR with extreme lambda values"""

    # Lambda = 1.0 (pure relevance, no diversity)
    pure_relevance = MMRDeduplicator(lambda_param=1.0)

    # Lambda = 0.0 (pure diversity, ignore relevance)
    pure_diversity = MMRDeduplicator(lambda_param=0.0)

    results = [
        SearchResult(f"Doc {i}", 1.0 - i*0.1, {}, None, i)
        for i in range(5)
    ]

    # Create embeddings where all docs are similar
    embeddings = [[1.0, 0.9, 0.9, 0.9, 0.9] for _ in range(5)]

    # Pure relevance should pick top 3 by score
    rel_results = await pure_relevance.deduplicate(results, 3, embeddings)
    assert [r.chunk_index for r in rel_results] == [0, 1, 2]

    # Pure diversity might pick different docs
    div_results = await pure_diversity.deduplicate(results, 3, embeddings)
    assert len(div_results) == 3
```

#### 4.2 Integration Tests

**File:** `backend/tests/services/test_rag_service_reranking.py`

```python
@pytest.mark.asyncio
async def test_search_with_reranking_integration(db_session):
    """Test full reranking pipeline"""
    settings = get_settings()
    settings.enable_reranking = True
    settings.enable_mmr = True

    rag_service = RAGService(settings=settings)

    # ... set up test collection with documents ...

    results = await rag_service.search_with_reranking(
        query="test query",
        collection_name="test_collection",
        final_k=5,
    )

    assert len(results) <= 5
    assert all(isinstance(r, SearchResult) for r in results)
    # Verify scores are from reranker (should have original_score in metadata)
    assert all("original_score" in r.metadata for r in results)
```

#### 4.3 Manual Testing Checklist

- [ ] Test with real documents and queries
- [ ] Verify reranking improves relevance visually
- [ ] Check MMR removes duplicate chunks
- [ ] Test with feature flags disabled (backward compatibility)
- [ ] Test with multiple knowledge pools
- [ ] Verify latency is acceptable (<500ms total)
- [ ] Check memory usage with large candidate sets

---

### Phase 5: Evaluation & Benchmarking (Day 3)

#### 5.1 Capture Baseline Metrics

**Before making any code changes:**

```bash
# Run evaluation to capture baseline
cd backend
pytest tests/evaluation/test_rag_evaluation.py -v --tb=short

# Save results
cp evaluation_results.json baseline_metrics_before_reranking.json
```

**Expected baseline (from current system):**
- Precision@5: ~0.40-0.50
- Recall@5: ~0.35-0.45
- NDCG@5: ~0.45-0.55
- MRR: ~0.50-0.60
- Latency: ~100-150ms

#### 5.2 Create Comparison Script

**File:** `backend/scripts/compare_reranking_approaches.py`

```python
#!/usr/bin/env python3
"""
Compare RAG retrieval with and without reranking.

Usage:
    python scripts/compare_reranking_approaches.py
"""

import asyncio
import json
from typing import Dict, List
from app.services.rag_service import RAGService
from app.config import get_settings
from app.database import get_async_session
from app.evaluation.runner import RAGEvaluationRunner

async def run_comparison():
    """Compare baseline vs reranking vs full pipeline"""

    settings = get_settings()

    # Configuration 1: Baseline (no reranking, no MMR)
    print("\n" + "="*60)
    print("CONFIGURATION 1: Baseline (Pure Vector Search)")
    print("="*60)
    settings.enable_reranking = False
    settings.enable_mmr = False
    baseline_service = RAGService(settings=settings)

    # Configuration 2: Reranking only
    print("\n" + "="*60)
    print("CONFIGURATION 2: With Reranking Only")
    print("="*60)
    settings.enable_reranking = True
    settings.enable_mmr = False
    rerank_service = RAGService(settings=settings)

    # Configuration 3: Full pipeline (reranking + MMR)
    print("\n" + "="*60)
    print("CONFIGURATION 3: Full Pipeline (Reranking + MMR)")
    print("="*60)
    settings.enable_reranking = True
    settings.enable_mmr = True
    full_service = RAGService(settings=settings)

    # Run evaluations
    runner = RAGEvaluationRunner()

    print("\nRunning baseline evaluation...")
    baseline_metrics = await runner.run_evaluation(baseline_service)

    print("\nRunning reranking-only evaluation...")
    rerank_metrics = await runner.run_evaluation(rerank_service)

    print("\nRunning full pipeline evaluation...")
    full_metrics = await runner.run_evaluation(full_service)

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Metric':<25} {'Baseline':<15} {'Rerank Only':<15} {'Full Pipeline':<15} {'Improvement':<15}")
    print("-"*80)

    metrics_to_compare = [
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "mrr",
        "avg_latency_ms",
    ]

    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0)
        rerank_val = rerank_metrics.get(metric, 0)
        full_val = full_metrics.get(metric, 0)

        # Calculate improvement (except for latency)
        if metric == "avg_latency_ms":
            improvement = f"+{full_val - baseline_val:.0f}ms"
        else:
            if baseline_val > 0:
                pct_improvement = ((full_val - baseline_val) / baseline_val) * 100
                improvement = f"+{pct_improvement:.1f}%"
            else:
                improvement = "N/A"

        print(f"{metric:<25} {baseline_val:<15.3f} {rerank_val:<15.3f} {full_val:<15.3f} {improvement:<15}")

    # Save detailed results
    results = {
        "baseline": baseline_metrics,
        "rerank_only": rerank_metrics,
        "full_pipeline": full_metrics,
        "timestamp": "2025-10-26",
        "settings": {
            "initial_retrieval_k": settings.initial_retrieval_k,
            "rerank_top_k": settings.rerank_top_k,
            "final_retrieval_k": settings.final_retrieval_k,
            "mmr_lambda": settings.mmr_lambda,
            "reranker_model": settings.reranker_model,
        }
    }

    with open("reranking_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Detailed results saved to: reranking_comparison_results.json")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_comparison())
```

**Run the comparison:**
```bash
python backend/scripts/compare_reranking_approaches.py
```

#### 5.3 Expected Results

**Target Improvements:**

| Metric | Baseline | With Reranking | Full Pipeline | Improvement |
|--------|----------|----------------|---------------|-------------|
| Precision@5 | 0.45 | 0.60 | 0.68 | **+51%** |
| Recall@5 | 0.38 | 0.47 | 0.54 | **+42%** |
| NDCG@5 | 0.50 | 0.65 | 0.74 | **+48%** |
| MRR | 0.55 | 0.68 | 0.75 | **+36%** |
| Latency | 120ms | 260ms | 320ms | +167% |

**Success Criteria:**
- ✅ Precision@5 improvement: >30%
- ✅ NDCG@5 improvement: >30%
- ✅ Latency increase: <400ms
- ✅ No errors or crashes
- ✅ Memory usage increase: <2GB

---

### Phase 6: Optimization & Tuning (Day 4)

#### 6.1 Hyperparameter Tuning

**Parameters to optimize:**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `initial_retrieval_k` | 40 | 20-50 | Higher = better recall, slower |
| `rerank_top_k` | 15 | 10-20 | Higher = better precision, slower |
| `final_retrieval_k` | 8 | 5-10 | Higher = more context, more tokens |
| `mmr_lambda` | 0.7 | 0.5-0.9 | Higher = relevance, lower = diversity |

**Tuning script:**

```python
# Try different configurations
configs = [
    {"initial_k": 30, "rerank_k": 12, "final_k": 6, "lambda": 0.7},
    {"initial_k": 40, "rerank_k": 15, "final_k": 8, "lambda": 0.7},
    {"initial_k": 50, "rerank_k": 20, "final_k": 10, "lambda": 0.7},
]

best_config = None
best_ndcg = 0

for config in configs:
    metrics = await run_evaluation_with_config(config)
    if metrics["ndcg_at_5"] > best_ndcg:
        best_ndcg = metrics["ndcg_at_5"]
        best_config = config

print(f"Best config: {best_config}")
```

#### 6.2 Performance Optimization

**Optimization checklist:**

- [ ] **Cache reranker model in memory** - Avoid reloading on each request
  ```python
  # Use singleton pattern or lru_cache
  @lru_cache()
  def get_reranker(settings: Settings) -> Reranker:
      return CrossEncoderReranker(settings.reranker_model)
  ```

- [ ] **Batch reranking calls** - Process multiple queries together if possible

- [ ] **Use GPU for reranking** - 4x faster than CPU
  ```python
  # Detect GPU availability
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = CrossEncoder(model_name, device=device)
  ```

- [ ] **Optimize MMR calculations** - Use numpy for vectorization
  ```python
  # Already implemented in deduplication.py
  embeddings_np = np.array(embeddings)
  similarities = cosine_similarity(...)
  ```

- [ ] **Profile slow operations**
  ```bash
  python -m cProfile -o profile.stats backend/app/main.py
  python -m pstats profile.stats
  ```

#### 6.3 Memory Optimization

**If memory usage is high:**

1. **Use smaller reranker model:**
   ```python
   # Switch from mxbai-rerank-large-v1 to MiniLM (6x smaller)
   reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
   ```

2. **Reduce candidate pool:**
   ```python
   initial_retrieval_k = 20  # Down from 40
   ```

3. **Clear model cache periodically:**
   ```python
   import gc
   import torch

   torch.cuda.empty_cache()
   gc.collect()
   ```

---

### Phase 7: Documentation & Deployment (Day 4, Afternoon)

#### 7.1 Update Documentation

**File:** `backend/README.md` or `CLAUDE.md`

**Add section:**

```markdown
## RAG Reranking & Deduplication

The RAG system uses a multi-stage retrieval pipeline for improved relevance:

1. **Initial Retrieval** - Fetch 40 candidates via vector search
2. **Cross-Encoder Reranking** - Re-score with sentence-transformers
3. **MMR Deduplication** - Remove redundant chunks

### Configuration

Set these environment variables to tune behavior:

```bash
# Enable/disable features
ENABLE_RERANKING=true
ENABLE_MMR=true

# Retrieval parameters
INITIAL_RETRIEVAL_K=40      # Candidates for reranking
RERANK_TOP_K=15             # Results after reranking
FINAL_RETRIEVAL_K=8         # Final results after MMR

# MMR diversity parameter
MMR_LAMBDA=0.7              # 0.5-0.9 range (higher = more relevance)

# Reranker model
RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v1
```

### Performance

- **Precision@5:** +51% improvement over baseline
- **NDCG@5:** +48% improvement
- **Latency:** +200-300ms per query
- **Memory:** +2GB for reranker model

### Troubleshooting

**High latency:**
- Reduce `INITIAL_RETRIEVAL_K` to 20-30
- Use smaller model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Enable GPU acceleration (4x faster)

**Low diversity:**
- Decrease `MMR_LAMBDA` to 0.5-0.6
- Increase `FINAL_RETRIEVAL_K` to 10

**Disable reranking:**
```bash
ENABLE_RERANKING=false
ENABLE_MMR=false
```
```

#### 7.2 Create Migration Guide

**File:** `RERANKING_MIGRATION_GUIDE.md`

```markdown
# Reranking Migration Guide

## Pre-Deployment Checklist

- [ ] Backup current environment variables
- [ ] Run baseline evaluation and save results
- [ ] Test on staging environment first
- [ ] Monitor Qdrant and backend resource usage
- [ ] Prepare rollback plan

## Deployment Steps

1. **Update Dependencies**
   ```bash
   cd backend
   poetry add sentence-transformers scikit-learn
   ```

2. **Download Reranker Model (Optional Pre-Download)**
   ```python
   from sentence_transformers import CrossEncoder
   model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")
   # Model cached at ~/.cache/huggingface/
   ```

3. **Update Environment Variables**
   ```bash
   # Add to .env
   ENABLE_RERANKING=true
   ENABLE_MMR=true
   INITIAL_RETRIEVAL_K=40
   RERANK_TOP_K=15
   FINAL_RETRIEVAL_K=8
   MMR_LAMBDA=0.7
   RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v1
   ```

4. **Deploy Code**
   ```bash
   git pull
   docker-compose restart backend
   ```

5. **Monitor Metrics**
   - Check `/health` endpoint
   - Monitor latency in logs
   - Watch Qdrant CPU/memory
   - Check backend memory usage

## Rollback Plan

If issues occur:

```bash
# Disable reranking via environment
ENABLE_RERANKING=false
ENABLE_MMR=false

# Restart backend
docker-compose restart backend
```

Or rollback code:
```bash
git revert <commit-hash>
docker-compose restart backend
```

## Gradual Rollout (Recommended)

1. **Week 1:** Deploy with `ENABLE_RERANKING=false` (code only)
2. **Week 2:** Enable for 10% of users (A/B test)
3. **Week 3:** Enable for 50% of users
4. **Week 4:** Enable for 100% of users

Feature flag implementation:
```python
# In chat.py
if user.id % 10 == 0:  # 10% of users
    use_reranking = True
else:
    use_reranking = False
```
```

#### 7.3 Environment Variable Template

**File:** `backend/.env.example`

```bash
# ... existing variables ...

# ===== RAG RERANKING & DEDUPLICATION =====
# Enable cross-encoder reranking for improved relevance
ENABLE_RERANKING=true

# Enable MMR (Maximal Marginal Relevance) for result diversification
ENABLE_MMR=true

# Number of candidates to retrieve before reranking (20-50 recommended)
INITIAL_RETRIEVAL_K=40

# Number of results to keep after reranking (10-20 recommended)
RERANK_TOP_K=15

# Final number of results after deduplication (5-10 recommended)
FINAL_RETRIEVAL_K=8

# MMR lambda: 1.0=pure relevance, 0.0=pure diversity (0.7 recommended)
MMR_LAMBDA=0.7

# Reranker model (options below)
# - mixedbread-ai/mxbai-rerank-large-v1 (best quality, ~1GB)
# - BAAI/bge-reranker-v2-m3 (multilingual)
# - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, small)
RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v1

# Optional: Cohere Rerank API (requires paid account)
# COHERE_API_KEY=your_api_key_here
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Code review completed
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Baseline metrics captured
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Dependencies added to pyproject.toml

### Deployment

- [ ] Deploy to staging environment
- [ ] Run comparison script on staging
- [ ] Verify improvements meet targets (>30%)
- [ ] Load test with expected traffic
- [ ] Monitor resource usage
- [ ] Deploy to production with feature flag disabled
- [ ] Gradually enable for 10% → 50% → 100% of users

### Post-Deployment

- [ ] Monitor latency metrics
- [ ] Monitor error rates
- [ ] Collect user feedback
- [ ] Run A/B test analysis
- [ ] Document actual improvements
- [ ] Update runbook with troubleshooting tips

---

## Success Metrics

### Must-Have (Go/No-Go)
- ✅ Precision@5 improvement: >30%
- ✅ NDCG@5 improvement: >30%
- ✅ No increase in error rate
- ✅ Latency p95 < 600ms

### Nice-to-Have
- ✅ Precision@5 improvement: >40%
- ✅ User satisfaction increase: >20%
- ✅ LLM citation rate increase: >10%
- ✅ Latency p95 < 400ms

---

## Troubleshooting

### Issue: High Memory Usage

**Symptoms:** Backend using >8GB RAM

**Solutions:**
1. Use smaller model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
2. Reduce `initial_retrieval_k` to 20-30
3. Add memory limits in docker-compose.yml

### Issue: Slow Response Times

**Symptoms:** Query latency >1 second

**Solutions:**
1. Enable GPU acceleration
2. Use FlashRank instead of CrossEncoder
3. Reduce `rerank_top_k` to 10
4. Cache frequent queries in Redis

### Issue: No Quality Improvement

**Symptoms:** Metrics same as baseline

**Solutions:**
1. Verify reranker is actually being called (add logging)
2. Check if `enable_reranking` is true
3. Ensure `initial_retrieval_k` is large enough (>20)
4. Review retrieved candidates - might be all irrelevant

### Issue: Too Much Diversity (Lost Relevance)

**Symptoms:** MMR removing relevant results

**Solutions:**
1. Increase `mmr_lambda` to 0.8-0.9
2. Disable MMR: `ENABLE_MMR=false`
3. Increase `final_retrieval_k` to include more results

---

## Timeline Summary

| Day | Phase | Tasks | Deliverables |
|-----|-------|-------|--------------|
| 1 AM | Config & Dependencies | Add settings, install packages | Updated config.py, pyproject.toml |
| 1 PM | Core Implementation | Create reranker.py, deduplication.py | New modules with docstrings |
| 2 AM | Integration | Update RAGService, chat.py | Integrated pipeline |
| 2 PM | Testing | Write unit tests, manual testing | Test files, validation |
| 3 | Evaluation | Run baseline, comparison, analysis | Metrics comparison report |
| 4 AM | Optimization | Tune hyperparameters, profile | Optimized configuration |
| 4 PM | Documentation | Write guides, update README | Migration guide, docs |

**Total:** 3-4 days for complete implementation and validation

---

## Next Steps After Implementation

Once reranking & deduplication are deployed:

1. **Monitor for 1-2 weeks** - Collect production metrics
2. **Analyze user feedback** - Survey or implicit feedback
3. **Tune hyperparameters** - Based on real usage patterns
4. **Plan Phase 2** - Merge with hybrid search PR
5. **Consider Phase 3** - Query enhancement (HyDE, multi-query)

---

## Questions or Issues?

Refer to:
- `RAG_RETRIEVAL_IMPROVEMENT_PROPOSAL.md` - Full research and context
- `CLAUDE.md` - Project overview and patterns
- Reranking research papers in proposal references section
