# RAG System Improvements Analysis

> **Note**: This document contains historical analysis from October 2024. Many issues listed here have since been resolved. See [Current Status](#current-status) section for what's been implemented.

---

## Current Status (November 2024)

### ✅ Implemented Improvements

The following critical improvements have been successfully deployed:

**Architecture**:
```
Document → Load → Chunk → Embed → Store (Qdrant + PostgreSQL BM25) → Hybrid Search → Rerank → Deduplicate → Retrieve
```

**Implemented Features**:
- ✅ **Hybrid Search**: Semantic (Qdrant) + Keyword (PostgreSQL BM25) with RRF fusion
- ✅ **Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- ✅ **Deduplication**: MMR and token-based strategies
- ✅ **PostgreSQL BM25**: Persistent keyword indexing with Full-Text Search
- ✅ **Smart Text Splitting**: Hierarchical boundary-respecting chunking
- ✅ **Multiple Document Loaders**: PDF, DOCX, TXT, MD, Web

**Configuration**:
- **Embedding**: OpenAI `text-embedding-3-small` (1536 dims)
- **Chunking**: 1000 chars with 200 char overlap (smart hierarchical)
- **Vector Store**: Qdrant with cosine similarity
- **Keyword Search**: PostgreSQL Full-Text Search with GIN indexes
- **Search**: Hybrid (semantic + keyword) with optional reranking and deduplication

For detailed architecture documentation, see:
- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [Backend Services Documentation](../architecture/BACKEND_SERVICES.md)

---

## Historical Analysis (October 2024)

### Original Architecture Overview

```
Document → Load → Chunk → Embed → Store (Qdrant) → Search (Cosine) → Retrieve
```

### Original Configuration

- **Embedding**: OpenAI `text-embedding-3-small` (1536 dims)
- **Chunking**: 1000 chars with 200 char overlap
- **Vector Store**: Qdrant with cosine similarity
- **Search**: Pure vector similarity (no reranking)

---

## 🔴 Critical Issues (Historical - Most Resolved)

### 1. **Character-Based Chunking (Not Token-Based)**

**Problem**: Current chunking uses 1000 characters, but LLMs work with tokens.

- 1000 chars ≈ 250-300 tokens (varies by language)
- Wastes context window when chunks are too small
- Risks truncation when chunks are too large

**Impact**: Suboptimal retrieval and context utilization

---

### 2. **No Reranking** ✅ RESOLVED

**Problem**: Vector search alone has limitations:

- Embeddings capture semantic meaning but miss exact keywords
- Top-k results may not be contextually relevant to specific query
- No consideration of query-document interaction

**Impact**: Lower precision - relevant docs may be ranked poorly

**✅ Resolution**: Cross-encoder reranking implemented in `backend/app/services/rag/reranker.py`
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` model
- Re-scores top candidates for improved relevance
- Configurable via `RAGService` initialization

---

### 3. **Single Search Strategy (Semantic Only)** ✅ RESOLVED

**Problem**: Pure vector search misses exact matches

- Query: "What is the GPT-4 context window?" → May miss exact numbers
- Technical terms, product names, IDs need exact matching
- Vector embeddings can conflate similar concepts

**Impact**: Poor recall on factual/exact-match queries

**✅ Resolution**: Hybrid search implemented with:
- PostgreSQL Full-Text Search (BM25) in `backend/app/services/rag/postgres_bm25.py`
- Reciprocal Rank Fusion in `backend/app/services/rag/hybrid_search.py`
- Configurable semantic/keyword weights via settings

---

### 4. **Fixed Chunk Size**

**Problem**: All content chunked uniformly

- Code: Should chunk by functions/classes (larger)
- Lists/tables: Should keep intact
- Conversations: Should preserve Q&A pairs
- Long paragraphs: Can be split mid-thought

**Impact**: Broken semantic units, loss of context

---

### 5. **No Query Processing**

**Problem**: User queries used as-is

- Conversational queries: "What about the pricing?" (lacks context)
- Multi-part questions: "How does X work and what are limitations?"
- Typos/informal language: "hw 2 setup auth"

**Impact**: Poor query-document matching

---

## 🟡 Medium Priority Issues

### 6. **No Contextual Compression**

**Problem**: Retrieved chunks may contain irrelevant information

- A 1000-char chunk may only have 200 chars relevant to query
- Wastes LLM context window
- Dilutes signal with noise

**Example**:

```
Query: "What is the return policy?"
Retrieved: [Full terms of service with embedded return policy paragraph]
```

---

### 7. **No Deduplication** ✅ RESOLVED

**Problem**: Similar chunks returned multiple times

- Overlapping chunks (by design) can lead to redundancy
- Similar content across documents
- Wastes context and confuses LLM

**✅ Resolution**: Deduplication implemented in `backend/app/services/rag/deduplication.py`
- MMR (Maximal Marginal Relevance) for diversity
- Token-based deduplication for exact/near duplicates
- Configurable thresholds and strategies

---

### 8. **Simple Multi-Pool Merging**

**Problem**: Results from different pools just sorted by score

- Scores from different embeddings not comparable
- No consideration of pool relevance to query
- No reciprocal rank fusion

---

### 9. **No Metadata Enrichment**

**Problem**: Lost context during chunking

- Parent document title not in chunk metadata
- No section headers preserved
- No surrounding context references

---

### 10. **Character-Based Splitting Logic**

**Problem**: Hierarchical separators are crude

- Doesn't understand document structure
- Treats all paragraphs equally
- No semantic boundary detection

---

## 🟢 Nice-to-Have Improvements

### 11. No Query Routing

### 12. No Parent Document Retrieval

### 13. No Temporal/Recency Boosting

### 14. No User Feedback Loop

### 15. No A/B Testing Framework

---

## 📋 Recommended Improvements (Prioritized)

> **Implementation Status**: Phase 1 (Hybrid Search + Reranking) has been completed. See [Current Status](#current-status) above.

## **Phase 1: Quick Wins (1-2 days)** ✅ COMPLETED

### ✅ 1.1 Token-Based Chunking

**Change from character-based to token-based chunking**

```python
# Current
chunk_size: int = 1000  # characters

# Proposed
chunk_size_tokens: int = 512  # tokens (GPT-4 compatible)
```

**Implementation**:

- Use `tiktoken` library for accurate token counting
- Target: 512 tokens per chunk (safe for most models)
- Adjust overlap to 50-100 tokens

**Benefits**:

- ✅ Predictable context window usage
- ✅ Better cross-model compatibility
- ✅ More efficient retrieval

**Effort**: 4 hours (add tiktoken, update SmartTextSplitter)

---

### ✅ 1.2 Add Hybrid Search (Semantic + Keyword) ✅ IMPLEMENTED

**Combine vector similarity with BM25 keyword search**

```python
# Proposed Architecture
query → [Semantic Search (vector)] + [Keyword Search (BM25)] → Fusion → Rerank
```

**Implementation Options**:

**Option A: Qdrant Sparse Vectors (Recommended)**

- Qdrant 1.7+ supports sparse vectors natively
- Store both dense (semantic) and sparse (BM25) vectors
- Hybrid search in single query

**Option B: Dual Storage**

- Keep Qdrant for semantic search
- Add Elasticsearch/Meilisearch for keyword search
- Merge results with Reciprocal Rank Fusion

**Benefits**:

- ✅ Captures both semantic meaning and exact terms
- ✅ Better for technical queries (IDs, versions, specific terms)
- ✅ Handles typos with semantic, exact matches with keyword

**Effort**:

- Option A: 8 hours (Qdrant upgrade + sparse vector generation)
- Option B: 16 hours (add new service + integration)

---

### ✅ 1.3 Basic Reranking ✅ IMPLEMENTED

**Add cross-encoder reranking for top results**

```python
# Proposed Pipeline
Vector Search (top 20) → Reranker → Top 5 final results
```

**Implementation**:

- Use `cross-encoder/ms-marco-MiniLM-L-6-v2` (40MB, fast)
- Or Cohere Rerank API (better quality, cost $1/1k queries)
- Rerank top 20 candidates → return top 5

**Benefits**:

- ✅ Significantly improves precision (10-30% typical improvement)
- ✅ Handles query-document interaction
- ✅ Filters out false positives from embedding search

**Effort**: 6 hours (add reranker, update search pipeline)

---

### ✅ 1.4 Query Expansion

**Expand queries with synonyms and context**

```python
# Example
Original: "How to setup auth?"
Expanded: "How to setup authentication? How to configure authorization? JWT setup guide"
```

**Implementation**:

- **Simple**: Use LLM to generate 2-3 query variations
- **Advanced**: Use query history to build expansion dictionary

```python
async def expand_query(query: str) -> List[str]:
    """Generate query variations using LLM."""
    prompt = f"""Generate 2 alternative phrasings of this query:
    Query: {query}

    Return only the alternative queries, one per line."""

    variations = await llm.generate(prompt)
    return [query] + variations.split('\n')
```

**Benefits**:

- ✅ Better recall (finds more relevant docs)
- ✅ Handles terminology variations
- ✅ Compensates for poor user query phrasing

**Effort**: 4 hours (LLM integration for query expansion)

---

## **Phase 2: Substantial Improvements (3-5 days)**

### ⭐ 2.1 Semantic Chunking

**Replace fixed-size chunks with meaning-based boundaries**

**Approaches**:

**A. LLM-Based Semantic Chunking**

- Use small model to identify topic boundaries
- Chunk on topic shifts, not arbitrary size
- Better for narrative documents

**B. Embedding Similarity Chunking**

- Split on sentences
- Merge consecutive sentences with high similarity
- Split when similarity drops (topic shift)

**C. Structure-Aware Chunking**

- Code: Split by functions/classes
- Markdown: Split by headers (H2/H3)
- JSON/XML: Split by logical structure
- Tables: Keep intact

**Recommendation**: Combine B + C

```python
if is_code(document):
    chunks = chunk_by_ast(document)  # Use AST parser
elif is_markdown(document):
    chunks = chunk_by_headers(document)
else:
    chunks = semantic_similarity_chunking(document)
```

**Benefits**:

- ✅ Preserves semantic coherence
- ✅ No mid-thought splits
- ✅ Better for structured documents

**Effort**: 12 hours (implement + test multiple strategies)

---

### ⭐ 2.2 Contextual Compression

**Extract only relevant portions from retrieved chunks**

```python
# Current
chunk = "... [1000 chars of content, 200 relevant to query] ..."

# With Compression
compressed = "... [200 chars directly relevant to query] ..."
```

**Implementation**:

- Use LLM to extract relevant sentences from each chunk
- Or use extractive summarization model (faster)

```python
async def compress_context(query: str, chunks: List[str]) -> List[str]:
    """Extract only relevant portions of chunks."""
    compressed = []
    for chunk in chunks:
        prompt = f"""Extract ONLY the sentences from this text that are relevant to the query.
        Query: {query}
        Text: {chunk}

        Return only the relevant sentences, maintaining original wording."""

        relevant = await llm.generate(prompt, max_tokens=500)
        if relevant.strip():
            compressed.append(relevant)
    return compressed
```

**Benefits**:

- ✅ 2-3x more results fit in context window
- ✅ Reduces noise for LLM
- ✅ Improves answer quality

**Effort**: 8 hours (LLM integration + prompt tuning)

---

### ⭐ 2.3 Hypothetical Document Embeddings (HyDE)

**Generate hypothetical answer, embed it, search**

```python
# Instead of embedding the query directly
query = "How does JWT auth work?"

# Generate hypothetical answer
hypothetical_answer = await llm.generate(
    f"Write a detailed answer to: {query}"
)

# Embed and search with the answer (not the question)
embedding = await embed(hypothetical_answer)
results = await search(embedding)
```

**Rationale**:

- Questions and answers have different embedding spaces
- Query: "How does X work?" vs Answer: "X works by doing Y and Z"
- Searching with answer embeddings matches better to document embeddings

**Benefits**:

- ✅ 20-40% improvement in retrieval for complex queries
- ✅ Especially good for "how-to" questions
- ✅ No changes to indexing needed

**Effort**: 6 hours (LLM integration + evaluation)

---

### ⭐ 2.4 Parent Document Retrieval

**Retrieve small chunks, return larger context**

```python
# Index Strategy
Document → Chunk into small pieces (256 tokens)
         → Also store parent chunks (1024 tokens)

# Retrieval Strategy
Search small chunks (high precision)
→ Return parent chunks (high context)
```

**Implementation**:

```python
# During indexing
small_chunks = split_text(doc, chunk_size=256)
parent_chunks = split_text(doc, chunk_size=1024)

for small, parent in zip(small_chunks, parent_chunks):
    metadata = {
        "parent_id": parent.id,
        "parent_content": parent.content,
    }
    store(small, metadata)

# During retrieval
results = search(query)  # Returns small chunks
parent_contents = [r.metadata["parent_content"] for r in results]
return parent_contents  # Return parents instead
```

**Benefits**:

- ✅ Best of both worlds: precision + context
- ✅ Solves "chunk too small" problem
- ✅ Maintains coherence

**Effort**: 10 hours (modify indexing + retrieval pipeline)

---

## **Phase 3: Advanced (1-2 weeks)**

### 🚀 3.1 Query Routing

Route different query types to different strategies

```python
if is_factual_query(query):
    # Use exact matching + reranking
    results = hybrid_search(query, boost_keywords=True)
elif is_conversational(query):
    # Expand query with conversation history
    query = add_conversation_context(query, history)
    results = semantic_search(query)
elif is_comparison(query):
    # Retrieve diverse results
    results = mmr_search(query, diversity=0.7)
```

---

### 🚀 3.2 Reciprocal Rank Fusion (RRF)

Better multi-query and multi-pool merging

```python
# For query expansion or multi-pool search
queries = expand_query(original_query)
all_results = []

for query in queries:
    results = search(query)
    all_results.append(results)

# Fuse with RRF instead of simple score sorting
final_results = reciprocal_rank_fusion(all_results)
```

---

### 🚀 3.3 Metadata-Aware Retrieval

Boost results based on metadata

```python
# Boost factors
- Recency: Recent docs scored higher
- Source credibility: Official docs > user comments
- Document type: Code examples vs explanations
- User feedback: Upvoted chunks boosted
```

---

### 🚀 3.4 Adaptive Retrieval

Dynamically adjust retrieval based on query confidence

```python
initial_results = search(query, top_k=5)

if max_score < CONFIDENCE_THRESHOLD:
    # Low confidence, retrieve more + expand query
    expanded = expand_query(query)
    results = search_multi(expanded, top_k=10)
    results = rerank(results)
    return top_5(results)
else:
    return initial_results
```

---

## 🎯 Recommended Implementation Order

### **Week 1: Foundation**

1. ✅ Token-based chunking (4h)
2. ✅ Basic reranking with cross-encoder (6h)
3. ✅ Query expansion (4h)
4. ✅ Hybrid search setup (8h)

**Expected improvement**: 30-50% better precision/recall

---

### **Week 2: Advanced Retrieval**

5. ⭐ Semantic chunking (12h)
6. ⭐ Contextual compression (8h)
7. ⭐ HyDE implementation (6h)

**Expected improvement**: Another 20-30% improvement + better context usage

---

### **Week 3: Polish**

8. ⭐ Parent document retrieval (10h)
9. 🚀 Query routing (8h)
10. 🚀 RRF for multi-pool (4h)

**Expected improvement**: More consistent, handles edge cases

---

## 📊 Expected Impact Summary

| Improvement | Precision Gain | Recall Gain | Effort | Priority |
|-------------|---------------|-------------|--------|----------|
| Token-based chunking | +5% | +5% | 4h | ⭐⭐⭐ |
| Hybrid search | +15% | +25% | 8h | ⭐⭐⭐ |
| Reranking | +20% | +5% | 6h | ⭐⭐⭐ |
| Query expansion | +10% | +20% | 4h | ⭐⭐ |
| Semantic chunking | +10% | +10% | 12h | ⭐⭐ |
| Contextual compression | +15% | 0% | 8h | ⭐⭐ |
| HyDE | +25% | +15% | 6h | ⭐⭐ |
| Parent retrieval | +10% | +10% | 10h | ⭐ |

**Total Expected Improvement**: 50-80% better overall performance

---

## 🛠 Implementation Notes

### Dependencies to Add

```bash
# Token counting
pip install tiktoken

# Reranking
pip install sentence-transformers
# Or use Cohere API

# BM25 (for hybrid search option B)
pip install rank-bm25

# AST parsing (for code chunking)
pip install tree-sitter tree-sitter-languages
```

### Configuration Updates

```python
# config.py additions
class Settings(BaseSettings):
    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # Retrieval
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Search
    initial_retrieval_k: int = 20
    final_results_k: int = 5
    enable_query_expansion: bool = True
```

---

## 🧪 Evaluation Strategy

### Create Test Dataset

```python
# test_queries.json
[
    {
        "query": "How do I reset my password?",
        "relevant_docs": ["doc_123", "doc_456"],
        "type": "factual"
    },
    {
        "query": "What are the differences between plans?",
        "relevant_docs": ["doc_789"],
        "type": "comparison"
    }
]
```

### Metrics to Track

- **Precision@5**: Relevant docs in top 5
- **Recall@5**: % of relevant docs retrieved
- **MRR**: Mean reciprocal rank
- **NDCG**: Normalized discounted cumulative gain
- **Latency**: P50, P95, P99

### A/B Testing

```python
# Compare old vs new retrieval
results_old = old_rag_pipeline(query)
results_new = new_rag_pipeline(query)

log_comparison(query, results_old, results_new, user_feedback)
```

---

## 💡 Quick Experiment: Measure Current Performance

Before implementing improvements, establish baseline:

```python
# Create test set
test_queries = [
    ("What is the pricing?", ["pricing_doc_id"]),
    ("How does authentication work?", ["auth_doc_id"]),
    # ... 20-30 queries
]

# Measure current performance
for query, relevant_docs in test_queries:
    results = rag_service.search(query, limit=5)
    result_ids = [r.document_id for r in results]

    precision = len(set(result_ids) & set(relevant_docs)) / len(result_ids)
    recall = len(set(result_ids) & set(relevant_docs)) / len(relevant_docs)

    print(f"Query: {query}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
```

This baseline helps quantify improvements from each change.

---

## 🤔 Discussion Questions

1. **Which improvements should we prioritize first?**
   - My recommendation: Hybrid search + Reranking (biggest impact for effort)

2. **Budget constraints?**
   - Cohere reranking: $1/1k queries (high quality)
   - vs Local cross-encoder: Free but slower (50-100ms per rerank)

3. **What types of documents/queries are most common?**
   - Technical docs → prioritize code-aware chunking
   - Conversational → prioritize query expansion
   - Mixed → prioritize hybrid search

4. **Acceptable latency?**
   - Current: ~100-200ms (embedding + vector search)
   - With reranking: +50-150ms (local) or +200-300ms (API)
   - With HyDE: +500-1000ms (LLM generation)

5. **How to measure success?**
   - User feedback loop?
   - Automated eval set?
   - A/B testing?

---

**Ready to discuss and prioritize! Which improvements are most interesting to you?**
