# RAG Chunking Improvements Proposal - 2025 Best Practices

**Date**: 2025-10-24
**Current System**: Character-based chunking with hierarchical separators
**Goal**: Implement state-of-the-art chunking strategies based on latest research

---

## Executive Summary

Based on analysis of your current implementation and latest 2024-2025 research, I propose a **phased approach** to modernize your RAG chunking strategy:

**Phase 1 (Week 1)**: Foundation improvements - **30-50% expected improvement**
- Token-based chunking
- Anthropic's Contextual Retrieval
- Basic hybrid search

**Phase 2 (Week 2)**: Advanced techniques - **Additional 20-30% improvement**
- Semantic chunking with embedding similarity
- Late chunking (2025 cutting-edge technique)
- Cross-encoder reranking

**Phase 3 (Week 3+)**: Production optimization
- Agentic chunking for complex documents
- Query decomposition & routing
- Adaptive retrieval strategies

---

## Current Implementation Analysis

### File: `/home/user/rag-system/backend/app/services/rag/text_splitter.py`

**Current Approach**: Character-based hierarchical splitting

```python
# Current Configuration
chunk_size: int = 1000          # CHARACTERS (not tokens!)
chunk_overlap: int = 200         # CHARACTERS
```

**Separators Hierarchy**:
```python
["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
```

### Strengths
✅ Clean protocol-based architecture
✅ Respects sentence boundaries
✅ Multiple document loaders (PDF, DOCX, web, text)
✅ Hierarchical separator logic
✅ Evaluation framework exists

### Critical Weaknesses
❌ **Character-based** (1000 chars ≈ 250-300 tokens - highly variable)
❌ **No contextual enrichment** (chunks lack document context)
❌ **Fixed chunk sizes** (doesn't adapt to content type)
❌ **No semantic awareness** (splits on syntax, not meaning)
❌ **Incomplete markdown handling** (code/tables extraction not merged back)
❌ **No token-level precision** (causes context window waste/overflow)

---

## Latest Research Findings (2024-2025)

### 1. Anthropic's Contextual Retrieval (Sept 2024) 🔥

**Source**: https://www.anthropic.com/news/contextual-retrieval

**The Problem**: Traditional RAG chunks lack context
```
Chunk: "Revenue increased 15% to $30M in Q3"
Missing: Which company? Which year? What product line?
```

**The Solution**: Prepend chunk-specific context using Claude

```python
# Before: Raw chunk
"Revenue increased 15% to $30M in Q3"

# After: Contextualized chunk
"This chunk is from Acme Corp's 2024 Q3 earnings report.
The Software Division reported revenue increased 15% to $30M in Q3"
```

**Performance**:
- 35% reduction in retrieval failures (Contextual Embeddings alone)
- 49% reduction (Contextual Embeddings + Contextual BM25)
- **67% reduction with reranking** 🎯

**Cost**: ~$1.02 per million document tokens (one-time, using prompt caching)

**Implementation Strategy**:
```python
async def add_contextual_prefix(document: str, chunk: str) -> str:
    """Generate chunk-specific context using Claude."""
    prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the
chunk. Answer only with the succinct context and nothing else.
"""

    context = await claude.generate(prompt, max_tokens=100, cache_document=True)
    return f"{context}\n\n{chunk}"
```

**Key Insight**: Use prompt caching! Load document once, generate context for all chunks efficiently.

---

### 2. Late Chunking (2025) 🚀

**Source**: arXiv:2409.04701 "Late Chunking: Contextual Chunk Embeddings"

**Revolutionary Approach**: Process entire document, then chunk embeddings

```
Traditional Chunking:
Text → Split into chunks → Embed each chunk separately
Problem: Each chunk embedded without full document context

Late Chunking:
Text → Tokenize entire document → Apply transformer to ALL tokens →
Mean pool token embeddings into chunks → Chunk embeddings with full context
```

**How It Works**:
1. Tokenize entire document (e.g., 10,000 tokens)
2. Pass through embedding model transformer (gets full document attention)
3. Get contextualized token embeddings for all 10,000 tokens
4. Split token embedding sequence into logical chunks
5. Mean pool each chunk's token embeddings into final chunk embedding

**Benefits**:
- Each chunk embedding contains information from entire document
- Better captures cross-chunk relationships
- Especially powerful for technical documents with forward/backward references

**Performance**:
- Significant improvements on code documentation
- Better handling of pronouns, references, acronyms

**Implementation Complexity**: Medium-High (requires custom embedding pipeline)

**Recommendation**: Phase 2 implementation after foundational improvements

---

### 3. Semantic Chunking (Industry Consensus)

**Sources**: Multiple 2024-2025 papers, LangChain/LlamaIndex implementations

**Consensus**: Semantic chunking outperforms fixed-size in most scenarios

**Three Approaches**:

#### A. Embedding Similarity Chunking
```python
# Split on sentences first
sentences = split_into_sentences(text)

# Embed each sentence
embeddings = [embed(s) for s in sentences]

# Merge consecutive sentences while similarity is high
chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    similarity = cosine_similarity(embeddings[i-1], embeddings[i])

    if similarity > THRESHOLD:  # e.g., 0.75
        current_chunk.append(sentences[i])
    else:
        # Topic shift detected
        chunks.append(' '.join(current_chunk))
        current_chunk = [sentences[i]]
```

**Pros**: Content-aware, preserves semantic coherence
**Cons**: Expensive (must embed every sentence), variable chunk sizes

#### B. Structure-Aware Chunking
```python
if is_code(document):
    chunks = chunk_by_ast(document)  # Functions/classes
elif is_markdown(document):
    chunks = chunk_by_headers(document)  # H2/H3 sections
elif is_conversation(document):
    chunks = chunk_by_qa_pairs(document)
else:
    chunks = semantic_embedding_chunking(document)
```

**Pros**: Respects document structure, fast
**Cons**: Requires format detection, multiple implementations

#### C. Max-Min Semantic Chunking (2025)
**Source**: Springer Journal, Jan 2025

Novel algorithm using semantic similarity + Max-Min optimization to find optimal chunk boundaries

**Recommendation**: Implement Structure-Aware (B) first, then add Embedding Similarity (A) for unstructured text

---

### 4. Agentic Chunking (IBM + LangChain 2024)

**Source**: IBM Think Tutorial - Agentic Chunking with watsonx.ai

**Concept**: Use an LLM agent to decide how to chunk documents

```python
async def agentic_chunk(document: str) -> List[str]:
    """LLM decides optimal chunking strategy for this document."""

    # Agent analyzes document
    analysis = await llm.generate(f"""
    Analyze this document and determine the optimal chunking strategy:

    Document preview:
    {document[:1000]}...

    Provide:
    1. Document type (code, prose, technical doc, conversation, etc.)
    2. Optimal chunk boundaries (headers, functions, paragraphs, etc.)
    3. Recommended chunk size
    4. Any special handling needed
    """)

    # Agent proposes chunks
    proposed_chunks = await llm.generate(f"""
    Based on your analysis, split this document into optimal chunks:
    {document}

    Return chunks as JSON array.
    """)

    return json.loads(proposed_chunks)
```

**Benefits**:
- Adapts to document structure automatically
- Handles edge cases intelligently
- Can preserve complex structures (nested lists, code blocks, etc.)

**Drawbacks**:
- Expensive (LLM call per document)
- Slower (500-2000ms per document)
- Variable quality

**Recommendation**: Use for complex/important documents, not bulk ingestion

---

### 5. Token-Based Chunking (Standard Practice 2024+)

**Sources**: LangChain, LlamaIndex documentation

**Critical**: Always use tokenizer matching your embedding model

```python
import tiktoken

# OpenAI embeddings use cl100k_base tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 64):
    """Chunk text by token count, not characters."""
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_tokens

    return chunks
```

**Recommended Token Sizes** (2024 best practices):
- **Small chunks**: 256-512 tokens (better precision, more chunks)
- **Medium chunks**: 512-1024 tokens (balanced, most common)
- **Large chunks**: 1024-2048 tokens (more context, fewer chunks)

**Overlap**: 10-20% of chunk size (50-100 tokens for 512-token chunks)

**Why This Matters**:
- 1000 characters ≈ 250-300 tokens (highly variable by language)
- English: ~4 chars/token
- Code: ~2.5 chars/token
- Chinese: ~1.5 chars/token
- Token-based gives predictable context usage

---

## Proposed Implementation Plan

---

## 🎯 Phase 1: Foundation (Week 1) - Target: 30-50% Improvement

### 1.1 Token-Based Chunking ⭐⭐⭐
**Priority**: CRITICAL
**Effort**: 4-6 hours
**Files**: `text_splitter.py`, `config.py`

**Implementation**:

```python
# config.py - Update settings
class Settings(BaseSettings):
    # Replace character-based with token-based
    chunk_size_tokens: int = 512  # Target chunk size in tokens
    chunk_overlap_tokens: int = 64  # 12.5% overlap

    # Keep character-based as fallback for non-tokenizable content
    chunk_size_chars: int = 2000  # ~500 tokens
    chunk_overlap_chars: int = 250

    # Tokenizer configuration
    tokenizer_encoding: str = "cl100k_base"  # For OpenAI embeddings
```

```python
# text_splitter.py - Add TokenAwareTextSplitter
import tiktoken
from typing import List, Optional, Dict, Any

class TokenAwareTextSplitter(SmartTextSplitter):
    """
    Token-based text splitter using tiktoken for accurate token counting.

    Uses same tokenizer as embedding model to ensure chunks fit in context windows.
    """

    def __init__(
        self,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        encoding_name: str = "cl100k_base",  # OpenAI default
        separators: List[str] | None = None,
    ):
        """Initialize token-aware splitter."""
        self.chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Convert token limits to approximate character limits for base class
        chars_per_token = 4  # Average for English
        chunk_size_chars = self.chunk_size_tokens * chars_per_token
        chunk_overlap_chars = self.chunk_overlap_tokens * chars_per_token

        super().__init__(
            chunk_size=chunk_size_chars,
            chunk_overlap=chunk_overlap_chars,
            separators=separators,
        )

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split text by token count with hierarchical separators."""
        if not text.strip():
            return []

        metadata = metadata or {}

        # Use hierarchical splitting with token-aware size checks
        chunks_text = self._split_with_token_awareness(text, self.separators)

        # Create DocumentChunk objects with token counts
        chunks: List[DocumentChunk] = []
        for i, chunk_text in enumerate(chunks_text):
            token_count = len(self.tokenizer.encode(chunk_text))

            chunk_metadata = metadata.copy()
            chunk_metadata["token_count"] = token_count

            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
            )
            chunks.append(chunk)

        return chunks

    def _split_with_token_awareness(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Split text using hierarchical separators with token counting."""
        tokens = self.tokenizer.encode(text)

        # If text fits in one chunk, return it
        if len(tokens) <= self.chunk_size_tokens:
            return [text] if text.strip() else []

        # Try each separator
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by tokens directly
                return self._split_by_tokens(text)

            splits = text.split(separator)
            splits = [s for s in splits if s.strip()]

            if len(splits) <= 1:
                continue

            # Merge splits respecting token limits
            chunks = []
            current_chunk = []
            current_tokens = 0

            for split in splits:
                split_tokens = len(self.tokenizer.encode(split + separator))

                # If split is too large, recursively split it
                if split_tokens > self.chunk_size_tokens:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0

                    sub_chunks = self._split_with_token_awareness(
                        split,
                        separators[i + 1:],
                    )
                    chunks.extend(sub_chunks)
                    continue

                # Check if adding split exceeds limit
                if current_tokens + split_tokens > self.chunk_size_tokens and current_chunk:
                    chunks.append(separator.join(current_chunk))

                    # Create overlap
                    overlap_chunks = []
                    overlap_tokens = 0
                    for prev_split in reversed(current_chunk):
                        split_tok = len(self.tokenizer.encode(prev_split + separator))
                        if overlap_tokens + split_tok > self.chunk_overlap_tokens:
                            break
                        overlap_chunks.insert(0, prev_split)
                        overlap_tokens += split_tok

                    current_chunk = overlap_chunks
                    current_tokens = overlap_tokens

                current_chunk.append(split)
                current_tokens += split_tokens

            if current_chunk:
                chunks.append(separator.join(current_chunk))

            return chunks

        return self._split_by_tokens(text)

    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by token count directly (last resort)."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

            start = end - self.chunk_overlap_tokens if end < len(tokens) else end

        return chunks

    @property
    def chunk_size(self) -> int:
        """Get chunk size in tokens."""
        return self.chunk_size_tokens

    @property
    def chunk_overlap(self) -> int:
        """Get overlap size in tokens."""
        return self.chunk_overlap_tokens
```

**Dependencies**:
```bash
pip install tiktoken
```

**Expected Impact**:
- ✅ Predictable context window usage
- ✅ 5-10% improvement in retrieval quality
- ✅ Foundation for all other improvements

---

### 1.2 Anthropic's Contextual Retrieval ⭐⭐⭐
**Priority**: HIGH
**Effort**: 8-10 hours
**Files**: New `contextual_chunking.py`, update `rag_service.py`

**Implementation**:

```python
# app/services/rag/contextual_chunking.py
"""
Contextual Retrieval implementation based on Anthropic's 2024 research.

Adds document-specific context to each chunk before embedding to improve
retrieval accuracy by 35-67%.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from typing import List, Dict, Any
import anthropic
from app.config import settings
from app.services.rag.protocols import DocumentChunk

class ContextualChunker:
    """
    Adds contextual prefixes to chunks using Claude.

    Performance improvements:
    - 35% fewer retrieval failures (with embeddings alone)
    - 49% fewer failures (with embeddings + BM25)
    - 67% fewer failures (with embeddings + BM25 + reranking)
    """

    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        self.client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key
        )

    async def add_context_to_chunks(
        self,
        document: str,
        chunks: List[DocumentChunk],
        batch_size: int = 10,
    ) -> List[DocumentChunk]:
        """
        Add contextual prefixes to chunks using Claude.

        Uses prompt caching to reduce cost:
        - First call: Caches document (~$0.30/1M tokens)
        - Subsequent calls: Use cached document (~$0.03/1M tokens)

        Cost: ~$1.02/1M document tokens for initial processing

        Args:
            document: Full source document text
            chunks: List of chunks to contextualize
            batch_size: Number of chunks to process in parallel

        Returns:
            Chunks with contextual prefixes added
        """
        contextualized_chunks = []

        # Process chunks in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Process batch in parallel
            tasks = [
                self._add_context_to_chunk(document, chunk)
                for chunk in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            contextualized_chunks.extend(batch_results)

        return contextualized_chunks

    async def _add_context_to_chunk(
        self,
        document: str,
        chunk: DocumentChunk,
    ) -> DocumentChunk:
        """Add context to a single chunk."""

        # Generate contextual prefix using Claude
        prompt = self._build_context_prompt(document, chunk.content)

        try:
            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,  # Context should be concise
                temperature=0.0,  # Deterministic
                system=[
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that provides concise context for text chunks.",
                    },
                    {
                        "type": "text",
                        "text": document,
                        "cache_control": {"type": "ephemeral"}  # Cache the document!
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            context = response.content[0].text.strip()

            # Prepend context to chunk
            contextualized_content = f"{context}\n\n{chunk.content}"

            # Create new chunk with context
            return DocumentChunk(
                content=contextualized_content,
                metadata={
                    **chunk.metadata,
                    "has_context": True,
                    "context": context,
                },
                chunk_index=chunk.chunk_index,
            )

        except Exception as e:
            # Fallback: return original chunk if context generation fails
            print(f"Failed to generate context for chunk {chunk.chunk_index}: {e}")
            return chunk

    def _build_context_prompt(self, document: str, chunk: str) -> str:
        """Build prompt for context generation."""
        return f"""Here is the chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please give a short succinct context (2-3 sentences max) to situate this chunk
within the overall document for the purposes of improving search retrieval of
the chunk.

Include:
- What document/section this is from
- Key entities or topics being discussed
- Relevant temporal or categorical context

Answer only with the succinct context and nothing else."""


# Update rag_service.py to use contextual chunking
class RAGService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        text_splitter: TextSplitter,
        use_contextual_chunking: bool = True,  # New parameter
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.text_splitter = text_splitter
        self.use_contextual_chunking = use_contextual_chunking

        if use_contextual_chunking:
            self.contextual_chunker = ContextualChunker()

    async def ingest_document(
        self,
        document: Document,
        collection_name: str,
    ) -> str:
        """Ingest document with optional contextual chunking."""

        # 1. Split document into chunks
        chunks = self.text_splitter.split_text(
            document.content,
            metadata=document.metadata,
        )

        # 2. Add contextual prefixes (if enabled)
        if self.use_contextual_chunking:
            chunks = await self.contextual_chunker.add_context_to_chunks(
                document=document.content,
                chunks=chunks,
            )

        # 3. Embed chunks (rest of implementation stays the same)
        # ... existing code ...
```

**Configuration**:
```python
# config.py additions
class Settings(BaseSettings):
    # Contextual Retrieval
    enable_contextual_chunking: bool = True
    contextual_chunking_batch_size: int = 10
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
```

**Expected Impact**:
- ✅ 35-67% reduction in retrieval failures
- ✅ Better handling of ambiguous queries
- ✅ Improved cross-chunk reference resolution
- 💰 One-time cost: ~$1/million doc tokens

---

### 1.3 Basic Hybrid Search (Semantic + BM25) ⭐⭐⭐
**Priority**: HIGH
**Effort**: 6-8 hours
**Files**: New `hybrid_search.py`, update `vector_store.py`

**Implementation**:

```python
# app/services/rag/hybrid_search.py
"""
Hybrid search combining semantic (vector) and keyword (BM25) search.

Research shows this improves recall by 15-25% vs semantic search alone.
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearchService:
    """
    Combines semantic and keyword search with score fusion.

    Strategies:
    1. Semantic search (vector similarity) - captures meaning
    2. BM25 keyword search - captures exact terms
    3. Reciprocal Rank Fusion (RRF) - combines results
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

        # BM25 indices: collection_name -> BM25Okapi instance
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        # Document content: collection_name -> List[document_content]
        self.doc_contents: Dict[str, List[Dict[str, Any]]] = {}

    async def index_for_hybrid_search(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Build BM25 index for a collection.

        Args:
            collection_name: Name of collection
            documents: List of dicts with 'content' and 'id' keys
        """
        # Tokenize documents for BM25
        tokenized_docs = [
            doc["content"].lower().split()
            for doc in documents
        ]

        # Build BM25 index
        self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
        self.doc_contents[collection_name] = documents

    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        initial_k: int = 20,  # Retrieve more, then rerank
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            collection_name: Collection to search
            limit: Number of final results
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for BM25 search (0-1)
            initial_k: Number of candidates from each method

        Returns:
            Top results ranked by combined score
        """
        # 1. Semantic search
        semantic_results = await self._semantic_search(
            query=query,
            collection_name=collection_name,
            limit=initial_k,
        )

        # 2. BM25 keyword search
        keyword_results = await self._bm25_search(
            query=query,
            collection_name=collection_name,
            limit=initial_k,
        )

        # 3. Combine using Reciprocal Rank Fusion (RRF)
        combined_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

        return combined_results[:limit]

    async def _semantic_search(
        self,
        query: str,
        collection_name: str,
        limit: int,
    ) -> List[tuple[str, float]]:
        """Perform semantic vector search."""
        # Embed query
        query_vector = await self.embedding_provider.embed_text(query)

        # Search vector store
        results = await self.vector_store.search(
            collection=collection_name,
            query_vector=query_vector,
            limit=limit,
        )

        return [(r.document_id, r.score) for r in results]

    async def _bm25_search(
        self,
        query: str,
        collection_name: str,
        limit: int,
    ) -> List[tuple[str, float]]:
        """Perform BM25 keyword search."""
        if collection_name not in self.bm25_indices:
            return []

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        bm25 = self.bm25_indices[collection_name]
        scores = bm25.get_scores(query_tokens)

        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:limit]

        documents = self.doc_contents[collection_name]
        results = [
            (documents[i]["id"], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[tuple[str, float]],
        keyword_results: List[tuple[str, float]],
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        k: int = 60,  # RRF constant
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ w_i / (k + rank_i(d))

        More robust than score normalization since scores from different
        systems aren't directly comparable.
        """
        # Build rank maps
        semantic_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(semantic_results)}
        keyword_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(keyword_results)}

        # Get all unique document IDs
        all_doc_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0

            if doc_id in semantic_ranks:
                score += semantic_weight / (k + semantic_ranks[doc_id])

            if doc_id in keyword_ranks:
                score += keyword_weight / (k + keyword_ranks[doc_id])

            rrf_scores[doc_id] = score

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Convert to SearchResult objects
        # (You'll need to fetch full documents from vector store)
        return [
            SearchResult(document_id=doc_id, score=score, content="...", metadata={})
            for doc_id, score in sorted_docs
        ]
```

**Dependencies**:
```bash
pip install rank-bm25
```

**Expected Impact**:
- ✅ 15-25% improvement in recall
- ✅ Better exact-match queries (IDs, names, numbers)
- ✅ Handles typos better (semantic fallback)

---

### 1.4 Query Expansion ⭐⭐
**Priority**: MEDIUM
**Effort**: 4-6 hours
**Files**: New `query_processing.py`

**Implementation**:

```python
# app/services/rag/query_processing.py
"""Query processing utilities for improved retrieval."""

from typing import List
from openai import AsyncOpenAI

class QueryProcessor:
    """Processes and expands queries for better retrieval."""

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def expand_query(
        self,
        query: str,
        num_variations: int = 2,
    ) -> List[str]:
        """
        Generate query variations using LLM.

        Example:
          Input: "How to setup auth?"
          Output: [
              "How to setup auth?",
              "How to configure authentication?",
              "JWT setup guide"
          ]
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap
            messages=[
                {
                    "role": "system",
                    "content": "Generate alternative phrasings of user queries.",
                },
                {
                    "role": "user",
                    "content": f"""Generate {num_variations} alternative phrasings of this query.

Query: {query}

Return only the alternative queries, one per line, without numbering."""
                }
            ],
            temperature=0.7,
            max_tokens=150,
        )

        variations = response.choices[0].message.content.strip().split('\n')
        variations = [v.strip() for v in variations if v.strip()]

        return [query] + variations[:num_variations]
```

**Expected Impact**:
- ✅ 10-20% improvement in recall
- ✅ Handles terminology variations
- ✅ Better conversational query handling

---

## 🚀 Phase 2: Advanced Techniques (Week 2) - Target: +20-30% Improvement

### 2.1 Semantic Chunking with Structure Awareness ⭐⭐⭐

**Implementation**:

```python
# app/services/rag/semantic_splitter.py
"""Semantic chunking using embedding similarity and structure awareness."""

from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticTextSplitter:
    """
    Chunks text based on semantic similarity between sentences.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Merge consecutive sentences while similarity > threshold
    4. Split when similarity drops (topic boundary detected)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.75,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 800,
    ):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split text using semantic similarity."""

        # 1. Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [DocumentChunk(content=text, metadata=metadata or {}, chunk_index=0)]

        # 2. Embed all sentences
        sentence_embeddings = await self.embedding_provider.embed_batch(sentences)

        # 3. Find chunk boundaries based on similarity drops
        boundaries = self._find_semantic_boundaries(
            sentences=sentences,
            embeddings=sentence_embeddings,
        )

        # 4. Create chunks from boundaries
        chunks = []
        for i, (start, end) in enumerate(boundaries):
            chunk_text = ' '.join(sentences[start:end])

            chunk_metadata = (metadata or {}).copy()
            chunk_metadata["token_count"] = len(self.tokenizer.encode(chunk_text))
            chunk_metadata["sentence_count"] = end - start

            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
            ))

        return chunks

    def _find_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[tuple[int, int]]:
        """Find chunk boundaries based on semantic similarity."""
        boundaries = []
        current_start = 0
        current_sentences = [sentences[0]]
        current_tokens = len(self.tokenizer.encode(sentences[0]))

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                [embeddings[i-1]],
                [embeddings[i]]
            )[0][0]

            sentence_tokens = len(self.tokenizer.encode(sentences[i]))

            # Check if we should start a new chunk
            should_split = (
                # Semantic boundary detected
                similarity < self.similarity_threshold or
                # Max size reached
                current_tokens + sentence_tokens > self.max_chunk_tokens
            )

            if should_split and current_tokens >= self.min_chunk_tokens:
                # Save current chunk
                boundaries.append((current_start, i))
                current_start = i
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentences[i])
            current_tokens += sentence_tokens

        # Add final chunk
        if current_sentences:
            boundaries.append((current_start, len(sentences)))

        return boundaries
```

---

### 2.2 Structure-Aware Chunking (Code, Markdown, etc.) ⭐⭐⭐

```python
# app/services/rag/structure_aware_splitter.py
"""Content-type-specific chunking strategies."""

import ast
from typing import List, Dict, Any

class StructureAwareSplitter:
    """Routes to appropriate chunking strategy based on content type."""

    async def split_text(
        self,
        text: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split text using content-appropriate strategy."""

        if content_type == "python":
            return self._chunk_python_code(text, metadata)
        elif content_type == "markdown":
            return self._chunk_markdown(text, metadata)
        elif content_type == "json":
            return self._chunk_json(text, metadata)
        else:
            # Fall back to semantic chunking
            return await self.semantic_splitter.split_text(text, metadata)

    def _chunk_python_code(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[DocumentChunk]:
        """Chunk Python code by functions and classes."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to line-based splitting
            return self._chunk_by_lines(code, metadata)

        chunks = []
        chunk_index = 0

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Extract function/class with docstring
                start_line = node.lineno - 1
                end_line = node.end_lineno

                lines = code.split('\n')
                chunk_text = '\n'.join(lines[start_line:end_line])

                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                    "name": node.name,
                    "line_start": start_line + 1,
                    "line_end": end_line,
                })

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        return chunks

    def _chunk_markdown(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[DocumentChunk]:
        """Chunk markdown by headers (H2, H3 sections)."""
        import re

        # Split on H2 headers
        sections = re.split(r'\n## ', text)

        chunks = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Add back the header marker (except for first section)
            if i > 0:
                section = "## " + section

            chunk_metadata = (metadata or {}).copy()

            # Extract header title
            lines = section.split('\n')
            if lines[0].startswith('##'):
                header = lines[0].replace('##', '').strip()
                chunk_metadata["section_header"] = header

            chunks.append(DocumentChunk(
                content=section.strip(),
                metadata=chunk_metadata,
                chunk_index=i,
            ))

        return chunks
```

---

### 2.3 Late Chunking Implementation 🚀

**Note**: Advanced technique, requires custom embedding pipeline

```python
# app/services/rag/late_chunking.py
"""
Late chunking: Tokenize entire document, then chunk the embeddings.

Based on: arXiv:2409.04701 "Late Chunking: Contextual Chunk Embeddings"

This is a 2025 cutting-edge technique that produces superior embeddings
by allowing each chunk to be influenced by the entire document context.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any

class LateChunker:
    """
    Implements late chunking for contextual chunk embeddings.

    Traditional: Text → Chunk → Embed each chunk
    Late Chunking: Text → Tokenize all → Embed all tokens → Chunk embeddings
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens: int = 512,
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.chunk_size_tokens = chunk_size_tokens
        self.device = device

    async def embed_with_late_chunking(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """
        Embed document using late chunking.

        Returns:
            List of dicts with 'chunk_text', 'embedding', 'start_token', 'end_token'
        """
        # 1. Tokenize entire document
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,  # Don't truncate
            padding=False,
        ).to(self.device)

        # 2. Get contextualized token embeddings for ENTIRE document
        with torch.no_grad():
            outputs = self.model(**tokens)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

        # 3. Determine chunk boundaries (by tokens)
        num_tokens = token_embeddings.shape[0]
        chunk_boundaries = []

        start = 0
        while start < num_tokens:
            end = min(start + self.chunk_size_tokens, num_tokens)
            chunk_boundaries.append((start, end))
            start = end

        # 4. Create chunk embeddings by mean pooling token embeddings
        chunks = []
        for i, (start, end) in enumerate(chunk_boundaries):
            # Mean pool token embeddings for this chunk
            chunk_embedding = token_embeddings[start:end].mean(dim=0)

            # Decode tokens back to text
            chunk_token_ids = tokens['input_ids'][0][start:end]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

            chunks.append({
                "chunk_index": i,
                "chunk_text": chunk_text,
                "embedding": chunk_embedding.cpu().numpy().tolist(),
                "start_token": start,
                "end_token": end,
            })

        return chunks
```

**Expected Impact**:
- ✅ 10-20% improvement on documents with cross-references
- ✅ Better handling of pronouns, acronyms
- ✅ Especially good for technical documentation

---

### 2.4 Cross-Encoder Reranking ⭐⭐⭐

```python
# app/services/rag/reranker.py
"""Reranking service using cross-encoder models."""

from typing import List
from sentence_transformers import CrossEncoder
from app.services.rag.protocols import SearchResult

class Reranker:
    """
    Rerank search results using cross-encoder.

    Cross-encoders process query+document together, giving better
    relevance scores than separate embeddings (bi-encoders).

    Typical workflow:
    1. Retrieve top-20 with vector search (fast)
    2. Rerank top-20 with cross-encoder (slow but accurate)
    3. Return top-5
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize reranker.

        Model options:
        - ms-marco-MiniLM-L-6-v2: Fast, 40MB, good quality
        - ms-marco-MiniLM-L-12-v2: Slower, better quality
        - Or use Cohere Rerank API for best quality
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Rerank search results.

        Args:
            query: User query
            results: Initial search results
            top_k: Number of top results to return

        Returns:
            Reranked results (top_k)
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [[query, result.content] for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Sort by score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update scores and return top-k
        reranked = []
        for result, score in scored_results[:top_k]:
            result.score = float(score)
            reranked.append(result)

        return reranked
```

**Dependencies**:
```bash
pip install sentence-transformers
```

**Expected Impact**:
- ✅ 20-30% improvement in precision
- ✅ Filters false positives from vector search
- ⚠️ Adds 50-150ms latency per search

---

## 🎨 Phase 3: Production Polish (Week 3+)

### 3.1 Adaptive Chunking Strategy

```python
class AdaptiveChunker:
    """
    Automatically selects optimal chunking strategy based on content.

    Decision tree:
    - Code files → AST-based chunking (functions/classes)
    - Markdown → Header-based chunking
    - Conversational → Q&A pair chunking
    - Dense prose → Semantic similarity chunking
    - Mixed → Token-based with overlap
    """

    async def chunk_document(
        self,
        document: Document,
    ) -> List[DocumentChunk]:
        """Automatically select and apply best chunking strategy."""

        # Detect content type
        content_type = self._detect_content_type(document)

        # Route to appropriate chunker
        if content_type == "code":
            return await self.structure_aware_splitter.split_text(
                document.content,
                content_type=self._detect_language(document),
                metadata=document.metadata,
            )
        elif content_type == "markdown":
            return self.markdown_splitter.split_text(
                document.content,
                metadata=document.metadata,
            )
        elif content_type == "conversation":
            return self.conversation_splitter.split_text(
                document.content,
                metadata=document.metadata,
            )
        else:
            # Default: semantic chunking
            return await self.semantic_splitter.split_text(
                document.content,
                metadata=document.metadata,
            )
```

---

### 3.2 Parent Document Retrieval

```python
class ParentDocumentRetriever:
    """
    Retrieve small chunks, return large contexts.

    Strategy:
    - Index small chunks (256 tokens) for precision
    - Store parent chunks (1024 tokens) for context
    - Search with small, return parents
    """

    async def index_with_parents(
        self,
        document: Document,
        collection_name: str,
    ) -> None:
        """Index document with small chunks and large parents."""

        # Create small chunks
        small_chunks = self.small_chunker.split_text(
            document.content,
            chunk_size_tokens=256,
        )

        # Create parent chunks
        parent_chunks = self.large_chunker.split_text(
            document.content,
            chunk_size_tokens=1024,
        )

        # Map small chunks to parents
        for small in small_chunks:
            # Find overlapping parent
            parent = self._find_parent_chunk(small, parent_chunks)

            # Store small chunk with parent reference
            small.metadata["parent_id"] = parent.id
            small.metadata["parent_content"] = parent.content

            await self.vector_store.upsert(collection_name, small)
```

---

## 📊 Expected Performance Improvements

| Technique | Precision Gain | Recall Gain | Latency Impact | Implementation Effort | Priority |
|-----------|---------------|-------------|----------------|----------------------|----------|
| **Token-based chunking** | +5% | +5% | 0ms | 6h | ⭐⭐⭐ |
| **Contextual Retrieval** | +25% | +15% | 0ms (one-time cost) | 10h | ⭐⭐⭐ |
| **Hybrid search** | +15% | +25% | +10-20ms | 8h | ⭐⭐⭐ |
| **Query expansion** | +10% | +20% | +100-200ms | 6h | ⭐⭐ |
| **Semantic chunking** | +10% | +10% | 0ms (index time) | 12h | ⭐⭐ |
| **Late chunking** | +15% | +10% | 0ms (index time) | 16h | ⭐⭐ |
| **Reranking** | +20% | +5% | +50-150ms | 6h | ⭐⭐⭐ |

**Total Expected Improvement**: 50-80% better retrieval quality

---

## 🧪 Evaluation & Testing Strategy

### 1. Create Baseline Metrics

```python
# scripts/evaluate_rag.py
"""Evaluate RAG performance before and after improvements."""

from app.evaluation.runner import EvaluationRunner
from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# Load test queries
test_queries = [
    {
        "query": "How do I reset my password?",
        "relevant_docs": ["doc_123", "doc_456"],
        "category": "factual",
    },
    {
        "query": "What are the differences between starter and pro plans?",
        "relevant_docs": ["doc_789"],
        "category": "comparison",
    },
    # ... 50-100 test queries
]

# Evaluate current system
runner = EvaluationRunner(rag_service)
results = await runner.evaluate(test_queries)

print(f"Precision@5: {results['precision@5']:.2%}")
print(f"Recall@5: {results['recall@5']:.2%}")
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

### 2. A/B Testing Framework

```python
# app/services/rag/ab_testing.py
"""A/B testing for RAG improvements."""

class RAGABTest:
    """Compare two RAG configurations side-by-side."""

    def __init__(self, rag_service_a: RAGService, rag_service_b: RAGService):
        self.service_a = rag_service_a  # Current system
        self.service_b = rag_service_b  # Improved system

    async def compare(
        self,
        query: str,
        relevant_docs: List[str],
    ) -> Dict[str, Any]:
        """Compare both systems on a single query."""

        # Run both systems
        results_a = await self.service_a.search(query, limit=5)
        results_b = await self.service_b.search(query, limit=5)

        # Calculate metrics
        return {
            "query": query,
            "system_a": {
                "precision": self._precision(results_a, relevant_docs),
                "recall": self._recall(results_a, relevant_docs),
                "latency_ms": results_a.latency_ms,
            },
            "system_b": {
                "precision": self._precision(results_b, relevant_docs),
                "recall": self._recall(results_b, relevant_docs),
                "latency_ms": results_b.latency_ms,
            },
        }
```

---

## 🛠 Implementation Checklist

### Phase 1 (Week 1)
- [ ] Add tiktoken dependency
- [ ] Implement TokenAwareTextSplitter
- [ ] Update config.py with token-based settings
- [ ] Implement ContextualChunker with Anthropic API
- [ ] Add anthropic client to dependencies
- [ ] Implement HybridSearchService with BM25
- [ ] Add rank-bm25 dependency
- [ ] Implement QueryProcessor for query expansion
- [ ] Update RAGService to use new chunkers
- [ ] Add configuration flags for toggling features
- [ ] Create evaluation baseline (run current metrics)
- [ ] Test token-based chunking vs character-based
- [ ] Test contextual retrieval impact
- [ ] Test hybrid search vs pure semantic

### Phase 2 (Week 2)
- [ ] Implement SemanticTextSplitter with embedding similarity
- [ ] Implement StructureAwareSplitter for code/markdown
- [ ] Add tree-sitter for AST parsing (optional)
- [ ] Implement Reranker with cross-encoder
- [ ] Add sentence-transformers dependency
- [ ] Implement LateChunker (optional, advanced)
- [ ] Update RAGService to support multiple chunking strategies
- [ ] Add content-type detection logic
- [ ] Create A/B testing framework
- [ ] Run comparative evaluation (old vs new)

### Phase 3 (Week 3+)
- [ ] Implement AdaptiveChunker with auto-detection
- [ ] Implement ParentDocumentRetriever
- [ ] Add query routing logic
- [ ] Implement metadata-aware ranking
- [ ] Add user feedback collection
- [ ] Create monitoring dashboard
- [ ] Production testing with real queries
- [ ] Performance optimization
- [ ] Documentation updates

---

## 💰 Cost Analysis

### One-Time Costs (Document Ingestion)

| Component | Cost per 1M Tokens | Notes |
|-----------|-------------------|-------|
| Token-based chunking | $0 | Free (local tiktoken) |
| Contextual prefixes | ~$1.02 | Claude with prompt caching |
| Semantic embeddings | $0.13 | OpenAI text-embedding-3-small |
| BM25 indexing | $0 | Free (local computation) |
| Late chunking | ~$0.50 | Local model (GPU optional) |

**Total one-time cost**: ~$1.65 per million document tokens

### Per-Query Costs (Retrieval)

| Component | Cost per 1K Queries | Latency Impact |
|-----------|-------------------|----------------|
| Vector search | $0 | +10ms |
| BM25 search | $0 | +5ms |
| Query expansion | ~$0.05 | +100-200ms |
| Reranking (local) | $0 | +50-150ms |
| Reranking (Cohere API) | $1.00 | +200-300ms |

**Recommended**: Local reranking for cost-effectiveness

---

## 📚 Key References

1. **Anthropic Contextual Retrieval (2024)**
   - https://www.anthropic.com/news/contextual-retrieval
   - 67% reduction in retrieval failures

2. **Late Chunking (2025)**
   - arXiv:2409.04701
   - Contextual chunk embeddings

3. **Semantic Chunking Research (2024-2025)**
   - Max-Min semantic chunking (Springer, Jan 2025)
   - LangChain/LlamaIndex implementations

4. **Hybrid Search Best Practices**
   - Weaviate RAG guide
   - Zilliz chunking strategies

5. **Agentic Chunking**
   - IBM Think tutorial
   - LLM-driven chunking decisions

---

## 🤔 Discussion Questions

### 1. Which phase should we prioritize?

**My recommendation**: Start with Phase 1 - highest ROI for effort

- Token-based chunking (6h) - Foundation for everything else
- Contextual Retrieval (10h) - Biggest single improvement (35-67%)
- Hybrid search (8h) - Essential for production
- Query expansion (6h) - Quick win

Total: ~30 hours, 30-50% expected improvement

### 2. Budget considerations?

- Contextual retrieval: One-time $1/million tokens (very affordable)
- Query expansion: ~$0.05/1k queries (minimal)
- Reranking: Free if using local model, $1/1k if using Cohere API

**Recommendation**: Use local reranking to keep costs near zero

### 3. Latency targets?

Current: ~100-200ms
With improvements: ~200-400ms

If latency is critical:
- Skip query expansion (saves 100-200ms)
- Use local reranking (saves 100-150ms vs API)
- Consider caching popular queries

### 4. Content characteristics?

What types of documents are most common?
- **Technical docs**: Prioritize structure-aware + code chunking
- **Conversational**: Prioritize query expansion + contextual retrieval
- **Mixed**: Prioritize hybrid search + adaptive chunking

### 5. Evaluation approach?

- [ ] Build test query set (50-100 queries)
- [ ] Establish baseline metrics
- [ ] Implement improvements incrementally
- [ ] A/B test each improvement
- [ ] Monitor production metrics

---

## 🚀 Next Steps

1. **Review this proposal** and prioritize techniques
2. **Create test query set** for evaluation (essential!)
3. **Establish baseline metrics** with current system
4. **Start Phase 1** implementation:
   - Token-based chunking (highest priority)
   - Contextual Retrieval (biggest impact)
   - Hybrid search (essential for production)
5. **Measure improvement** after each change
6. **Iterate** based on results

---

**Questions? Let's discuss which improvements are most valuable for your use case!**
