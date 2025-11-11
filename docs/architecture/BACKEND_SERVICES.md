# Backend Services Documentation

**Last Updated:** November 10, 2025

This document provides detailed documentation for all RAG-related and memory system backend services and components.

---

## Table of Contents

- [Overview](#overview)
- [Memory Service](#memory-service) ⭐ **NEW**
- [RAG Service Orchestrator](#rag-service-orchestrator)
- [Embeddings Service](#embeddings-service)
- [Vector Store Service](#vector-store-service)
- [Text Splitter](#text-splitter)
- [Hybrid Search](#hybrid-search)
- [BM25 Implementations](#bm25-implementations)
- [Reranker](#reranker)
- [Deduplication](#deduplication)
- [Document Loaders](#document-loaders)
- [Protocols](#protocols)
- [Configuration](#configuration)

---

## Overview

The RAG system is built with a **provider pattern** architecture, allowing components to be easily swapped or extended. All components implement protocol interfaces defined in `app/services/rag/protocols.py`.

### Component Hierarchy

```
MemoryService (Intelligent User Memory)
├── EmbeddingProvider (OpenAI)
├── VectorStore (Qdrant - user_memories collection)
└── LLM (GPT-4 Turbo for extraction)

RAGService (Orchestrator)
├── EmbeddingProvider (OpenAI)
├── VectorStore (Qdrant)
├── TextSplitter (SmartTextSplitter)
├── HybridSearch
│   ├── SemanticSearch (Qdrant)
│   └── BM25Index (PostgreSQL or In-Memory)
├── Reranker (Cross-Encoder)
├── Deduplicator (MMR + Token-based)
└── DocumentLoaders (PDF, DOCX, Text, Web)
```

---

## Memory Service

**File:** `backend/app/services/memory_service.py`

The memory service intelligently extracts, stores, and retrieves user preferences, facts, and context from conversations and journal entries.

### Overview

The memory system enables personalized, context-aware responses by:
- Automatically extracting important facts from conversations and journal entries
- Storing memories with semantic embeddings in Qdrant
- Retrieving relevant memories based on multi-factor scoring
- Deduplicating similar memories to avoid redundancy
- Injecting memory context into chat sessions

### Initialization

```python
from app.services.memory_service import MemoryService
from app.database import get_session

async with get_session() as db:
    memory_service = MemoryService(db=db)
```

### Key Methods

#### `add_memory()`

Manually add a user memory.

```python
memory = await memory_service.add_memory(
    user_id=user.id,
    content="User prefers Python for backend development",
    importance=0.8,  # 0.0-1.0 importance score
    category="preference",  # preference, fact, goal, context
    source_conversation_id=conversation_id  # Optional
)
```

**Process:**
1. Generate embedding for memory content
2. Check for similar existing memories (deduplication)
3. Store in PostgreSQL (`user_memories` table)
4. Store embedding in Qdrant (`user_memories` collection)
5. Return created/updated memory

#### `retrieve_memories()`

Retrieve relevant memories for a query.

```python
memories = await memory_service.retrieve_memories(
    user_id=user.id,
    query="What programming languages does the user know?",
    limit=10
)
```

**Multi-Factor Scoring:**
```python
final_score = (
    semantic_weight * semantic_similarity +
    recency_weight * recency_score +
    importance_weight * importance_score
)
```

Default weights:
- Semantic: 0.6
- Recency: 0.25
- Importance: 0.15

#### `extract_memories_from_conversation()`

Automatically extract memories from conversation history using LLM.

```python
new_memories = await memory_service.extract_memories_from_conversation(
    conversation_id=conversation.id,
    user_id=user.id,
    limit_messages=20  # Analyze last 20 messages
)
```

**Process:**
1. Fetch recent conversation messages
2. Get existing memories for context (avoid duplicates)
3. Call GPT-4 with extraction prompt
4. Parse JSON response with extracted memories
5. Store each memory (with automatic deduplication)

#### `extract_memories_from_journal()`

Extract memories from journal entries.

```python
new_memories = await memory_service.extract_memories_from_journal(
    user_id=user.id,
    days_back=7  # Analyze last 7 days
)
```

**Process:**
1. Fetch recent journal entries
2. Combine entries with date headers
3. Call GPT-4 for extraction
4. Store extracted memories

#### `update_memory()`

Update memory content or importance.

```python
updated = await memory_service.update_memory(
    memory_id=memory.id,
    user_id=user.id,
    content="Updated content",
    importance=0.9
)
```

#### `delete_memory()`

Delete a specific memory.

```python
success = await memory_service.delete_memory(
    memory_id=memory.id,
    user_id=user.id
)
```

### Memory Deduplication

The service automatically prevents duplicate memories using semantic similarity:

```python
# In add_memory()
similar_memory = await self._find_similar_memory(
    user_id=user_id,
    content=content,
    embedding=embedding
)

if similar_memory and similarity > threshold:
    # Update existing memory instead of creating new one
    return await self.update_memory(...)
```

**Threshold:** 0.85 (configurable via `MEMORY_SIMILARITY_THRESHOLD`)

### Configuration

All memory settings are in `backend/app/config.py`:

```python
# Memory System settings
ENABLE_MEMORY: bool = True
MEMORY_COLLECTION_NAME: str = "user_memories"
MEMORY_RETRIEVAL_LIMIT: int = 10
MEMORY_SIMILARITY_THRESHOLD: float = 0.85
MEMORY_EXTRACTION_MODEL: str = "gpt-4-turbo-preview"

# Memory scoring weights
MEMORY_SEMANTIC_WEIGHT: float = 0.6
MEMORY_RECENCY_WEIGHT: float = 0.25
MEMORY_IMPORTANCE_WEIGHT: float = 0.15
```

### Usage in Chat

Memories are automatically injected into chat context:

```python
# In chat API (api/chat.py)
if use_memory:
    memory_context = await get_memory_context(
        db=db,
        user_id=user.id,
        query=user_query
    )
    # memory_context is formatted and added to system message
```

### Database Schema

```sql
CREATE TABLE user_memories (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    category VARCHAR(50),
    qdrant_id VARCHAR(255) UNIQUE,
    source_conversation_id UUID,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (source_conversation_id) REFERENCES conversations(id)
);

-- Indexes for performance
CREATE INDEX idx_user_memories_user_id ON user_memories(user_id);
CREATE INDEX idx_user_memories_importance ON user_memories(importance DESC);
CREATE INDEX idx_user_memories_updated_at ON user_memories(updated_at DESC);
```

### API Endpoints

**File:** `backend/app/api/memory.py`

```python
# List memories
GET /api/memory/?limit=100

# Create memory
POST /api/memory/
{
  "content": "User prefers Python",
  "importance": 0.8,
  "category": "preference"
}

# Update memory
PUT /api/memory/{memory_id}?content=New+content&importance=0.9

# Delete memory
DELETE /api/memory/{memory_id}

# Delete all memories
DELETE /api/memory/

# Extract from conversation
POST /api/memory/extract/conversation/{conversation_id}

# Extract from journal
POST /api/memory/extract/journal?days_back=7

# Search memories
GET /api/memory/search?query=programming&limit=10
```

### Testing

Comprehensive test suite in `backend/tests/`:

```bash
# Run memory tests
pytest tests/test_memory_service.py  # 26 tests
pytest tests/api/test_memory.py      # 26 tests

# All 52 tests passing ✅
```

### Performance Considerations

1. **Batch Extraction:** Extract from multiple journal entries at once
2. **Deduplication:** Prevents memory bloat (threshold: 0.85 similarity)
3. **Caching:** Retrieved memories include timestamps for recency scoring
4. **Indexing:** PostgreSQL indexes on user_id, importance, updated_at
5. **Qdrant Storage:** UUID-based point IDs for efficient updates

---

## RAG Service Orchestrator

**File:** `backend/app/services/rag_service.py`

The main orchestration layer that coordinates all RAG operations.

### Initialization

```python
from app.services.rag_service import RAGService
from app.database import get_session

async with get_session() as db:
    rag_service = RAGService(
        db_session=db,  # Required for PostgreSQL BM25
        enable_hybrid_search=True  # Enable hybrid search
    )
```

### Key Methods

#### `ingest_document()`

Processes and indexes a document.

```python
await rag_service.ingest_document(
    source="document.pdf",          # File path or URL
    collection_name="my_collection", # Knowledge pool
    source_type="upload"             # "upload", "web", etc.
)
```

**Process:**

1. Detect loader based on file extension
2. Load document content
3. Split into chunks
4. Generate embeddings (batch)
5. Store in Qdrant (vector store)
6. Store in PostgreSQL (BM25 index if hybrid search enabled)

#### `search()`

Searches for relevant documents.

```python
results = await rag_service.search(
    query="How does Python handle memory?",
    collection_name="my_collection",
    limit=10,
    enable_reranking=True,
    enable_deduplication=True
)
```

**Process:**

1. If hybrid search enabled:
   - Parallel semantic search (Qdrant) + BM25 search (PostgreSQL)
   - Combine with Reciprocal Rank Fusion
2. Else: Semantic search only
3. Optional: Rerank with cross-encoder
4. Optional: Deduplicate results
5. Return top K results

#### `delete_collection()`

Removes all documents from a collection.

```python
await rag_service.delete_collection(collection_name="my_collection")
```

**Process:**

1. Delete from Qdrant
2. Delete from BM25 index (if enabled)

---

## Embeddings Service

**File:** `backend/app/services/rag/embeddings.py`

Handles text-to-vector conversion using OpenAI embeddings.

### OpenAIEmbeddings

```python
from app.services.rag.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Default
    api_key=settings.OPENAI_API_KEY   # From config
)
```

#### Methods

```python
# Single text embedding
vector = await embeddings.embed_text("Hello world")
# Returns: List[float] with 1536 dimensions

# Batch embedding (efficient)
vectors = await embeddings.embed_batch([
    "Text 1",
    "Text 2",
    "Text 3"
])
# Returns: List[List[float]]
```

**Performance Notes:**

- Batch processing is ~5x faster than sequential
- OpenAI rate limits apply (500 requests/min on tier 1)
- Automatic retries with exponential backoff

---

## Vector Store Service

**File:** `backend/app/services/rag/vector_store.py`

Manages vector storage and retrieval in Qdrant.

### QdrantVectorStore

```python
from app.services.rag.vector_store import QdrantVectorStore

vector_store = QdrantVectorStore(
    url=settings.QDRANT_URL,        # Default: http://localhost:6333
    api_key=settings.QDRANT_API_KEY  # Optional
)
```

#### Methods

##### `ensure_collection()`

Creates collection if it doesn't exist.

```python
await vector_store.ensure_collection(
    collection_name="my_collection",
    vector_size=1536  # OpenAI embedding dimension
)
```

##### `upsert()`

Inserts or updates document vectors.

```python
await vector_store.upsert(
    collection_name="my_collection",
    documents=[
        {
            "id": "doc1_chunk0",
            "vector": [0.1, 0.2, ...],  # 1536-dim vector
            "metadata": {
                "content": "Chunk text...",
                "source": "document.pdf",
                "page": 1,
                "chunk_index": 0
            }
        }
    ]
)
```

##### `search()`

Performs semantic similarity search.

```python
results = await vector_store.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],  # Query embedding
    limit=10
)
```

**Returns:** `List[SearchResult]` with `metadata` and `score`

---

## Text Splitter

**File:** `backend/app/services/rag/text_splitter.py`

Splits documents into chunks while preserving semantic coherence.

### SmartTextSplitter

```python
from app.services.rag.text_splitter import SmartTextSplitter

splitter = SmartTextSplitter(
    chunk_size=1000,     # Characters
    chunk_overlap=200    # Characters
)

chunks = splitter.split_text(text="Long document...")
```

#### Features

- **Hierarchical Splitting:** Tries to split on paragraphs, then sentences, then words
- **Separator Hierarchy:** `["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]`
- **Overlap:** Preserves context between chunks
- **Metadata:** Each chunk includes `chunk_index` and `num_chunks`

#### Split Documents

```python
from app.services.rag.protocols import Document

docs = [
    Document(content="Text 1", metadata={"source": "doc1.pdf"}),
    Document(content="Text 2", metadata={"source": "doc2.pdf"})
]

chunks = splitter.split_documents(docs)
# Returns: List[Document] with chunked content
```

---

## Hybrid Search

**File:** `backend/app/services/rag/hybrid_search.py`

Combines semantic (vector) and keyword (BM25) search for improved retrieval.

### HybridSearchService

```python
from app.services.rag.hybrid_search import HybridSearchService

hybrid = HybridSearchService(
    vector_store=vector_store,
    bm25_index=bm25_index,
    embedding_provider=embeddings,
    semantic_weight=0.5,  # Weight for vector search
    keyword_weight=0.5,   # Weight for BM25 search
    rrf_k=60              # RRF parameter
)
```

#### Method: `search()`

```python
results = await hybrid.search(
    query="How does Python handle memory?",
    collection_name="my_collection",
    limit=10,
    retrieval_k=20  # Retrieve 20 from each source, then fuse
)
```

**Algorithm:**

1. **Parallel Search:**
   - Semantic: Retrieve top `retrieval_k` by vector similarity
   - BM25: Retrieve top `retrieval_k` by keyword relevance

2. **Reciprocal Rank Fusion (RRF):**

   ```python
   score(doc) = semantic_weight * rrf_score(doc, semantic_results) +
                keyword_weight * rrf_score(doc, bm25_results)

   rrf_score(doc, results) = sum(1 / (k + rank(doc)))
   ```

3. **Sort & Limit:** Return top `limit` documents

**Benefits:**

- Catches documents missed by semantic-only search
- Handles exact term matches better
- Improves recall by 20-40% in practice

---

## BM25 Implementations

Two implementations available:

### 1. PostgreSQL BM25 (Recommended)

**File:** `backend/app/services/rag/postgres_bm25.py`

Uses PostgreSQL Full-Text Search with `tsvector` and GIN indexes.

```python
from app.services.rag.postgres_bm25 import PostgresBM25Index

bm25 = PostgresBM25Index(db_session=db)

# Index documents
await bm25.index_documents(
    collection_name="my_collection",
    chunks=[
        {
            "document_id": "doc1",
            "chunk_index": 0,
            "content": "Chunk text...",
            "metadata": {"page": 1}
        }
    ]
)

# Search
results = await bm25.search(
    collection_name="my_collection",
    query="Python memory",
    limit=10
)
```

**Advantages:**

- ✅ Persistent (survives restarts)
- ✅ Scalable (millions of documents)
- ✅ ACID guarantees
- ✅ Automatic tsvector updates via PostgreSQL trigger

**How it Works:**

- `to_tsvector('english', content)` creates searchable tokens
- `ts_rank_cd()` ranks results (similar to BM25)
- GIN index on `content_tsv` for fast lookups

### 2. In-Memory BM25 (Fallback)

**File:** `backend/app/services/rag/bm25_index.py`

Python-based BM25 implementation (rank-bm25 library).

```python
from app.services.rag.bm25_index import BM25Index

bm25 = BM25Index()

# Same API as PostgresBM25Index
```

**Advantages:**

- ✅ Fast search (in-memory)
- ✅ No database required

**Disadvantages:**

- ❌ Lost on restart
- ❌ Limited by RAM

**Use Cases:**

- Development/testing
- Small datasets (<10k documents)
- Fallback when PostgreSQL unavailable

---

## Reranker

**File:** `backend/app/services/rag/reranker.py`

Re-scores search results using a cross-encoder model for better relevance.

### Reranker

```python
from app.services.rag.reranker import get_reranker

reranker = get_reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default
)

# Rerank search results
reranked = await reranker.rerank(
    query="How does Python handle memory?",
    results=search_results,
    top_k=10
)
```

**How it Works:**

1. **Cross-Encoder:** Processes query + document pairs together
2. **Scoring:** Assigns relevance score (0-1) to each pair
3. **Re-sorting:** Returns top K by new scores

**Model Options:**

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, lightweight)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (more accurate, slower)
- Custom models via HuggingFace

**Performance:**

- Improves nDCG by 10-20%
- Adds ~50-200ms latency depending on model
- Can process 100+ candidates efficiently

---

## Deduplication

**File:** `backend/app/services/rag/deduplication.py`

Removes redundant or highly similar chunks from search results.

### Methods

#### 1. Token-Based Deduplication

```python
from app.services.rag.deduplication import TokenDeduplicator

dedup = TokenDeduplicator(
    threshold=0.8  # Remove if >80% token overlap
)

deduplicated = dedup.deduplicate(results)
```

**Algorithm:**

- Tokenize each document
- Calculate Jaccard similarity between token sets
- Remove documents exceeding threshold

#### 2. MMR Deduplication

```python
from app.services.rag.deduplication import MMRDeduplicator

dedup = MMRDeduplicator(
    lambda_param=0.7,  # Diversity vs relevance (0-1)
    similarity_threshold=0.85
)

deduplicated = await dedup.deduplicate(
    results=results,
    embeddings=embeddings_provider
)
```

**Algorithm (Maximal Marginal Relevance):**

```python
score(doc) = lambda * relevance(doc) - (1-lambda) * max_similarity(doc, selected_docs)
```

- Balances relevance and diversity
- Avoids selecting near-duplicate chunks
- Iteratively selects documents with highest MMR score

**When to Use:**

- Token-based: Fast, good for exact duplicates
- MMR: Better for semantic duplicates, slower

---

## Document Loaders

**Directory:** `backend/app/services/rag/loaders/`

Extract text from various document formats.

### Base Protocol

```python
from app.services.rag.protocols import DocumentLoader

class CustomLoader(DocumentLoader):
    def supports(self, source: str) -> bool:
        return source.endswith(".custom")

    async def load(self, source: str) -> List[Document]:
        # Load and return documents
        pass
```

### Implemented Loaders

#### PDFLoader

**File:** `loaders/pdf.py`

```python
from app.services.rag.loaders import PDFLoader

loader = PDFLoader()
docs = await loader.load("document.pdf")
```

- Uses `PyPDF2` library
- Extracts text page by page
- Metadata includes page numbers

#### DocxLoader

**File:** `loaders/docx.py`

```python
from app.services.rag.loaders import DocxLoader

loader = DocxLoader()
docs = await loader.load("document.docx")
```

- Uses `python-docx` library
- Extracts paragraphs and tables
- Preserves some formatting

#### TextLoader

**File:** `loaders/text.py`

```python
from app.services.rag.loaders import TextLoader

loader = TextLoader()
docs = await loader.load("document.txt")
```

- Supports: `.txt`, `.md`, `.json`, `.csv`
- Simple file reading
- UTF-8 encoding

#### WebLoader

**File:** `loaders/web.py`

```python
from app.services.rag.loaders import WebLoader

loader = WebLoader()
docs = await loader.load("https://example.com/article")
```

- Uses `httpx` + `BeautifulSoup4`
- Extracts main content
- Removes scripts, styles, navigation

---

## Protocols

**File:** `backend/app/services/rag/protocols.py`

Type protocols define interfaces for all components.

### Key Protocols

```python
from typing import Protocol, List

class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

class VectorStore(Protocol):
    async def ensure_collection(self, name: str, size: int): ...
    async def upsert(self, collection: str, documents: List[dict]): ...
    async def search(self, collection: str, vector: List[float], limit: int): ...
    async def delete_collection(self, collection: str): ...

class TextSplitter(Protocol):
    def split_text(self, text: str) -> List[str]: ...
    def split_documents(self, documents: List[Document]) -> List[Document]: ...

class DocumentLoader(Protocol):
    def supports(self, source: str) -> bool: ...
    async def load(self, source: str) -> List[Document]: ...
```

**Benefits:**

- Type safety
- Easy to swap implementations
- Clear contracts for new components

---

## Configuration

**File:** `backend/app/config.py`

All RAG settings are configured via environment variables.

### Key Settings

```python
class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None

    # Hybrid Search
    ENABLE_HYBRID_SEARCH: bool = True
    BM25_BACKEND: str = "postgresql"  # or "memory"
    HYBRID_SEARCH_SEMANTIC_WEIGHT: float = 0.5
    HYBRID_SEARCH_KEYWORD_WEIGHT: float = 0.5
    HYBRID_SEARCH_RRF_K: int = 60
    HYBRID_SEARCH_RETRIEVAL_K: int = 20

    # Reranking
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Text Splitting
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
```

### Example `.env`

```bash
# Required
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat
QDRANT_URL=http://localhost:6333

# Optional (with defaults)
ENABLE_HYBRID_SEARCH=true
BM25_BACKEND=postgresql
HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5
HYBRID_SEARCH_KEYWORD_WEIGHT=0.5
```

---

## Usage Examples

### Full RAG Pipeline

```python
from app.services.rag_service import RAGService
from app.database import get_session

async def rag_example():
    async with get_session() as db:
        # Initialize service
        rag = RAGService(
            db_session=db,
            enable_hybrid_search=True
        )

        # 1. Ingest document
        await rag.ingest_document(
            source="research_paper.pdf",
            collection_name="research_pool"
        )

        # 2. Search
        results = await rag.search(
            query="What are the main findings?",
            collection_name="research_pool",
            limit=5,
            enable_reranking=True,
            enable_deduplication=True
        )

        # 3. Use results in LLM context
        context = "\n\n".join([r.metadata["content"] for r in results])

        # 4. Delete collection when done
        # await rag.delete_collection("research_pool")
```

### Custom Component

```python
from app.services.rag_service import RAGService
from app.services.rag.embeddings import OpenAIEmbeddings

# Use custom embeddings
custom_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Higher dimension
    api_key="sk-custom-key"
)

rag = RAGService(
    embedding_provider=custom_embeddings
)
```

---

## Performance Tips

1. **Batch Embeddings:** Always use `embed_batch()` for multiple texts
2. **Hybrid Search:** Enable for 20-40% better recall
3. **Reranking:** Use on final top 20-50 results (not all candidates)
4. **Deduplication:** Apply after reranking to avoid wasting context
5. **Chunk Size:** Tune based on your documents (500-1500 characters typical)
6. **PostgreSQL BM25:** Use over in-memory for production

---

## Troubleshooting

### Issue: Slow Search

**Check:**

- Are you using hybrid search? (Adds ~100-300ms)
- Is reranking enabled? (Adds ~50-200ms per 10 results)
- Is PostgreSQL BM25 index created? (`idx_bm25_documents_fts`)

**Solution:**

- Disable hybrid search for speed: `ENABLE_HYBRID_SEARCH=false`
- Reduce `retrieval_k` in hybrid search
- Use lighter reranker model

### Issue: Poor Retrieval Quality

**Check:**

- Is hybrid search enabled? (Improves recall)
- Are chunks too large/small?
- Is reranking enabled?

**Solution:**

- Enable hybrid search: `ENABLE_HYBRID_SEARCH=true`
- Tune chunk size (try 800-1200)
- Enable reranking for top results

### Issue: BM25 No Results

**Check:**

- Is PostgreSQL BM25 backend configured?
- Are documents indexed in `bm25_documents` table?
- Is trigger installed? (Auto-updates `content_tsv`)

**Solution:**

```sql
-- Check if documents indexed
SELECT COUNT(*) FROM bm25_documents;

-- Check if trigger exists
SELECT tgname FROM pg_trigger WHERE tgname = 'bm25_documents_content_tsv_update';
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Project Status](../status/PROJECT_STATUS.md)
- [PostgreSQL BM25 Migration](../plans/archive/POSTGRES_BM25_MIGRATION.md)
- [RAG Improvements Reference](../reference/RAG_IMPROVEMENTS.md)

---

**Last Updated:** November 7, 2024
