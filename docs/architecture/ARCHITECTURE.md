# RAG System Architecture

**Last Updated:** November 7, 2024
**Status:** Production-Ready MVP

This document provides a comprehensive overview of the RAG Chat System architecture, focusing on what is **actually implemented** (not planned features).

---

## Table of Contents

- [System Overview](#system-overview)
- [Technology Stack](#technology-stack)
- [Architecture Diagram](#architecture-diagram)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Data Flow](#data-flow)
- [Database Schema](#database-schema)
- [RAG Pipeline](#rag-pipeline)
- [Security](#security)
- [Deployment](#deployment)

---

## System Overview

The RAG Chat System is a production-ready conversational AI application that combines:

- **Retrieval-Augmented Generation (RAG)** for grounding responses in uploaded documents
- **Hybrid Search** (semantic + keyword) for optimal retrieval accuracy
- **Integrated Scratchpad** for todos, notes, and journal entries
- **Multi-User Support** with JWT authentication
- **Knowledge Pools** for organizing documents by topic
- **Advanced RAG Features** including reranking, deduplication, and query rewriting

---

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | Latest | Async web framework |
| **Language** | Python | 3.13+ | Backend programming |
| **Database** | PostgreSQL | Latest | Relational data storage |
| **Vector Store** | Qdrant | Latest | Vector embeddings storage |
| **Cache** | Redis | Latest | Session management (configured) |
| **Migrations** | Alembic | Latest | Database schema versioning |
| **Embeddings** | OpenAI | text-embedding-3-small | Vector generation |
| **LLM** | OpenAI | GPT-4 | Chat completions |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | Next.js | 15 | React framework with App Router |
| **Language** | TypeScript | Latest | Type-safe development |
| **UI Library** | Tailwind CSS + shadcn/ui | Latest | Styling and components |
| **State Management** | Zustand | Latest | Global state |
| **Streaming** | Vercel AI SDK | Latest | SSE streaming support |
| **Layout** | react-resizable-panels | Latest | Split-pane interface |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Containerization** | Docker + Docker Compose | Local development |
| **Package Manager (Python)** | uv | Fast Python package management |
| **Package Manager (Node)** | npm | Node.js dependencies |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js 15)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Conversations│  │   Chat UI    │  │     Scratchpad       │  │
│  │   Sidebar    │  │  + Streaming │  │  (Todos/Notes/RAG)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                  │                      │              │
│         └──────────────────┴──────────────────────┘              │
│                            │                                     │
│                    ┌───────▼──────┐                             │
│                    │  Zustand     │                             │
│                    │   Stores     │                             │
│                    └───────┬──────┘                             │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTPS/SSE
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Layer                              │  │
│  │  /auth  /chat  /conversations  /scratchpad  /rag         │  │
│  └────┬──────────┬──────────┬──────────┬──────────┬─────────┘  │
│       │          │          │          │          │             │
│  ┌────▼──────────▼──────────▼──────────▼──────────▼─────────┐  │
│  │                  Service Layer                            │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │           RAG Service (Orchestration)               │ │  │
│  │  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │ │  │
│  │  │  │ Embeddings  │  │ Vector Store │  │   Loaders  │ │ │  │
│  │  │  │  (OpenAI)   │  │   (Qdrant)   │  │ (PDF/DOCX) │ │ │  │
│  │  │  └─────────────┘  └──────────────┘  └────────────┘ │ │  │
│  │  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │ │  │
│  │  │  │Text Splitter│  │Hybrid Search │  │  Reranker  │ │ │  │
│  │  │  └─────────────┘  └──────────────┘  └────────────┘ │ │  │
│  │  │  ┌─────────────┐  ┌──────────────┐                 │ │  │
│  │  │  │BM25 (PG FTS)│  │Deduplication │                 │ │  │
│  │  │  └─────────────┘  └──────────────┘                 │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘│
│       │                                                         │
└───────┼─────────────────────────────────────────────────────────┘
        │
        │ SQL/async
        ▼
┌───────────────────────────────────────────┐
│         PostgreSQL Database               │
│  ┌────────────────────────────────────┐   │
│  │ Tables:                            │   │
│  │  - users                           │   │
│  │  - conversations                   │   │
│  │  - messages                        │   │
│  │  - scratchpad_entries              │   │
│  │  - knowledge_pools                 │   │
│  │  - documents                       │   │
│  │  - user_memories                   │   │
│  │  - bm25_documents (FTS index)      │   │
│  └────────────────────────────────────┘   │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│         Qdrant Vector Store               │
│  ┌────────────────────────────────────┐   │
│  │ Collections (per knowledge pool):  │   │
│  │  - Document vectors (1536 dims)    │   │
│  │  - Metadata (source, page, etc.)   │   │
│  └────────────────────────────────────┘   │
└───────────────────────────────────────────┘
```

---

## Backend Architecture

### Directory Structure

```
backend/
├── app/
│   ├── api/                    # API route handlers
│   │   ├── auth.py            # JWT authentication
│   │   ├── chat.py            # Streaming chat endpoint
│   │   ├── conversations.py   # Conversation CRUD
│   │   ├── scratchpad.py      # Scratchpad CRUD
│   │   └── rag.py             # RAG endpoints (upload, pools, search)
│   │
│   ├── models/
│   │   └── database.py        # SQLAlchemy models (8 tables)
│   │
│   ├── services/
│   │   ├── rag_service.py     # RAG orchestration
│   │   └── rag/               # RAG components
│   │       ├── embeddings.py        # OpenAI embeddings
│   │       ├── vector_store.py      # Qdrant client
│   │       ├── text_splitter.py     # Smart chunking
│   │       ├── hybrid_search.py     # Semantic + keyword search
│   │       ├── bm25_index.py        # In-memory BM25 (fallback)
│   │       ├── postgres_bm25.py     # PostgreSQL FTS
│   │       ├── reranker.py          # Cross-encoder reranking
│   │       ├── deduplication.py     # MMR + token deduplication
│   │       ├── protocols.py         # Type protocols
│   │       └── loaders/             # Document loaders
│   │           ├── pdf.py
│   │           ├── docx.py
│   │           ├── text.py
│   │           └── web.py
│   │
│   ├── evaluation/            # RAG testing framework
│   ├── auth.py               # JWT utilities
│   ├── config.py             # Pydantic settings
│   ├── database.py           # SQLAlchemy session
│   ├── dependencies.py       # FastAPI dependencies
│   └── main.py               # Application entry point
│
├── alembic/                   # Database migrations
│   └── versions/
│       ├── 92a0af6a0f93_initial_migration.py
│       └── d3fb5a9677e5_add_bm25_table.py
│
├── tests/                     # Test suite
└── scripts/                   # Utility scripts
```

### Key Design Patterns

#### 1. Provider Pattern

All RAG components implement protocols for easy swapping:

```python
# Protocols defined in app/services/rag/protocols.py
class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

class VectorStore(Protocol):
    async def upsert(self, collection: str, documents: List[dict]): ...
    async def search(self, collection: str, vector: List[float], limit: int): ...
```

#### 2. Async/Await Throughout

All I/O operations are async for maximum performance:

```python
async def search(
    self,
    query: str,
    collection_name: str,
    limit: int = 10
) -> List[SearchResult]:
    # Async embedding generation
    # Async vector search
    # Async BM25 search
    # All run concurrently
```

#### 3. Dependency Injection

FastAPI dependencies for clean separation:

```python
@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    ...
```

---

## Frontend Architecture

### Directory Structure

```
frontend/
├── app/
│   ├── api/                   # API proxy routes (for auth)
│   ├── login/                 # Login page
│   ├── register/              # Register page
│   ├── page.tsx               # Main chat interface
│   ├── layout.tsx             # Root layout
│   └── globals.css            # Global styles
│
├── components/
│   ├── AuthGuard.tsx          # Protected route wrapper
│   ├── ConversationSidebar.tsx    # Left panel
│   ├── Scratchpad.tsx         # Right panel (4 tabs)
│   ├── KnowledgePoolList.tsx  # RAG pool management
│   ├── DocumentUpload.tsx     # Drag-and-drop upload
│   ├── DocumentList.tsx       # Document status display
│   ├── KnowledgePoolSelector.tsx  # Multi-pool selector
│   └── SearchInterface.tsx    # Direct RAG search
│
├── lib/
│   └── api.ts                 # API client utilities
│
└── store/
    ├── auth.ts                # Auth state (Zustand)
    ├── chat.ts                # Chat state
    ├── scratchpad.ts          # Scratchpad state
    └── rag.ts                 # RAG state
```

### State Management

Uses Zustand for lightweight, scalable state:

```typescript
// Example: Auth store
const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  login: async (credentials) => { ... },
  logout: () => { ... },
}))
```

### Three-Panel Layout

Using `react-resizable-panels`:

```
┌───────────────┬──────────────────────┬─────────────────┐
│               │                      │                 │
│ Conversations │    Chat + Stream     │   Scratchpad    │
│   Sidebar     │                      │   (4 tabs)      │
│               │                      │                 │
│  [+ New]      │  [Knowledge Pools ▼] │ • Todos         │
│  Conv 1       │                      │ • Notes         │
│  Conv 2       │  User: ...           │ • Journal       │
│  Conv 3       │  AI:  ...            │ • RAG           │
│               │                      │                 │
└───────────────┴──────────────────────┴─────────────────┘
```

---

## Data Flow

### 1. Document Upload Flow

```
User selects file
    ↓
Frontend: POST /api/rag/upload
    ↓
Backend: Save to disk + create DB record (status: PENDING)
    ↓
Background task: Process document
    ├─ Load file content
    ├─ Split into chunks
    ├─ Generate embeddings (OpenAI)
    ├─ Store in Qdrant (vectors)
    ├─ Store in PostgreSQL (BM25 FTS)
    └─ Update document status: COMPLETED
```

### 2. Chat Flow with RAG

```
User sends message
    ↓
Frontend: POST /api/chat/stream (SSE)
    ↓
Backend:
    ├─ Save user message to DB
    ├─ Check if RAG enabled
    │   ├─ Yes: Retrieve relevant chunks
    │   │   ├─ Hybrid search (semantic + BM25)
    │   │   ├─ Rerank results
    │   │   ├─ Deduplicate
    │   │   └─ Build context
    │   └─ No: Skip retrieval
    ├─ Check if scratchpad enabled
    │   ├─ Yes: Fetch todos/notes/journal
    │   └─ No: Skip
    ├─ Build prompt with context
    ├─ Stream LLM response (OpenAI)
    └─ Save assistant message to DB
    ↓
Frontend: Display streaming response
```

### 3. Hybrid Search Flow

```
Query: "How does Python handle memory?"
    ↓
┌──────────────────┬──────────────────┐
│  Semantic Search │   BM25 Search    │
│    (Qdrant)      │  (PostgreSQL)    │
└────────┬─────────┴─────────┬────────┘
         │                   │
    Vector results      Keyword results
         │                   │
         └────────┬──────────┘
                  ▼
        Reciprocal Rank Fusion (RRF)
                  ▼
          Combined ranking
                  ▼
        Cross-encoder reranking
                  ▼
           Deduplication
                  ▼
         Top K results
```

---

## Database Schema

### Tables

#### `users`

- **Purpose:** User accounts for authentication
- **Key Fields:** `id`, `email`, `username`, `hashed_password`
- **Relationships:** 1-to-many with conversations, scratchpad, knowledge_pools

#### `conversations`

- **Purpose:** Chat threads
- **Key Fields:** `id`, `user_id`, `title`, `use_rag`, `use_scratchpad`
- **Relationships:** Belongs to user, has many messages

#### `messages`

- **Purpose:** Individual chat messages
- **Key Fields:** `id`, `conversation_id`, `role` (user/assistant/system), `content`

#### `scratchpad_entries`

- **Purpose:** User's todos, notes, journal entries
- **Key Fields:** `id`, `user_id`, `entry_type`, `content`, `is_completed`

#### `knowledge_pools`

- **Purpose:** Document collections/categories
- **Key Fields:** `id`, `user_id`, `name`, `collection_name` (Qdrant reference)
- **Relationships:** Has many documents

#### `documents`

- **Purpose:** Uploaded files for RAG
- **Key Fields:** `id`, `knowledge_pool_id`, `filename`, `status`, `num_chunks`
- **Status Values:** `pending`, `processing`, `completed`, `failed`

#### `user_memories`

- **Purpose:** Extracted preferences/facts from conversations (schema ready, not yet implemented)
- **Key Fields:** `id`, `user_id`, `content`, `importance`

#### `bm25_documents`

- **Purpose:** Full-text search index for hybrid search
- **Key Fields:** `id`, `collection_name`, `document_id`, `content`, `content_tsv` (tsvector)
- **Indexes:** GIN index on `content_tsv` for fast FTS

### Migrations

Two migrations applied:

1. **92a0af6a0f93:** Initial schema (7 tables)
2. **d3fb5a9677e5:** Add BM25 full-text search table

---

## RAG Pipeline

### Components

#### Text Splitting

- **Implementation:** `SmartTextSplitter` in `backend/app/services/rag/text_splitter.py`
- **Strategy:** Hierarchical splitting respecting sentence boundaries
- **Chunk Size:** 1000 characters default
- **Overlap:** 200 characters for context preservation

#### Embeddings

- **Provider:** OpenAI `text-embedding-3-small`
- **Dimensions:** 1536
- **Batch Processing:** Yes (efficient bulk embedding)

#### Vector Storage

- **Technology:** Qdrant
- **Collections:** One per knowledge pool
- **Metadata:** Source filename, page number, chunk index, etc.

#### Hybrid Search

- **Semantic Search:** Qdrant vector similarity (cosine distance)
- **Keyword Search:** PostgreSQL Full-Text Search (BM25-style ranking)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Weights:** Configurable via `HYBRID_SEARCH_SEMANTIC_WEIGHT` and `HYBRID_SEARCH_KEYWORD_WEIGHT`

#### Reranking

- **Model:** Cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Purpose:** Re-score search results for better relevance
- **When Used:** After hybrid search fusion

#### Deduplication

- **Methods:**
  - **Token-based:** Remove chunks with high token overlap
  - **MMR:** Maximal Marginal Relevance for diversity
- **Purpose:** Avoid redundant context in LLM prompts

### Document Loaders

Supported formats:

- **PDF** (`PDFLoader`): PyPDF2-based extraction
- **DOCX** (`DocxLoader`): Microsoft Word documents
- **TXT/MD** (`TextLoader`): Plain text and Markdown
- **Web** (`WebLoader`): HTML content extraction

---

## Security

### Authentication

- **Method:** JWT (JSON Web Tokens)
- **Tokens:** Access token (short-lived) + Refresh token (long-lived)
- **Storage:** HTTP-only cookies (future) or localStorage (current)
- **Password Hashing:** bcrypt

### Authorization

- **User Isolation:** All data scoped to `user_id`
- **Row-Level Security:** Enforced in API endpoints via `get_current_user` dependency

### API Security

- **CORS:** Configured for localhost:3000 (development)
- **Input Validation:** Pydantic models for all requests
- **SQL Injection:** Protected via SQLAlchemy ORM
- **XSS:** Frontend sanitizes rendered content

### Planned Security Enhancements

- Rate limiting (not yet implemented)
- File upload size limits (not yet implemented)
- CSRF tokens for state-changing operations

---

## Deployment

### Development

**Prerequisites:**

- Docker & Docker Compose
- Python 3.13+ with `uv`
- Node.js 18+

**Start Infrastructure:**

```bash
docker-compose up -d  # PostgreSQL, Qdrant, Redis
```

**Backend:**

```bash
cd backend
uv pip install -e .
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000
```

### Production (Not Yet Configured)

Recommended setup:

- **Backend:** Docker container with Gunicorn + Uvicorn workers
- **Frontend:** Next.js static export or server deployment
- **Database:** Managed PostgreSQL (e.g., AWS RDS, DigitalOcean)
- **Vector Store:** Qdrant Cloud or self-hosted
- **Reverse Proxy:** Nginx with SSL termination
- **Monitoring:** Sentry + Prometheus (not yet implemented)

---

## Performance Characteristics

### Current Scale (Estimated)

- **Codebase:** ~4,600 lines Python + ~3,500 lines TypeScript
- **Database Tables:** 8 tables
- **API Endpoints:** 20+ endpoints
- **UI Components:** 15+ React components

### Benchmarks (Not Yet Measured)

- Document upload time: TBD
- Embedding generation: TBD
- Search latency (hybrid): TBD
- Chat response time (streaming): TBD

---

## References

- [Backend Services Documentation](./BACKEND_SERVICES.md) - Detailed RAG component docs
- [Project Status](../status/PROJECT_STATUS.md) - Current features and roadmap
- [Complete Development Guide](../development/COMPLETE_DEVELOPMENT_GUIDE.md) - Full development guide
- [CLAUDE.md](../../CLAUDE.md) - Project conventions for AI assistance

---

**Last Updated:** November 7, 2024
**Architecture Version:** 1.0 (MVP Complete)
