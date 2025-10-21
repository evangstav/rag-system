# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) chat system with scratchpad functionality. The system combines conversational AI with document retrieval and user note-taking capabilities.

**Current Status**: Early scaffolding stage - only `main.py` and `pyproject.toml` exist.

## Tech Stack

### Backend (Planned)

- **Framework**: FastAPI with async/await
- **Python**: 3.13+
- **Database**: PostgreSQL (users, conversations, messages, scratchpad)
- **Vector Store**: Qdrant (RAG document embeddings)
- **Cache/Session**: Redis
- **LLM Integration**: OpenAI/Anthropic via async clients

### Frontend (Planned)

- **Framework**: Next.js 14+ with App Router
- **UI**: Tailwind CSS + shadcn/ui
- **State**: Zustand for global state
- **Streaming**: Vercel AI SDK (`ai` package)
- **Layout**: `react-resizable-panels` for split-pane interface

## Development Commands

### Setup

```bash
# Install dependencies (uv recommended)
uv pip install -e .

# With standard pip
pip install -e .
```

### Run Application

```bash
# Development
python main.py

# Production (when FastAPI is added)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Services

```bash
# Start infrastructure (when docker-compose.yml exists)
docker-compose up -d

# Stop services
docker-compose down
```

### Database

```bash
# Run migrations (when Alembic is configured)
alembic upgrade head

# Create migration
alembic revision --autogenerate -m "description"
```

### Testing

```bash
# Run tests (when test suite exists)
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## Architecture Patterns

### Backend Structure (Target)

```
app/
├── main.py              # FastAPI app initialization
├── config.py            # Pydantic settings
├── api/                 # Route handlers
│   ├── chat.py         # Streaming chat endpoint (SSE)
│   ├── scratchpad.py   # CRUD for todos/notes/journal
│   ├── rag.py          # Document upload/knowledge pools
│   └── memory.py       # Conversation memory
├── services/           # Business logic
│   ├── llm.py         # LLM client wrapper (streaming)
│   ├── rag_service.py # RAG orchestration (provider pattern)
│   ├── memory_service.py
│   └── ingestion_service.py
└── models/
    ├── database.py    # SQLAlchemy models
    └── schemas.py     # Pydantic request/response models
```

### Key Architectural Decisions

**1. RAG Provider Pattern**

- Abstract interfaces for embedding, vector store, text splitting
- Swap implementations without changing core logic
- Support OpenAI embeddings + Qdrant, LlamaIndex, or custom providers

**2. Streaming Chat with SSE**

- Use FastAPI `StreamingResponse` with Server-Sent Events
- Stream LLM tokens as they arrive for real-time UX
- Format: `data: {json_payload}\n\n`
- Handle errors gracefully in stream

**3. Scratchpad Context Injection**

- Scratchpad data (todos/notes/journal) can be injected into LLM context
- User controls via toggle whether to include scratchpad
- Auto-save scratchpad every 30 seconds
- Daily journal entries archived at midnight

**4. Multi-Source RAG Ingestion**

- Support PDF, DOCX, web pages, CSV/Excel, APIs, databases
- Content-aware chunking (respect sentence/function/header boundaries)
- Store metadata (source, page number, timestamps)
- Background processing for large files

**5. Memory System**

- Extract user preferences/facts from conversations
- Store with importance scores in PostgreSQL
- Retrieve relevant memories via semantic search
- Include in system prompt for personalization

## Code Conventions

### FastAPI Endpoints

```python
# Use async/await consistently
@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    async def generate():
        async for chunk in llm_service.stream_chat(...):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
```

### RAG Service Pattern

```python
# All providers implement these protocols
class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

class VectorStore(Protocol):
    async def upsert(self, collection: str, documents: List[dict]) -> List[str]: ...
    async def search(self, collection: str, query_vector: List[float], limit: int) -> List[RAGDocument]: ...

# Swap implementations in dependencies.py
@lru_cache()
def get_rag_service() -> RAGService:
    return RAGService(
        embedding_provider=OpenAIEmbeddings(settings.OPENAI_API_KEY),
        vector_store=QdrantVectorStore(settings.QDRANT_URL),
        text_splitter=SmartTextSplitter()
    )
```

### Database Sessions

```python
# Use AsyncSession with proper cleanup
from app.dependencies import get_db

async def some_handler(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    # Session auto-closed by dependency
```

### Document Loaders

```python
# Each loader implements common interface
class DocumentLoader(ABC):
    @abstractmethod
    async def load(self, source: str) -> List[Document]: ...

    @abstractmethod
    def supports(self, source: str) -> bool: ...

    def clean_text(self, text: str) -> str:
        # Common text cleaning
        pass
```

## Environment Configuration

Required `.env` variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat

# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # optional

# Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...  # if using cloud

# App
SECRET_KEY=...  # generate with: openssl rand -hex 32
CORS_ORIGINS=["http://localhost:3000"]
REDIS_URL=redis://localhost:6379
```

## Development Workflow

### Adding New RAG Data Sources

1. Create loader in `app/services/rag/loaders/`
2. Implement `DocumentLoader` interface
3. Register in `IngestionService.loaders` list
4. Add API endpoint in `app/api/ingestion.py`

### Adding New Chat Features

1. Extend `ChatRequest` schema if needed
2. Modify context building in `api/chat.py`
3. Update LLM service streaming logic
4. Update frontend chat component

### Adding Database Tables

1. Add SQLAlchemy model in `models/database.py`
2. Create Alembic migration: `alembic revision --autogenerate`
3. Review migration SQL
4. Apply: `alembic upgrade head`

## Performance Considerations

- **Batch Embeddings**: Always use `embed_batch()` for multiple texts
- **Connection Pooling**: Configure for PostgreSQL and Redis
- **Background Tasks**: Use FastAPI `BackgroundTasks` for file uploads/indexing
- **Streaming**: Stream LLM responses token-by-token for perceived speed
- **Caching**: Cache embeddings for frequently accessed docs

## Common Pitfalls

- **Don't split tables**: Keep markdown tables intact when chunking
- **Preserve context**: Use overlapping chunks (200 chars default)
- **Metadata is critical**: Always include source info for citation
- **Handle streaming errors**: Wrap stream generation in try/except
- **Auto-save race conditions**: Debounce scratchpad saves
- **Token limits**: Monitor context window usage with long conversations

## Testing Strategy

- Unit tests for RAG providers (mock embeddings/vector store)
- Integration tests for chat streaming (use test LLM)
- API tests with FastAPI TestClient
- Frontend tests with React Testing Library
- E2E tests for critical flows (chat, scratchpad, RAG toggle)

## Reference Documentation

See comprehensive implementation details in:

- `COMPLETE_DEVELOPMENT_GUIDE.md` - Full production architecture
- `QUICK_START_GUIDE.md` - 2-hour MVP setup
- `RAG_SYSTEM_SNIPPET_COLLECTION.md` - Code snippets (if exists)
