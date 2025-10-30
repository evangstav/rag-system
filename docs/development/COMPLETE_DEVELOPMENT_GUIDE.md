# Complete Development Guide: RAG Chat System with Scratchpad

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Chat         │  │ Scratchpad   │  │ Artifacts    │     │
│  │ Interface    │  │ Editor       │  │ Panel        │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             │ HTTP/WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /api/chat (SSE streaming)                           │  │
│  │  /api/scratchpad                                     │  │
│  │  /api/memory                                         │  │
│  │  /api/rag/toggle                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌────────────┬────────────┴────────────┬─────────────┐   │
│  │            │                         │             │   │
│  ▼            ▼                         ▼             ▼   │
│ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐       │
│ │ LLM    │ │ Vector │ │ Memory   │ │ PostgreSQL │       │
│ │ Client │ │ Store  │ │ Store    │ │            │       │
│ └────────┘ └────────┘ └──────────┘ └────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack Recommendations

### Backend

- **FastAPI** (Python 3.11+): Async support, automatic OpenAPI docs
- **PostgreSQL**: Primary database for users, conversations, scratchpad data
- **Redis**: Session management, rate limiting, caching
- **Qdrant/Weaviate**: Vector database for RAG
- **LangChain**: RAG orchestration and memory management

### Frontend

**Recommended: Next.js 14+ with App Router**

Why Next.js over alternatives:

- Built-in API routes protect API keys server-side
- Native streaming support with React Server Components
- Excellent TypeScript support and developer experience
- Vercel AI SDK provides zero-boilerplate streaming

**Core Libraries:**

- `ai` (Vercel AI SDK): Streaming and state management
- `react-resizable-panels`: Split-pane layout
- `react-markdown` + `remark-gfm`: Markdown rendering
- `@monaco-editor/react`: Scratchpad code editing
- `zustand`: Global state management
- `tailwindcss` + `shadcn/ui`: UI components

### Infrastructure

- Docker for containerization
- PostgreSQL + Redis via Docker Compose
- Vector DB (Qdrant recommended for ease of setup)

## Database Schema Design

### PostgreSQL Tables

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    archived BOOLEAN DEFAULT FALSE
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB -- For storing sources, artifacts, etc.
);

-- Scratchpad entries
CREATE TABLE scratchpad_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    entry_type VARCHAR(50) NOT NULL, -- 'todo', 'note', 'journal'
    content TEXT NOT NULL,
    is_archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    archived_at TIMESTAMP
);

-- Memory store (for conversation memory)
CREATE TABLE memory_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id),
    memory_type VARCHAR(50) NOT NULL, -- 'fact', 'preference', 'context'
    content TEXT NOT NULL,
    importance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- RAG knowledge pools
CREATE TABLE knowledge_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Documents in knowledge pools
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_id UUID REFERENCES knowledge_pools(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    content TEXT,
    metadata JSONB,
    vector_ids TEXT[], -- IDs in vector database
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX idx_scratchpad_user ON scratchpad_entries(user_id, created_at);
CREATE INDEX idx_memory_user ON memory_entries(user_id, importance_score DESC);
CREATE INDEX idx_documents_pool ON documents(pool_id);
```

## Backend Implementation

### Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization
│   ├── config.py               # Configuration management
│   ├── dependencies.py         # Dependency injection
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py            # Chat endpoints
│   │   ├── scratchpad.py      # Scratchpad CRUD
│   │   ├── memory.py          # Memory management
│   │   ├── rag.py             # RAG endpoints
│   │   └── auth.py            # Authentication
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py        # SQLAlchemy models
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py             # LLM client wrapper
│   │   ├── rag_service.py     # RAG orchestration
│   │   ├── memory_service.py  # Memory operations
│   │   └── scratchpad_service.py
│   └── utils/
│       ├── __init__.py
│       ├── streaming.py       # SSE streaming helpers
│       └── vector_store.py    # Vector DB client
├── alembic/                   # Database migrations
├── tests/
├── requirements.txt
└── Dockerfile
```

### Core Backend Implementation

#### 1. Configuration (`app/config.py`)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    
    # LLM
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str | None = None
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    
    # Vector Store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    
    # App
    SECRET_KEY: str
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

#### 2. Main Application (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as redis

from app.config import get_settings
from app.api import chat, scratchpad, memory, rag, auth

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = redis.from_url(settings.REDIS_URL)
    yield
    # Shutdown
    await app.state.redis.close()

app = FastAPI(
    title="RAG Chat System",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(scratchpad.router, prefix="/api/scratchpad", tags=["scratchpad"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

#### 3. Streaming Chat Endpoint (`app/api/chat.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator

from app.dependencies import get_db, get_current_user
from app.services.llm import LLMService
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService
from app.models.database import User, Conversation, Message
from app.models.schemas import ChatRequest, ChatMessage

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    use_rag: bool = False
    knowledge_pool_ids: list[str] = []
    scratchpad_context: str | None = None

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    llm_service: LLMService = Depends(),
    rag_service: RAGService = Depends(),
    memory_service: MemoryService = Depends(),
):
    """Stream chat response with Server-Sent Events"""
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Get or create conversation
            conversation_id = request.conversation_id
            if not conversation_id:
                conversation = Conversation(user_id=user.id)
                db.add(conversation)
                await db.commit()
                conversation_id = str(conversation.id)
            
            # Get conversation history
            history = await get_conversation_history(db, conversation_id)
            
            # Build context
            context_parts = []
            
            # Add memory context
            if await memory_service.is_enabled(user.id):
                memories = await memory_service.get_relevant_memories(
                    user.id, 
                    request.message,
                    limit=5
                )
                if memories:
                    context_parts.append(
                        "Relevant information about the user:\n" +
                        "\n".join(f"- {m.content}" for m in memories)
                    )
            
            # Add scratchpad context
            if request.scratchpad_context:
                context_parts.append(
                    f"User's current scratchpad:\n{request.scratchpad_context}"
                )
            
            # Add RAG context
            if request.use_rag and request.knowledge_pool_ids:
                rag_results = await rag_service.retrieve(
                    query=request.message,
                    pool_ids=request.knowledge_pool_ids,
                    top_k=5
                )
                if rag_results:
                    sources = "\n\n".join([
                        f"Source {i+1} ({r.metadata.get('filename', 'Unknown')}):\n{r.content}"
                        for i, r in enumerate(rag_results)
                    ])
                    context_parts.append(f"Retrieved information:\n{sources}")
            
            # Build system message
            system_message = "You are a helpful AI assistant."
            if context_parts:
                system_message += "\n\n" + "\n\n".join(context_parts)
            
            # Save user message
            user_message = Message(
                conversation_id=conversation_id,
                role="user",
                content=request.message
            )
            db.add(user_message)
            await db.commit()
            
            # Stream response
            full_response = ""
            sources = []
            
            async for chunk in llm_service.stream_chat(
                messages=[
                    {"role": "system", "content": system_message},
                    *history,
                    {"role": "user", "content": request.message}
                ]
            ):
                if chunk.get("type") == "content":
                    content = chunk.get("content", "")
                    full_response += content
                    
                    # Send SSE formatted response
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                
                elif chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Save assistant message
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                metadata={"sources": sources} if sources else None
            )
            db.add(assistant_message)
            await db.commit()
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

async def get_conversation_history(db: AsyncSession, conversation_id: str) -> list[dict]:
    """Get conversation history for context"""
    result = await db.execute(
        "SELECT role, content FROM messages "
        "WHERE conversation_id = :conv_id "
        "ORDER BY created_at ASC "
        "LIMIT 20",  # Keep last 20 messages
        {"conv_id": conversation_id}
    )
    messages = result.fetchall()
    return [{"role": m.role, "content": m.content} for m in messages]
```

#### 4. LLM Service (`app/services/llm.py`)

```python
from typing import AsyncGenerator
import openai
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.DEFAULT_MODEL
    
    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[dict, None]:
        """Stream chat completion with proper token handling"""
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "content",
                        "content": chunk.choices[0].delta.content
                    }
                    
        except Exception as e:
            yield {"type": "error", "error": str(e)}
```

#### 5. Scratchpad Endpoint (`app/api/scratchpad.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from datetime import datetime, date

from app.dependencies import get_db, get_current_user
from app.models.database import User, ScratchpadEntry

router = APIRouter()

class ScratchpadUpdate(BaseModel):
    content: str
    entry_type: str  # 'todo', 'note', 'journal'

class ScratchpadResponse(BaseModel):
    id: str
    entry_type: str
    content: str
    created_at: datetime
    updated_at: datetime

@router.get("/")
async def get_scratchpad(
    entry_type: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get current scratchpad entries"""
    query = select(ScratchpadEntry).where(
        ScratchpadEntry.user_id == user.id,
        ScratchpadEntry.is_archived == False
    )
    
    if entry_type:
        query = query.where(ScratchpadEntry.entry_type == entry_type)
    
    result = await db.execute(query.order_by(ScratchpadEntry.updated_at.desc()))
    entries = result.scalars().all()
    
    return [
        ScratchpadResponse(
            id=str(e.id),
            entry_type=e.entry_type,
            content=e.content,
            created_at=e.created_at,
            updated_at=e.updated_at
        )
        for e in entries
    ]

@router.post("/")
async def create_or_update_scratchpad(
    data: ScratchpadUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Create or update scratchpad entry"""
    
    # For journal entries, find today's entry
    if data.entry_type == "journal":
        today = date.today()
        result = await db.execute(
            select(ScratchpadEntry).where(
                ScratchpadEntry.user_id == user.id,
                ScratchpadEntry.entry_type == "journal",
                ScratchpadEntry.is_archived == False,
                ScratchpadEntry.created_at >= today
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.content = data.content
            existing.updated_at = datetime.utcnow()
            await db.commit()
            return {"id": str(existing.id), "message": "Updated"}
    
    # Create new entry
    entry = ScratchpadEntry(
        user_id=user.id,
        entry_type=data.entry_type,
        content=data.content
    )
    db.add(entry)
    await db.commit()
    
    return {"id": str(entry.id), "message": "Created"}

@router.post("/archive-daily")
async def archive_daily_entries(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Archive journal entries at end of day"""
    
    await db.execute(
        update(ScratchpadEntry)
        .where(
            ScratchpadEntry.user_id == user.id,
            ScratchpadEntry.entry_type == "journal",
            ScratchpadEntry.is_archived == False
        )
        .values(is_archived=True, archived_at=datetime.utcnow())
    )
    await db.commit()
    
    return {"message": "Archived daily entries"}
```

#### 6. RAG Service (Framework Agnostic)

The RAG service uses a **provider pattern** so you can easily swap implementations:

**Base Interface (`app/services/rag/base.py`)**

```python
from abc import ABC, abstractmethod
from typing import List, Protocol
from dataclasses import dataclass

@dataclass
class RAGDocument:
    content: str
    metadata: dict
    score: float = 0.0

class EmbeddingProvider(Protocol):
    """Interface for embedding providers"""
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        ...

class TextSplitter(Protocol):
    """Interface for text splitting"""
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        ...

class VectorStore(Protocol):
    """Interface for vector storage"""
    async def upsert(self, collection: str, documents: List[dict]) -> List[str]:
        """Store documents and return IDs"""
        ...
    
    async def search(self, collection: str, query_vector: List[float], limit: int) -> List[RAGDocument]:
        """Search for similar documents"""
        ...

class RAGService:
    """Framework-agnostic RAG orchestrator"""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        text_splitter: TextSplitter,
    ):
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.text_splitter = text_splitter
    
    async def index_document(
        self,
        pool_id: str,
        content: str,
        metadata: dict
    ) -> List[str]:
        """Index document - works with any provider"""
        collection_name = f"pool_{pool_id}"
        
        # Split text
        chunks = self.text_splitter.split_text(content)
        
        # Generate embeddings
        embeddings = await self.embeddings.embed_batch(chunks)
        
        # Prepare documents
        documents = [
            {
                "content": chunk,
                "embedding": embedding,
                "metadata": {**metadata, "chunk_index": i}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Store in vector DB
        return await self.vector_store.upsert(collection_name, documents)
    
    async def retrieve(
        self,
        query: str,
        pool_ids: List[str],
        top_k: int = 5
    ) -> List[RAGDocument]:
        """Retrieve relevant documents"""
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)
        
        # Search across all pools
        all_results = []
        for pool_id in pool_ids:
            collection_name = f"pool_{pool_id}"
            results = await self.vector_store.search(
                collection_name,
                query_embedding,
                top_k
            )
            all_results.extend(results)
        
        # Sort by score and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
```

**Implementation Examples**

**Option 1: Direct OpenAI + Qdrant (No frameworks)**

```python
# app/services/rag/providers/openai_qdrant.py
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class OpenAIEmbeddings:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def embed_text(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

class QdrantVectorStore:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
    
    async def upsert(self, collection: str, documents: List[dict]) -> List[str]:
        # Ensure collection exists
        try:
            await self.client.get_collection(collection)
        except:
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        # Create points
        point_ids = []
        points = []
        
        for doc in documents:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            points.append(PointStruct(
                id=point_id,
                vector=doc["embedding"],
                payload={
                    "content": doc["content"],
                    **doc["metadata"]
                }
            ))
        
        await self.client.upsert(collection_name=collection, points=points)
        return point_ids
    
    async def search(self, collection: str, query_vector: List[float], limit: int) -> List[RAGDocument]:
        try:
            results = await self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                RAGDocument(
                    content=result.payload["content"],
                    metadata=result.payload,
                    score=result.score
                )
                for result in results
            ]
        except:
            return []

class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """Simple sliding window splitter"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        
        return chunks

# Create service
def create_rag_service():
    from app.config import get_settings
    settings = get_settings()
    
    return RAGService(
        embedding_provider=OpenAIEmbeddings(settings.OPENAI_API_KEY),
        vector_store=QdrantVectorStore(settings.QDRANT_URL),
        text_splitter=SimpleTextSplitter()
    )
```

**Option 2: LlamaIndex (if you want to use it)**

```python
# app/services/rag/providers/llamaindex.py
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client

class LlamaIndexRAGService:
    """Alternative implementation using LlamaIndex"""
    
    def __init__(self, qdrant_url: str, openai_key: str):
        self.client = qdrant_client.QdrantClient(url=qdrant_url)
        self.embed_model = OpenAIEmbedding(api_key=openai_key)
    
    async def index_document(self, pool_id: str, content: str, metadata: dict) -> List[str]:
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=f"pool_{pool_id}"
        )
        
        doc = Document(text=content, metadata=metadata)
        index = VectorStoreIndex.from_documents(
            [doc],
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        return [doc.doc_id]
    
    async def retrieve(self, query: str, pool_ids: List[str], top_k: int = 5) -> List[RAGDocument]:
        all_results = []
        
        for pool_id in pool_ids:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=f"pool_{pool_id}"
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = await retriever.aretrieve(query)
            
            for node in nodes:
                all_results.append(RAGDocument(
                    content=node.text,
                    metadata=node.metadata,
                    score=node.score
                ))
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
```

**Option 3: Weaviate**

```python
# app/services/rag/providers/weaviate.py
import weaviate
from weaviate.classes.query import MetadataQuery

class WeaviateVectorStore:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=weaviate.AuthApiKey(api_key) if api_key else None
        )
    
    async def upsert(self, collection: str, documents: List[dict]) -> List[str]:
        # Create class if not exists
        class_name = collection.replace("-", "_").title()
        
        if not self.client.schema.exists(class_name):
            self.client.schema.create_class({
                "class": class_name,
                "vectorizer": "none",  # We provide vectors
            })
        
        # Batch insert
        with self.client.batch as batch:
            for doc in documents:
                batch.add_data_object(
                    data_object={
                        "content": doc["content"],
                        **doc["metadata"]
                    },
                    class_name=class_name,
                    vector=doc["embedding"]
                )
        
        return [str(i) for i in range(len(documents))]
    
    async def search(self, collection: str, query_vector: List[float], limit: int) -> List[RAGDocument]:
        class_name = collection.replace("-", "_").title()
        
        result = (
            self.client.query
            .get(class_name, ["content"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .with_additional(["distance"])
            .do()
        )
        
        documents = []
        for item in result.get("data", {}).get("Get", {}).get(class_name, []):
            documents.append(RAGDocument(
                content=item["content"],
                metadata=item,
                score=1 - item["_additional"]["distance"]  # Convert distance to similarity
            ))
        
        return documents
```

**Configuration (`app/dependencies.py`)**

```python
from functools import lru_cache
from app.services.rag.base import RAGService
from app.config import get_settings

@lru_cache()
def get_rag_service() -> RAGService:
    """Factory function - swap implementations here"""
    settings = get_settings()
    
    # Option 1: Direct OpenAI + Qdrant (recommended)
    from app.services.rag.providers.openai_qdrant import create_rag_service
    return create_rag_service()
    
    # Option 2: LlamaIndex
    # from app.services.rag.providers.llamaindex import LlamaIndexRAGService
    # return LlamaIndexRAGService(settings.QDRANT_URL, settings.OPENAI_API_KEY)
    
    # Option 3: Custom implementation
    # return YourCustomRAGService()
```

**Adding Your Own RAG Implementation**

Just implement the three protocols:

```python
# app/services/rag/providers/custom.py

class MyCustomEmbeddings:
    async def embed_text(self, text: str) -> List[float]:
        # Your embedding logic (Cohere, HuggingFace, local model, etc.)
        pass
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding
        pass

class MyCustomVectorStore:
    async def upsert(self, collection: str, documents: List[dict]) -> List[str]:
        # Store in Pinecone, Milvus, ChromaDB, etc.
        pass
    
    async def search(self, collection: str, query_vector: List[float], limit: int) -> List[RAGDocument]:
        # Search implementation
        pass

class MyCustomSplitter:
    def split_text(self, text: str) -> List[str]:
        # Semantic chunking, sentence splitting, etc.
        pass

# Wire it up
def create_custom_rag():
    return RAGService(
        embedding_provider=MyCustomEmbeddings(),
        vector_store=MyCustomVectorStore(),
        text_splitter=MyCustomSplitter()
    )
```

## Frontend Implementation

### Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Home page
│   ├── api/
│   │   └── chat/
│   │       └── route.ts        # Chat API proxy
│   └── chat/
│       └── page.tsx            # Chat page
├── components/
│   ├── chat/
│   │   ├── ChatInterface.tsx
│   │   ├── ChatMessages.tsx
│   │   ├── ChatInput.tsx
│   │   └── MessageBubble.tsx
│   ├── scratchpad/
│   │   ├── Scratchpad.tsx
│   │   ├── TodoList.tsx
│   │   ├── Notes.tsx
│   │   └── Journal.tsx
│   ├── artifacts/
│   │   ├── ArtifactPanel.tsx
│   │   └── CodeDisplay.tsx
│   └── layout/
│       └── SplitLayout.tsx
├── lib/
│   ├── api.ts                  # API client
│   └── hooks/
│       ├── useChat.ts
│       ├── useScratchpad.ts
│       └── useRAG.ts
├── store/
│   └── store.ts                # Zustand store
├── package.json
└── next.config.js
```

### Core Frontend Components

#### 1. Chat Interface (`components/chat/ChatInterface.tsx`)

```typescript
'use client';

import { useChat } from 'ai/react';
import { useState } from 'react';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { useScratchpadStore } from '@/store/store';

export function ChatInterface() {
  const [useRAG, setUseRAG] = useState(false);
  const [useScratchpad, setUseScratchpad] = useState(true);
  const [selectedPools, setSelectedPools] = useState<string[]>([]);
  const getScratchpadContent = useScratchpadStore((state) => state.getFullContent);
  
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    body: {
      use_rag: useRAG,
      knowledge_pool_ids: selectedPools,
      scratchpad_context: useScratchpad ? getScratchpadContent() : null,
    },
  });
  
  return (
    <div className="flex flex-col h-full">
      {/* Context Options */}
      <div className="p-4 border-b bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Context Options</h3>
        
        <div className="space-y-2">
          {/* RAG Toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useRAG}
              onChange={(e) => setUseRAG(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Use RAG Knowledge</span>
          </label>
          
          {useRAG && (
            <div className="ml-6">
              <KnowledgePoolSelector
                selected={selectedPools}
                onChange={setSelectedPools}
              />
            </div>
          )}
          
          {/* Scratchpad Toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useScratchpad}
              onChange={(e) => setUseScratchpad(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Include Scratchpad Context</span>
          </label>
        </div>
        
        {/* Active Context Indicator */}
        {(useRAG || useScratchpad) && (
          <div className="mt-3 text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded inline-block">
            Active context: {[
              useRAG && `RAG (${selectedPools.length} pools)`,
              useScratchpad && 'Scratchpad'
            ].filter(Boolean).join(' + ')}
          </div>
        )}
      </div>
      
      {/* Messages */}
      <ChatMessages messages={messages} isLoading={isLoading} />
      
      {/* Input */}
      <ChatInput
        input={input}
        handleInputChange={handleInputChange}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
      />
    </div>
  );
}
```

#### 2. Scratchpad Component (`components/scratchpad/Scratchpad.tsx`)

```typescript
'use client';

import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TodoList } from './TodoList';
import { Notes } from './Notes';
import { Journal } from './Journal';
import { useScratchpadStore } from '@/store/store';

export function Scratchpad() {
  const { todos, notes, journal, loadScratchpad, saveScratchpad } = useScratchpadStore();
  const [activeTab, setActiveTab] = useState('todos');
  
  useEffect(() => {
    loadScratchpad();
    
    // Auto-save interval
    const interval = setInterval(() => {
      saveScratchpad();
    }, 30000); // Save every 30 seconds
    
    return () => clearInterval(interval);
  }, [loadScratchpad, saveScratchpad]);
  
  // Archive journal at midnight
  useEffect(() => {
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    
    const timeUntilMidnight = tomorrow.getTime() - now.getTime();
    
    const timeout = setTimeout(async () => {
      await fetch('/api/scratchpad/archive-daily', { method: 'POST' });
      loadScratchpad();
    }, timeUntilMidnight);
    
    return () => clearTimeout(timeout);
  }, [loadScratchpad]);
  
  return (
    <div className="h-full flex flex-col p-4">
      <h2 className="text-lg font-semibold mb-4">Scratchpad</h2>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="todos">Todos</TabsTrigger>
          <TabsTrigger value="notes">Notes</TabsTrigger>
          <TabsTrigger value="journal">Journal</TabsTrigger>
        </TabsList>
        
        <TabsContent value="todos" className="flex-1">
          <TodoList />
        </TabsContent>
        
        <TabsContent value="notes" className="flex-1">
          <Notes />
        </TabsContent>
        
        <TabsContent value="journal" className="flex-1">
          <Journal />
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

#### 3. Split Layout (`components/layout/SplitLayout.tsx`)

```typescript
'use client';

import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { Scratchpad } from '@/components/scratchpad/Scratchpad';
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel';
import { useState } from 'react';

export function SplitLayout() {
  const [showArtifacts, setShowArtifacts] = useState(false);
  
  return (
    <PanelGroup direction="horizontal" autoSaveId="main-layout">
      {/* Left: Scratchpad */}
      <Panel defaultSize={20} minSize={15} maxSize={30} collapsible>
        <Scratchpad />
      </Panel>
      
      <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-blue-500 transition-colors" />
      
      {/* Center: Chat */}
      <Panel defaultSize={showArtifacts ? 40 : 60} minSize={30}>
        <ChatInterface />
      </Panel>
      
      {/* Right: Artifacts (conditional) */}
      {showArtifacts && (
        <>
          <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-blue-500 transition-colors" />
          <Panel defaultSize={40} minSize={30} collapsible>
            <ArtifactPanel onClose={() => setShowArtifacts(false)} />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
}
```

#### 4. State Management (`store/store.ts`)

```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

interface ScratchpadStore {
  todos: Todo[];
  notes: string;
  journal: string;
  
  // Actions
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  removeTodo: (id: string) => void;
  setNotes: (notes: string) => void;
  setJournal: (journal: string) => void;
  
  // Persistence
  loadScratchpad: () => Promise<void>;
  saveScratchpad: () => Promise<void>;
  getFullContent: () => string;
}

export const useScratchpadStore = create<ScratchpadStore>()(
  persist(
    (set, get) => ({
      todos: [],
      notes: '',
      journal: '',
      
      addTodo: (text) =>
        set((state) => ({
          todos: [...state.todos, { id: crypto.randomUUID(), text, completed: false }],
        })),
      
      toggleTodo: (id) =>
        set((state) => ({
          todos: state.todos.map((todo) =>
            todo.id === id ? { ...todo, completed: !todo.completed } : todo
          ),
        })),
      
      removeTodo: (id) =>
        set((state) => ({
          todos: state.todos.filter((todo) => todo.id !== id),
        })),
      
      setNotes: (notes) => set({ notes }),
      
      setJournal: (journal) => set({ journal }),
      
      loadScratchpad: async () => {
        const response = await fetch('/api/scratchpad');
        const data = await response.json();
        
        // Parse and set data
        const todos = data.filter((e: any) => e.entry_type === 'todo');
        const noteEntry = data.find((e: any) => e.entry_type === 'note');
        const journalEntry = data.find((e: any) => e.entry_type === 'journal');
        
        set({
          todos: todos.map((t: any) => JSON.parse(t.content)),
          notes: noteEntry?.content || '',
          journal: journalEntry?.content || '',
        });
      },
      
      saveScratchpad: async () => {
        const state = get();
        
        // Save todos
        await fetch('/api/scratchpad', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            entry_type: 'todo',
            content: JSON.stringify(state.todos),
          }),
        });
        
        // Save notes
        if (state.notes) {
          await fetch('/api/scratchpad', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              entry_type: 'note',
              content: state.notes,
            }),
          });
        }
        
        // Save journal
        if (state.journal) {
          await fetch('/api/scratchpad', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              entry_type: 'journal',
              content: state.journal,
            }),
          });
        }
      },
      
      getFullContent: () => {
        const state = get();
        const parts = [];
        
        if (state.todos.length > 0) {
          parts.push(
            'Todos:\n' +
            state.todos.map(t => `- [${t.completed ? 'x' : ' '}] ${t.text}`).join('\n')
          );
        }
        
        if (state.notes) {
          parts.push(`Notes:\n${state.notes}`);
        }
        
        if (state.journal) {
          parts.push(`Journal:\n${state.journal}`);
        }
        
        return parts.join('\n\n');
      },
    }),
    {
      name: 'scratchpad-storage',
    }
  )
);
```

## Development Workflow

### 1. Initial Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://user:pass@localhost:5432/ragchat
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
SECRET_KEY=$(openssl rand -hex 32)
EOF

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### 2. Docker Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ragchat
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:pass@postgres:5432/ragchat
      REDIS_URL: redis://redis:6379
      QDRANT_URL: http://qdrant:6333
    depends_on:
      - postgres
      - redis
      - qdrant
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
  qdrant_data:
```

## Data Source Onboarding for RAG

### Data Ingestion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │ PDF  │ │DOCX  │ │ Web  │ │ API  │ │ DB   │ │Notion│   │
│  └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘   │
└──────┼────────┼────────┼────────┼────────┼────────┼────────┘
       │        │        │        │        │        │
       └────────┴────────┴────────┴────────┴────────┘
                         │
              ┌──────────▼──────────┐
              │  Document Processor  │
              │  - Extract text      │
              │  - Clean/normalize   │
              │  - Extract metadata  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Text Splitter      │
              │  - Chunk by type     │
              │  - Preserve context  │
              │  - Add boundaries    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Embedding Pipeline  │
              │  - Generate vectors  │
              │  - Batch processing  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    Vector Store      │
              │  - Index documents   │
              │  - Store metadata    │
              └─────────────────────┘
```

### Document Processor Implementation

**Base Document Loader (`app/services/rag/loaders/base.py`)**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class Document:
    """Unified document representation"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source_type: str  # 'pdf', 'web', 'docx', etc.
    created_at: datetime
    
class DocumentLoader(ABC):
    """Base class for all document loaders"""
    
    @abstractmethod
    async def load(self, source: str) -> List[Document]:
        """Load documents from source"""
        pass
    
    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if loader supports this source"""
        pass
    
    def clean_text(self, text: str) -> str:
        """Common text cleaning operations"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"]+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
```

**PDF Loader (`app/services/rag/loaders/pdf_loader.py`)**

```python
import PyPDF2
import pdfplumber
from pathlib import Path
import uuid

class PDFLoader(DocumentLoader):
    """Load and process PDF files"""
    
    def supports(self, source: str) -> bool:
        return source.lower().endswith('.pdf')
    
    async def load(self, source: str) -> List[Document]:
        """Load PDF with multiple strategies"""
        
        # Try pdfplumber first (better for tables/structured content)
        try:
            return await self._load_with_pdfplumber(source)
        except Exception as e:
            print(f"pdfplumber failed: {e}, falling back to PyPDF2")
            return await self._load_with_pypdf2(source)
    
    async def _load_with_pdfplumber(self, source: str) -> List[Document]:
        documents = []
        
        with pdfplumber.open(source) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                
                # Extract tables
                tables = page.extract_tables()
                
                # Format tables as markdown
                table_text = ""
                for table in tables:
                    table_text += self._format_table_as_markdown(table)
                
                full_text = text + "\n\n" + table_text if table_text else text
                
                documents.append(Document(
                    content=self.clean_text(full_text),
                    metadata={
                        "page": i + 1,
                        "total_pages": len(pdf.pages),
                        "filename": Path(source).name,
                        "has_tables": len(tables) > 0
                    },
                    doc_id=str(uuid.uuid4()),
                    source_type="pdf",
                    created_at=datetime.utcnow()
                ))
        
        return documents
    
    async def _load_with_pypdf2(self, source: str) -> List[Document]:
        documents = []
        
        with open(source, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                documents.append(Document(
                    content=self.clean_text(text),
                    metadata={
                        "page": i + 1,
                        "total_pages": len(pdf.pages),
                        "filename": Path(source).name
                    },
                    doc_id=str(uuid.uuid4()),
                    source_type="pdf",
                    created_at=datetime.utcnow()
                ))
        
        return documents
    
    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Convert table to markdown format"""
        if not table or not table[0]:
            return ""
        
        lines = []
        
        # Header
        header = "| " + " | ".join(str(cell or "") for cell in table[0]) + " |"
        lines.append(header)
        
        # Separator
        separator = "| " + " | ".join("---" for _ in table[0]) + " |"
        lines.append(separator)
        
        # Rows
        for row in table[1:]:
            row_text = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            lines.append(row_text)
        
        return "\n".join(lines) + "\n"
```

**DOCX Loader (`app/services/rag/loaders/docx_loader.py`)**

```python
from docx import Document as DocxDocument
import uuid

class DOCXLoader(DocumentLoader):
    """Load Microsoft Word documents"""
    
    def supports(self, source: str) -> bool:
        return source.lower().endswith(('.docx', '.doc'))
    
    async def load(self, source: str) -> List[Document]:
        doc = DocxDocument(source)
        
        # Extract text with structure preservation
        sections = []
        current_section = {"heading": None, "content": []}
        
        for para in doc.paragraphs:
            # Check if paragraph is a heading
            if para.style.name.startswith('Heading'):
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "heading": para.text,
                    "level": para.style.name,
                    "content": []
                }
            else:
                current_section["content"].append(para.text)
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
        
        # Extract tables
        tables_text = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                table_data.append([cell.text for cell in row.cells])
            tables_text.append(self._format_table_as_markdown(table_data))
        
        # Create documents
        documents = []
        
        for i, section in enumerate(sections):
            content = f"# {section['heading']}\n\n" if section['heading'] else ""
            content += "\n".join(section['content'])
            
            documents.append(Document(
                content=self.clean_text(content),
                metadata={
                    "section": i + 1,
                    "heading": section.get('heading'),
                    "level": section.get('level'),
                    "filename": Path(source).name
                },
                doc_id=str(uuid.uuid4()),
                source_type="docx",
                created_at=datetime.utcnow()
            ))
        
        # Add tables as separate documents if present
        if tables_text:
            documents.append(Document(
                content="\n\n".join(tables_text),
                metadata={
                    "section": "tables",
                    "filename": Path(source).name
                },
                doc_id=str(uuid.uuid4()),
                source_type="docx",
                created_at=datetime.utcnow()
            ))
        
        return documents
```

**Web Scraper (`app/services/rag/loaders/web_loader.py`)**

```python
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import uuid

class WebLoader(DocumentLoader):
    """Load content from web pages"""
    
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.visited = set()
    
    def supports(self, source: str) -> bool:
        return source.startswith(('http://', 'https://'))
    
    async def load(self, source: str) -> List[Document]:
        """Load single page or crawl site"""
        return await self._crawl(source, depth=0)
    
    async def _crawl(self, url: str, depth: int) -> List[Document]:
        if depth > self.max_depth or url in self.visited:
            return []
        
        self.visited.add(url)
        documents = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                    
                    # Extract main content
                    main_content = soup.find('main') or soup.find('article') or soup.body
                    
                    if main_content:
                        # Extract text
                        text = main_content.get_text(separator='\n', strip=True)
                        
                        # Extract title
                        title = soup.find('h1')
                        title_text = title.get_text() if title else soup.title.string if soup.title else url
                        
                        # Extract metadata
                        meta_description = soup.find('meta', attrs={'name': 'description'})
                        description = meta_description['content'] if meta_description else ""
                        
                        documents.append(Document(
                            content=self.clean_text(text),
                            metadata={
                                "url": url,
                                "title": title_text,
                                "description": description,
                                "depth": depth
                            },
                            doc_id=str(uuid.uuid4()),
                            source_type="web",
                            created_at=datetime.utcnow()
                        ))
                        
                        # Find links for crawling (if depth allows)
                        if depth < self.max_depth:
                            links = soup.find_all('a', href=True)
                            base_domain = urlparse(url).netloc
                            
                            for link in links:
                                href = urljoin(url, link['href'])
                                link_domain = urlparse(href).netloc
                                
                                # Only follow same-domain links
                                if link_domain == base_domain:
                                    child_docs = await self._crawl(href, depth + 1)
                                    documents.extend(child_docs)
        
        except Exception as e:
            print(f"Error loading {url}: {e}")
        
        return documents
```

**CSV/Excel Loader (`app/services/rag/loaders/tabular_loader.py`)**

```python
import pandas as pd
import uuid

class TabularLoader(DocumentLoader):
    """Load CSV and Excel files"""
    
    def supports(self, source: str) -> bool:
        return source.lower().endswith(('.csv', '.xlsx', '.xls'))
    
    async def load(self, source: str) -> List[Document]:
        # Read file based on extension
        if source.endswith('.csv'):
            df = pd.read_csv(source)
        else:
            df = pd.read_excel(source, sheet_name=None)  # Read all sheets
            
            # If multiple sheets, process each
            if isinstance(df, dict):
                documents = []
                for sheet_name, sheet_df in df.items():
                    docs = await self._process_dataframe(sheet_df, source, sheet_name)
                    documents.extend(docs)
                return documents
        
        return await self._process_dataframe(df, source)
    
    async def _process_dataframe(
        self, 
        df: pd.DataFrame, 
        source: str, 
        sheet_name: str = None
    ) -> List[Document]:
        documents = []
        
        # Strategy 1: Convert entire table to markdown
        markdown_table = df.to_markdown(index=False)
        
        documents.append(Document(
            content=markdown_table,
            metadata={
                "filename": Path(source).name,
                "sheet_name": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "type": "table_overview"
            },
            doc_id=str(uuid.uuid4()),
            source_type="tabular",
            created_at=datetime.utcnow()
        ))
        
        # Strategy 2: Create searchable text summaries
        # Each row as a document (good for product catalogs, FAQs, etc.)
        for idx, row in df.iterrows():
            row_text = "\n".join([
                f"{col}: {val}" 
                for col, val in row.items() 
                if pd.notna(val)
            ])
            
            documents.append(Document(
                content=row_text,
                metadata={
                    "filename": Path(source).name,
                    "sheet_name": sheet_name,
                    "row_index": int(idx),
                    "type": "table_row"
                },
                doc_id=str(uuid.uuid4()),
                source_type="tabular",
                created_at=datetime.utcnow()
            ))
        
        return documents
```

**API Loader (`app/services/rag/loaders/api_loader.py`)**

```python
import aiohttp
import json
from typing import Dict, Any, List
import uuid

class APILoader(DocumentLoader):
    """Load data from REST APIs"""
    
    def __init__(self, headers: Dict[str, str] = None):
        self.headers = headers or {}
    
    def supports(self, source: str) -> bool:
        return source.startswith(('http://', 'https://')) and 'api' in source.lower()
    
    async def load(self, source: str) -> List[Document]:
        """Load from API endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(source, headers=self.headers) as response:
                data = await response.json()
                
                return await self._process_json(data, source)
    
    async def _process_json(self, data: Any, source: str) -> List[Document]:
        documents = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of objects
            for i, item in enumerate(data):
                doc_text = self._json_to_text(item)
                
                documents.append(Document(
                    content=doc_text,
                    metadata={
                        "source": source,
                        "index": i,
                        "type": "api_item"
                    },
                    doc_id=str(uuid.uuid4()),
                    source_type="api",
                    created_at=datetime.utcnow()
                ))
        
        elif isinstance(data, dict):
            # Single object or paginated response
            if 'results' in data or 'data' in data:
                # Paginated response
                items = data.get('results') or data.get('data')
                return await self._process_json(items, source)
            else:
                # Single object
                doc_text = self._json_to_text(data)
                documents.append(Document(
                    content=doc_text,
                    metadata={
                        "source": source,
                        "type": "api_object"
                    },
                    doc_id=str(uuid.uuid4()),
                    source_type="api",
                    created_at=datetime.utcnow()
                ))
        
        return documents
    
    def _json_to_text(self, data: Dict[str, Any]) -> str:
        """Convert JSON to searchable text"""
        lines = []
        
        def flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    flatten(value, new_prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    flatten(item, f"{prefix}[{i}]")
            else:
                lines.append(f"{prefix}: {obj}")
        
        flatten(data)
        return "\n".join(lines)
```

**Database Loader (`app/services/rag/loaders/database_loader.py`)**

```python
import asyncpg
from typing import List
import uuid

class DatabaseLoader(DocumentLoader):
    """Load data from database tables"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def supports(self, source: str) -> bool:
        return source.startswith(('postgresql://', 'mysql://'))
    
    async def load(self, source: str, table: str = None, query: str = None) -> List[Document]:
        """Load from database table or custom query"""
        conn = await asyncpg.connect(self.connection_string)
        
        try:
            if query:
                rows = await conn.fetch(query)
            elif table:
                rows = await conn.fetch(f"SELECT * FROM {table}")
            else:
                raise ValueError("Must provide either table or query")
            
            documents = []
            
            for row in rows:
                # Convert row to text
                row_dict = dict(row)
                text = "\n".join([
                    f"{key}: {value}"
                    for key, value in row_dict.items()
                ])
                
                documents.append(Document(
                    content=text,
                    metadata={
                        "table": table,
                        "source": "database",
                        **row_dict
                    },
                    doc_id=str(uuid.uuid4()),
                    source_type="database",
                    created_at=datetime.utcnow()
                ))
            
            return documents
        
        finally:
            await conn.close()
```

### Document Processing Orchestrator

**Main Ingestion Service (`app/services/rag/ingestion_service.py`)**

```python
from typing import List, Dict, Any
from app.services.rag.loaders import *
from app.services.rag.base import RAGService, TextSplitter

class IngestionService:
    """Orchestrates document loading, processing, and indexing"""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.loaders = [
            PDFLoader(),
            DOCXLoader(),
            WebLoader(),
            TabularLoader(),
            APILoader(),
            DatabaseLoader(),
        ]
    
    async def ingest(
        self,
        source: str,
        pool_id: str,
        source_type: str = "auto",
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest document from any source
        
        Args:
            source: Path, URL, or identifier
            pool_id: Knowledge pool to add to
            source_type: 'pdf', 'web', 'api', etc. or 'auto' to detect
            additional_metadata: Extra metadata to attach
        """
        
        # Step 1: Load documents
        loader = self._get_loader(source, source_type)
        documents = await loader.load(source)
        
        if not documents:
            raise ValueError(f"No documents loaded from {source}")
        
        # Step 2: Process and index
        indexed_count = 0
        failed_count = 0
        
        for doc in documents:
            try:
                # Add additional metadata
                if additional_metadata:
                    doc.metadata.update(additional_metadata)
                
                # Index into RAG system
                vector_ids = await self.rag_service.index_document(
                    pool_id=pool_id,
                    content=doc.content,
                    metadata=doc.metadata
                )
                
                indexed_count += 1
                
            except Exception as e:
                print(f"Failed to index document {doc.doc_id}: {e}")
                failed_count += 1
        
        return {
            "status": "success",
            "documents_loaded": len(documents),
            "documents_indexed": indexed_count,
            "documents_failed": failed_count,
            "source": source,
            "pool_id": pool_id
        }
    
    def _get_loader(self, source: str, source_type: str) -> DocumentLoader:
        """Get appropriate loader for source"""
        
        if source_type != "auto":
            # Find loader by type
            for loader in self.loaders:
                if loader.__class__.__name__.lower().startswith(source_type.lower()):
                    return loader
        
        # Auto-detect
        for loader in self.loaders:
            if loader.supports(source):
                return loader
        
        raise ValueError(f"No loader found for source: {source}")
```

### Intelligent Chunking Strategies

**Content-Aware Splitter (`app/services/rag/splitters/smart_splitter.py`)**

```python
from typing import List
import re

class SmartTextSplitter:
    """Context-aware text splitting"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        respect_boundaries: bool = True
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries
    
    def split_text(self, text: str, doc_type: str = "general") -> List[str]:
        """Split text based on content type"""
        
        if doc_type == "code":
            return self._split_code(text)
        elif doc_type == "markdown":
            return self._split_markdown(text)
        elif doc_type == "table":
            return self._split_table(text)
        else:
            return self._split_general(text)
    
    def _split_general(self, text: str) -> List[str]:
        """Split on sentence boundaries"""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_markdown(self, text: str) -> List[str]:
        """Split markdown preserving structure"""
        
        # Split on headers
        sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_section = []
        
        for i, section in enumerate(sections):
            if section.strip():
                if section.startswith('#'):
                    # New section header
                    if current_section:
                        chunks.append('\n'.join(current_section))
                    current_section = [section]
                else:
                    current_section.append(section)
        
        if current_section:
            chunks.append('\n'.join(current_section))
        
        return chunks
    
    def _split_code(self, text: str) -> List[str]:
        """Split code preserving function/class boundaries"""
        
        # Simple function-based splitting
        # This is a basic example - real implementation would use AST parsing
        
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        indent_stack = []
        
        for line in lines:
            line_length = len(line)
            
            # Detect function/class definitions
            if re.match(r'^(def|class|async def)\s+', line.strip()):
                # Start of new function/class
                if current_chunk and current_length > self.chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(line)
            current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _split_table(self, text: str) -> List[str]:
        """Keep tables intact"""
        # Don't split tables - return as single chunks
        return [text]
```

### API Endpoints for Data Ingestion

**Ingestion API (`app/api/ingestion.py`)**

```python
from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import shutil
from pathlib import Path

from app.dependencies import get_db, get_current_user
from app.services.rag.ingestion_service import IngestionService
from app.models.database import User

router = APIRouter()

class URLIngestionRequest(BaseModel):
    url: str
    pool_id: str
    crawl_depth: int = 0

class APIIngestionRequest(BaseModel):
    api_url: str
    pool_id: str
    headers: dict = {}
    auth_token: str | None = None

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    pool_id: str = None,
    background_tasks: BackgroundTasks = None,
    user: User = Depends(get_current_user),
    ingestion_service: IngestionService = Depends(),
):
    """Upload and process file"""
    
    # Save file temporarily
    upload_dir = Path("uploads") / str(user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(
        ingestion_service.ingest,
        source=str(file_path),
        pool_id=pool_id,
        additional_metadata={
            "uploaded_by": str(user.id),
            "original_filename": file.filename
        }
    )
    
    return {
        "status": "processing",
        "filename": file.filename,
        "pool_id": pool_id
    }

@router.post("/url")
async def ingest_url(
    request: URLIngestionRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    ingestion_service: IngestionService = Depends(),
):
    """Ingest content from URL"""
    
    background_tasks.add_task(
        ingestion_service.ingest,
        source=request.url,
        pool_id=request.pool_id,
        source_type="web",
        additional_metadata={
            "crawl_depth": request.crawl_depth,
            "ingested_by": str(user.id)
        }
    )
    
    return {
        "status": "processing",
        "url": request.url,
        "pool_id": request.pool_id
    }

@router.post("/api")
async def ingest_api(
    request: APIIngestionRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    ingestion_service: IngestionService = Depends(),
):
    """Ingest data from API endpoint"""
    
    headers = request.headers.copy()
    if request.auth_token:
        headers["Authorization"] = f"Bearer {request.auth_token}"
    
    # Initialize API loader with headers
    from app.services.rag.loaders.api_loader import APILoader
    loader = APILoader(headers=headers)
    
    background_tasks.add_task(
        ingestion_service.ingest,
        source=request.api_url,
        pool_id=request.pool_id,
        source_type="api"
    )
    
    return {
        "status": "processing",
        "api_url": request.api_url,
        "pool_id": request.pool_id
    }
```

### Best Practices Summary

**1. Choose Chunking Strategy by Content Type:**

- **General text**: Sentence boundaries (prevents mid-sentence cuts)
- **Code**: Function/class boundaries (preserves context)
- **Markdown**: Header boundaries (maintains structure)
- **Tables**: Keep intact (don't split rows)
- **Legal/Academic**: Paragraph boundaries (preserves arguments)

**2. Metadata Enrichment:**
Always extract and store:

- Source information (filename, URL, page number)
- Document structure (section, heading level)
- Timestamps (created, modified, indexed)
- Content type specific (has_tables, code_language, etc.)

**3. Quality Control:**

- Validate extracted text isn't garbled
- Check embedding generation succeeded
- Verify chunk sizes are reasonable
- Log failures for manual review

**4. Incremental Updates:**

- Track document versions
- Only re-index changed content
- Support partial updates

**5. Performance:**

- Batch embed multiple chunks together
- Process large files in background tasks
- Use connection pooling for databases
- Cache frequently accessed documents

## Advanced Features Implementation

### Memory System

The memory system automatically extracts and stores important facts about the user:

```python
# app/services/memory_service.py
from langchain.memory import ConversationSummaryMemory
from sqlalchemy.ext.asyncio import AsyncSession

class MemoryService:
    async def extract_memories(
        self,
        conversation_id: str,
        user_id: str,
        db: AsyncSession
    ):
        """Extract memories from conversation"""
        
        # Get recent messages
        messages = await self._get_recent_messages(conversation_id, db)
        
        # Use LLM to extract facts
        prompt = """
        Extract important facts about the user from this conversation.
        Focus on:
        - User preferences
        - Personal information
        - Goals and objectives
        - Recurring topics
        
        Return as JSON array of facts.
        """
        
        # Call LLM and parse response
        facts = await self._extract_facts(messages, prompt)
        
        # Store in database
        for fact in facts:
            memory = MemoryEntry(
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type='fact',
                content=fact['content'],
                importance_score=fact['importance']
            )
            db.add(memory)
        
        await db.commit()
```

### Knowledge Pool Management

Users can create and manage multiple knowledge pools:

```python
# app/api/rag.py
@router.post("/pools")
async def create_knowledge_pool(
    name: str,
    description: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Create new knowledge pool"""
    
    pool = KnowledgePool(
        user_id=user.id,
        name=name,
        description=description
    )
    db.add(pool)
    await db.commit()
    
    return {"id": str(pool.id), "name": pool.name}

@router.post("/pools/{pool_id}/documents")
async def upload_document(
    pool_id: str,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(),
):
    """Upload document to knowledge pool"""
    
    # Read file
    content = await file.read()
    text = content.decode('utf-8')
    
    # Index in vector store
    vector_ids = await rag_service.index_document(
        pool_id=pool_id,
        content=text,
        metadata={"filename": file.filename}
    )
    
    # Save to database
    doc = Document(
        pool_id=pool_id,
        filename=file.filename,
        content=text,
        vector_ids=vector_ids
    )
    db.add(doc)
    await db.commit()
    
    return {"document_id": str(doc.id)}
```

## Testing Strategy

### Backend Tests

```python
# tests/test_chat.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_stream_chat():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chat/stream",
            json={"message": "Hello"}
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

# tests/test_scratchpad.py
@pytest.mark.asyncio
async def test_create_scratchpad():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/scratchpad",
            json={
                "entry_type": "note",
                "content": "Test note"
            }
        )
    assert response.status_code == 200
```

### Frontend Tests

```typescript
// __tests__/ChatInterface.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatInterface } from '@/components/chat/ChatInterface';

describe('ChatInterface', () => {
  it('renders chat input', () => {
    render(<ChatInterface />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });
  
  it('sends message on submit', async () => {
    render(<ChatInterface />);
    const input = screen.getByRole('textbox');
    
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.submit(input.closest('form')!);
    
    // Assert message was sent
  });
});
```

## Deployment

### Production Checklist

- [ ] Environment variables secured
- [ ] Database migrations applied
- [ ] SSL certificates configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Monitoring and logging setup
- [ ] Backup strategy implemented
- [ ] API key rotation strategy
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring (DataDog/New Relic)

### Recommended Architecture

```
┌──────────────┐
│   Cloudflare │ ← SSL, DDoS protection
└──────┬───────┘
       │
┌──────▼───────┐
│   Vercel     │ ← Frontend (Next.js)
└──────┬───────┘
       │
┌──────▼───────┐
│   Railway/   │ ← Backend (FastAPI)
│   Render     │
└──────┬───────┘
       │
   ┌───┴────┬─────────┬─────────┐
   ▼        ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ RDS  │ │Redis │ │Qdrant│ │  S3  │
│Postgres│ │Cloud │ │Cloud │ │Files │
└──────┘ └──────┘ └──────┘ └──────┘
```

## Next Steps

1. **Week 1**: Set up backend with basic chat streaming
2. **Week 2**: Implement scratchpad with persistence
3. **Week 3**: Add RAG functionality with Qdrant
4. **Week 4**: Build frontend with split layout
5. **Week 5**: Implement memory system
6. **Week 6**: Testing and deployment

This architecture provides a solid foundation for a production-ready RAG system with all your requested features. Start with the MVP (chat + scratchpad) and incrementally add RAG, memory, and artifacts as you validate each component.
