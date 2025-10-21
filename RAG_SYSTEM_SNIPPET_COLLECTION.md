# =============================================================================

# RAG SYSTEM CODE SNIPPETS COLLECTION

# Copy-paste ready code organized by component

# =============================================================================

# =============================================================================

# BACKEND - FASTAPI

# =============================================================================

# -----------------------------------------------------------------------------

# 1. Main Application (app/main.py)

# -----------------------------------------------------------------------------

"""
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

app = FastAPI(title="RAG Chat System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(scratchpad.router, prefix="/api/scratchpad", tags=["scratchpad"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
"""

# -----------------------------------------------------------------------------

# 2. Configuration (app/config.py)

# -----------------------------------------------------------------------------

"""
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
"""

# -----------------------------------------------------------------------------

# 3. Streaming Chat Endpoint (app/api/chat.py)

# -----------------------------------------------------------------------------

"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import os

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False
    scratchpad_context: str | None = None

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        try:
            messages = [{"role": "user", "content": request.message}]

            # Add scratchpad context
            if request.scratchpad_context:
                messages[0]["content"] = (
                    f"User's scratchpad:\n{request.scratchpad_context}\n\n"
                    f"Question: {request.message}"
                )
            
            stream = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
"""

# -----------------------------------------------------------------------------

# 4. Scratchpad API (app/api/scratchpad.py)

# -----------------------------------------------------------------------------

"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

router = APIRouter()

# In-memory storage (replace with database)

scratchpad_store: Dict[str, dict] = {}

class ScratchpadData(BaseModel):
    todos: List[dict] = []
    notes: str = ""
    journal: str = ""

@router.get("/")
async def get_scratchpad():
    return scratchpad_store.get("default", ScratchpadData().dict())

@router.post("/")
async def save_scratchpad(data: ScratchpadData):
    scratchpad_store["default"] = data.dict()
    return {"status": "saved"}

@router.post("/archive-daily")
async def archive_daily_entries():
    # Archive journal entries at end of day
    return {"message": "Archived"}
"""

# -----------------------------------------------------------------------------

# 5. Simple RAG Service (app/services/simple_rag.py)

# -----------------------------------------------------------------------------

"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI
import uuid
import os

class SimpleRAG:
    def __init__(self):
        self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.qdrant.get_collection(self.collection)
        except:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    
    async def add_document(self, text: str, metadata: dict = None):
        # Generate embedding
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        
        # Store in Qdrant
        point_id = str(uuid.uuid4())
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": text, **(metadata or {})}
            )]
        )
        return point_id
    
    async def search(self, query: str, limit: int = 3):
        # Generate query embedding
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Search
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [{"text": r.payload["text"], "score": r.score} for r in results]
"""

# -----------------------------------------------------------------------------

# 6. PDF Loader (app/services/rag/loaders/pdf_loader.py)

# -----------------------------------------------------------------------------

"""
import PyPDF2
from pathlib import Path
import uuid
from typing import List
from datetime import datetime

class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
        self.doc_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()

class PDFLoader:
    def load(self, source: str) -> List[Document]:
        documents = []

        with open(source, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                documents.append(Document(
                    content=text,
                    metadata={
                        "page": i + 1,
                        "total_pages": len(pdf.pages),
                        "filename": Path(source).name
                    }
                ))
        
        return documents
"""

# -----------------------------------------------------------------------------

# 7. Smart Text Splitter (app/services/rag/splitters/smart_splitter.py)

# -----------------------------------------------------------------------------

"""
import re
from typing import List

class SmartTextSplitter:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Create overlap
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
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
"""

# =============================================================================

# FRONTEND - NEXT.JS / REACT

# =============================================================================

# -----------------------------------------------------------------------------

# 8. Chat Interface Component (components/chat/ChatInterface.tsx)

# -----------------------------------------------------------------------------

"""
'use client';

import { useChat } from 'ai/react';
import { useState, useEffect } from 'react';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { useScratchpadStore } from '@/store/store';

export function ChatInterface() {
  const [useRAG, setUseRAG] = useState(false);
  const [useScratchpad, setUseScratchpad] = useState(true);
  const getScratchpadContent = useScratchpadStore((state) => state.getFullContent);
  
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    body: {
      use_rag: useRAG,
      scratchpad_context: useScratchpad ? getScratchpadContent() : null
    },
  });
  
  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b space-y-2">
        <h3 className="text-sm font-semibold text-gray-600">Context Options</h3>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useRAG}
              onChange={(e) => setUseRAG(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Use RAG Knowledge</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useScratchpad}
              onChange={(e) => setUseScratchpad(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Include Scratchpad</span>
          </label>
        </div>
        
        {/* Context indicator */}
        {(useRAG || useScratchpad) && (
          <div className="text-xs text-gray-500">
            Active: {[
              useRAG && 'RAG',
              useScratchpad && 'Scratchpad'
            ].filter(Boolean).join(', ')}
          </div>
        )}
      </div>
      
      <ChatMessages messages={messages} isLoading={isLoading} />
      <ChatInput
        input={input}
        handleInputChange={handleInputChange}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
      />
    </div>
  );
}
"""

# -----------------------------------------------------------------------------

# 9. Scratchpad Component (components/scratchpad/Scratchpad.tsx)

# -----------------------------------------------------------------------------

"""
'use client';

import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface Todo {
  id: string;
  text: string;
  done: boolean;
}

export function Scratchpad() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [notes, setNotes] = useState('');
  const [journal, setJournal] = useState('');

  useEffect(() => {
    loadScratchpad();
    const interval = setInterval(saveScratchpad, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadScratchpad = async () => {
    const response = await fetch('/api/scratchpad');
    const data = await response.json();
    setTodos(data.todos || []);
    setNotes(data.notes || '');
    setJournal(data.journal || '');
  };

  const saveScratchpad = async () => {
    await fetch('/api/scratchpad', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ todos, notes, journal }),
    });
  };

  const addTodo = (text: string) => {
    setTodos([...todos, { id: crypto.randomUUID(), text, done: false }]);
  };

  return (
    <div className="h-full flex flex-col p-4">
      <h2 className="text-lg font-semibold mb-4">Scratchpad</h2>
      <Tabs defaultValue="todos" className="flex-1">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="todos">Todos</TabsTrigger>
          <TabsTrigger value="notes">Notes</TabsTrigger>
          <TabsTrigger value="journal">Journal</TabsTrigger>
        </TabsList>

        <TabsContent value="todos">
          {todos.map(todo => (
            <div key={todo.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={todo.done}
                onChange={() => {
                  setTodos(todos.map(t => 
                    t.id === todo.id ? {...t, done: !t.done} : t
                  ));
                }}
              />
              <span className={todo.done ? 'line-through' : ''}>{todo.text}</span>
            </div>
          ))}
        </TabsContent>
        
        <TabsContent value="notes">
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            className="w-full h-full p-2 border rounded"
          />
        </TabsContent>
        
        <TabsContent value="journal">
          <textarea
            value={journal}
            onChange={(e) => setJournal(e.target.value)}
            className="w-full h-full p-2 border rounded"
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
"""

# -----------------------------------------------------------------------------

# 10. Split Layout Component (components/layout/SplitLayout.tsx)

# -----------------------------------------------------------------------------

"""
'use client';

import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { Scratchpad } from '@/components/scratchpad/Scratchpad';
import { useState } from 'react';

export function SplitLayout() {
  const [showArtifacts, setShowArtifacts] = useState(false);
  
  return (
    <PanelGroup direction="horizontal" autoSaveId="main-layout">
      <Panel defaultSize={20} minSize={15} maxSize={30} collapsible>
        <Scratchpad />
      </Panel>

      <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-blue-500" />
      
      <Panel defaultSize={showArtifacts ? 40 : 60} minSize={30}>
        <ChatInterface />
      </Panel>
      
      {showArtifacts && (
        <>
          <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-blue-500" />
          <Panel defaultSize={40} minSize={30} collapsible>
            <div className="p-4">Artifacts Panel</div>
          </Panel>
        </>
      )}
    </PanelGroup>
  );
}
"""

# -----------------------------------------------------------------------------

# 11. Zustand Store (store/store.ts)

# -----------------------------------------------------------------------------

"""
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
  
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  setNotes: (notes: string) => void;
  setJournal: (journal: string) => void;
  
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
      
      setNotes: (notes) => set({ notes }),
      setJournal: (journal) => set({ journal }),
      
      loadScratchpad: async () => {
        const response = await fetch('/api/scratchpad');
        const data = await response.json();
        set(data);
      },
      
      saveScratchpad: async () => {
        const state = get();
        await fetch('/api/scratchpad', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            todos: state.todos,
            notes: state.notes,
            journal: state.journal,
          }),
        });
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
        
        if (state.notes) parts.push(`Notes:\n${state.notes}`);
        if (state.journal) parts.push(`Journal:\n${state.journal}`);
        
        return parts.join('\n\n');
      },
    }),
    { name: 'scratchpad-storage' }
  )
);
"""

# =============================================================================

# CONFIGURATION FILES

# =============================================================================

# -----------------------------------------------------------------------------

# 12. Docker Compose (docker-compose.yml)

# -----------------------------------------------------------------------------

"""
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

volumes:
  postgres_data:
  qdrant_data:
"""

# -----------------------------------------------------------------------------

# 13. Backend Requirements (backend/requirements.txt)

# -----------------------------------------------------------------------------

"""
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
asyncpg==0.29.0
alembic==1.13.1
redis==5.0.1
pydantic==2.5.3
pydantic-settings==2.1.0
python-multipart==0.0.6
openai==1.10.0
qdrant-client==1.7.3
PyPDF2==3.0.1
pdfplumber==0.10.3
python-docx==1.1.0
beautifulsoup4==4.12.3
aiohttp==3.9.3
pandas==2.2.0
"""

# -----------------------------------------------------------------------------

# 14. Backend Environment Variables (backend/.env)

# -----------------------------------------------------------------------------

"""
DATABASE_URL=postgresql://user:pass@localhost:5432/ragchat
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=<http://localhost:6333>
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000"]
DEFAULT_MODEL=gpt-4-turbo-preview
"""

# -----------------------------------------------------------------------------

# 15. Frontend Package.json (frontend/package.json)

# -----------------------------------------------------------------------------

"""
{
  "name": "rag-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "ai": "^3.0.0",
    "react-resizable-panels": "^2.0.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0",
    "zustand": "^4.5.0",
    "tailwindcss": "^3.4.0"
  }
}
"""

# =============================================================================

# DATABASE SCHEMA

# =============================================================================

# -----------------------------------------------------------------------------

# 16. Database Schema SQL

# -----------------------------------------------------------------------------

"""
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Scratchpad entries
CREATE TABLE scratchpad_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    entry_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    is_archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Knowledge pools
CREATE TABLE knowledge_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_id UUID REFERENCES knowledge_pools(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    content TEXT,
    metadata JSONB,
    vector_ids TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX idx_scratchpad_user ON scratchpad_entries(user_id, created_at);
"""

# =============================================================================

# USAGE EXAMPLES

# =============================================================================

# -----------------------------------------------------------------------------

# 17. Example Usage - Adding Documents to RAG

# -----------------------------------------------------------------------------

"""

# Python script to add documents

import asyncio
from app.services.simple_rag import SimpleRAG

async def add_documents():
    rag = SimpleRAG()

    # Add single document
    doc_id = await rag.add_document(
        "Python is a high-level programming language.",
        metadata={"source": "tutorial", "topic": "python"}
    )
    print(f"Added document: {doc_id}")
    
    # Search
    results = await rag.search("What is Python?")
    for r in results:
        print(f"Score: {r['score']}, Text: {r['text']}")

asyncio.run(add_documents())
"""

# -----------------------------------------------------------------------------

# 18. Example Usage - File Upload

# -----------------------------------------------------------------------------

"""

# Upload PDF via API

curl -X POST "<http://localhost:8000/api/ingestion/upload>" \
  -F "file=@document.pdf" \
  -F "pool_id=my-pool-id"

# Upload from URL

curl -X POST "<http://localhost:8000/api/ingestion/url>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "<https://example.com/article>",
    "pool_id": "my-pool-id",
    "crawl_depth": 1
  }'
"""

# =============================================================================

# END OF CODE SNIPPETS

# =============================================================================

# To use these snippets

# 1. Copy relevant sections to your project files

# 2. Update import paths as needed

# 3. Replace placeholder values (API keys, URLs, etc.)

# 4. Install required dependencies from requirements.txt and package.json

# 5. Run docker-compose up -d to start services

# 6. Start backend: uvicorn app.main:app --reload

# 7. Start frontend: npm run dev
