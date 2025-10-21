# RAG System Quick Start Guide

## Get Running in 2 Hours

This guide gets you from zero to a working RAG chat system with scratchpad in the fastest time possible.

## Prerequisites

```bash
# Required installations
- Python 3.11+
- Node.js 18+
- Docker Desktop
- Git
```

## 30-Minute MVP Setup

### Step 1: Project Structure (5 minutes)

```bash
# Create project
mkdir rag-chat-system
cd rag-chat-system

# Create directories
mkdir -p backend/app/{api,models,services,utils}
mkdir -p backend/alembic/versions
mkdir -p frontend/{app,components,lib,store}

# Initialize git
git init
echo "venv/
__pycache__/
.env
node_modules/
.next/" > .gitignore
```

### Step 2: Docker Services (5 minutes)

Create `docker-compose.yml`:

```yaml
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
```

Start services:

```bash
docker-compose up -d
```

### Step 3: Backend Minimal Setup (10 minutes)

**Create `backend/requirements.txt`:**

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
asyncpg==0.29.0
openai==1.10.0
qdrant-client==1.7.3
python-multipart==0.0.6
pydantic-settings==2.1.0
PyPDF2==3.0.1
```

**Install:**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Create `backend/.env`:**

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/ragchat
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=http://localhost:6333
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000"]
```

**Create `backend/app/main.py`:**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Chat System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn app.main:app --reload
```

### Step 4: Frontend Minimal Setup (10 minutes)

```bash
cd ../frontend
npx create-next-app@latest . --typescript --tailwind --app --no-src-dir
npm install ai react-resizable-panels zustand
```

**Create `frontend/app/page.tsx`:**

```typescript
'use client';

import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: 'http://localhost:8000/api/chat/stream',
  });

  return (
    <div className="flex flex-col h-screen p-4">
      <div className="flex-1 overflow-auto space-y-4">
        {messages.map(m => (
          <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
            <div className={`inline-block p-3 rounded-lg ${
              m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="mt-4">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Type a message..."
          className="w-full p-2 border rounded"
        />
      </form>
    </div>
  );
}
```

**Run:**

```bash
npm run dev
```

Visit `http://localhost:3000` - You now have a basic chat UI!

## Add Streaming Chat (20 minutes)

### Backend: Add Streaming Endpoint

**Create `backend/app/api/chat.py`:**

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import os

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        try:
            stream = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": request.message}],
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
```

**Update `backend/app/main.py`:**

```python
from app.api import chat

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
```

Restart backend - streaming now works!

## Add Scratchpad (30 minutes)

### Backend: Scratchpad API

**Create `backend/app/api/scratchpad.py`:**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import json

router = APIRouter()

# Simple in-memory storage for MVP (use database in production)
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
```

**Add to main.py:**

```python
from app.api import chat, scratchpad

app.include_router(scratchpad.router, prefix="/api/scratchpad", tags=["scratchpad"])
```

### Frontend: Scratchpad Component

**Create `frontend/components/Scratchpad.tsx`:**

```typescript
'use client';

import { useState, useEffect } from 'react';

export function Scratchpad() {
  const [todos, setTodos] = useState<Array<{id: string; text: string; done: boolean}>>([]);
  const [notes, setNotes] = useState('');
  const [newTodo, setNewTodo] = useState('');

  // Load from backend
  useEffect(() => {
    fetch('http://localhost:8000/api/scratchpad')
      .then(r => r.json())
      .then(data => {
        setTodos(data.todos || []);
        setNotes(data.notes || '');
      });
  }, []);

  // Auto-save
  useEffect(() => {
    const timer = setTimeout(() => {
      fetch('http://localhost:8000/api/scratchpad', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ todos, notes, journal: '' })
      });
    }, 1000);
    return () => clearTimeout(timer);
  }, [todos, notes]);

  const addTodo = () => {
    if (!newTodo.trim()) return;
    setTodos([...todos, { id: Date.now().toString(), text: newTodo, done: false }]);
    setNewTodo('');
  };

  return (
    <div className="p-4 space-y-4">
      <div>
        <h2 className="font-bold mb-2">Todos</h2>
        <div className="flex gap-2 mb-2">
          <input
            value={newTodo}
            onChange={e => setNewTodo(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && addTodo()}
            placeholder="Add todo..."
            className="flex-1 p-2 border rounded"
          />
          <button onClick={addTodo} className="px-4 py-2 bg-blue-500 text-white rounded">
            Add
          </button>
        </div>
        <div className="space-y-1">
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
        </div>
      </div>

      <div>
        <h2 className="font-bold mb-2">Notes</h2>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          className="w-full h-40 p-2 border rounded"
          placeholder="Write notes..."
        />
      </div>
    </div>
  );
}
```

**Update `frontend/app/page.tsx` to use split layout:**

```typescript
'use client';

import { useChat } from 'ai/react';
import { Scratchpad } from '@/components/Scratchpad';

export default function Home() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: 'http://localhost:8000/api/chat/stream',
  });

  return (
    <div className="flex h-screen">
      {/* Scratchpad */}
      <div className="w-80 border-r overflow-auto">
        <Scratchpad />
      </div>

      {/* Chat */}
      <div className="flex-1 flex flex-col p-4">
        <div className="flex-1 overflow-auto space-y-4">
          {messages.map(m => (
            <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
              <div className={`inline-block p-3 rounded-lg ${
                m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
              }`}>
                {m.content}
              </div>
            </div>
          ))}
        </div>
        
        <form onSubmit={handleSubmit} className="mt-4">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Type a message..."
            className="w-full p-2 border rounded"
          />
        </form>
      </div>
    </div>
  );
}
```

## Add Basic RAG (30 minutes)

### Backend: RAG Service

**Create `backend/app/services/simple_rag.py`:**

```python
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
        """Add document to RAG"""
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
        """Search for relevant documents"""
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
```

**Update `backend/app/api/chat.py` to use RAG:**

```python
from app.services.simple_rag import SimpleRAG

rag = SimpleRAG()

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        try:
            messages = [{"role": "user", "content": request.message}]
            
            # Add RAG context if enabled
            if request.use_rag:
                docs = await rag.search(request.message)
                if docs:
                    context = "\n\n".join([d["text"] for d in docs])
                    messages[0]["content"] = f"Context:\n{context}\n\nQuestion: {request.message}"
            
            stream = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/upload")
async def upload_document(text: str):
    """Quick document upload"""
    doc_id = await rag.add_document(text)
    return {"id": doc_id, "status": "indexed"}
```

### Frontend: Add RAG & Scratchpad Toggles

**Update `frontend/app/page.tsx`:**

```typescript
export default function Home() {
  const [useRag, setUseRag] = useState(false);
  const [useScratchpad, setUseScratchpad] = useState(true);
  const [scratchpadContent, setScratchpadContent] = useState('');
  
  // Get scratchpad content when toggle is enabled
  useEffect(() => {
    if (useScratchpad) {
      fetch('http://localhost:8000/api/scratchpad')
        .then(r => r.json())
        .then(data => {
          const parts = [];
          if (data.todos?.length > 0) {
            parts.push('Todos:\n' + data.todos.map(t => 
              `- [${t.done ? 'x' : ' '}] ${t.text}`
            ).join('\n'));
          }
          if (data.notes) parts.push(`Notes:\n${data.notes}`);
          if (data.journal) parts.push(`Journal:\n${data.journal}`);
          setScratchpadContent(parts.join('\n\n'));
        });
    }
  }, [useScratchpad]);
  
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: 'http://localhost:8000/api/chat/stream',
    body: { 
      use_rag: useRag,
      scratchpad_context: useScratchpad ? scratchpadContent : null 
    },
  });

  return (
    <div className="flex h-screen">
      <div className="w-80 border-r overflow-auto">
        <Scratchpad />
      </div>

      <div className="flex-1 flex flex-col p-4">
        {/* Context Toggles */}
        <div className="mb-4 flex items-center gap-4 p-3 bg-gray-50 rounded">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useRag}
              onChange={e => setUseRag(e.target.checked)}
            />
            <span className="text-sm">Use RAG Knowledge</span>
          </label>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useScratchpad}
              onChange={e => setUseScratchpad(e.target.checked)}
            />
            <span className="text-sm">Include Scratchpad Context</span>
          </label>
        </div>

        <div className="flex-1 overflow-auto space-y-4">
          {messages.map(m => (
            <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
              <div className={`inline-block p-3 rounded-lg ${
                m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
              }`}>
                {m.content}
              </div>
            </div>
          ))}
        </div>
        
        <form onSubmit={handleSubmit} className="mt-4">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Type a message..."
            className="w-full p-2 border rounded"
          />
        </form>
      </div>
    </div>
  );
}
```

## Testing Your System

### 1. Test Chat

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Browser: http://localhost:3000
```

### 2. Test RAG

```bash
# Add a document via API
curl -X POST "http://localhost:8000/api/chat/upload" \
  -H "Content-Type: application/json" \
  -d '{"text": "The capital of France is Paris. It is known for the Eiffel Tower."}'

# In the chat UI:
# 1. Enable "Use RAG Knowledge" toggle
# 2. Ask: "What is the capital of France?"
# Should get context-aware response!
```

### 3. Test Scratchpad

- Add todos in the left panel
- Write notes
- Refresh page - should persist!

## Next Steps

### Immediate (This Week)

1. **Add File Upload**: Use the PDF loader from main guide
2. **Add Markdown Rendering**: Install `react-markdown`
3. **Improve UI**: Add Tailwind styling, better layouts

### Near Term (Next 2 Weeks)

1. **Database Persistence**: Replace in-memory storage with PostgreSQL
2. **User Authentication**: Add JWT auth
3. **Multiple Knowledge Pools**: Let users create different RAG collections

### Long Term (Month 2)

1. **Memory System**: Track user preferences across conversations
2. **Artifacts Panel**: Add code preview panel
3. **Advanced RAG**: Implement hybrid search, re-ranking

## Troubleshooting

**"Connection refused" errors:**

```bash
# Check Docker containers are running
docker ps

# Restart if needed
docker-compose restart
```

**"Module not found" errors:**

```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

**OpenAI API errors:**

- Verify API key in `.env`
- Check API quota at platform.openai.com
- Try using `gpt-3.5-turbo` instead (cheaper)

**Qdrant connection issues:**

```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Should return: {"result":{"collections":[...]}}
```

## Production Deployment

**Quick Deploy Options:**

1. **Backend → Railway.app**
   - Connect GitHub repo
   - Add environment variables
   - Deploy automatically

2. **Frontend → Vercel**
   - Import GitHub repo
   - Zero config deployment
   - Automatic HTTPS

3. **Database → Supabase** (PostgreSQL)
   - Free tier available
   - Built-in auth
   - Real-time subscriptions

**Total Cost: ~$0-20/month for MVP**

## Resources

- **Full Guide**: See the complete development guide for production details
- **Vercel AI SDK Docs**: <https://sdk.vercel.ai/docs>
- **FastAPI Docs**: <https://fastapi.tiangolo.com>
- **Next.js Docs**: <https://nextjs.org/docs>

You now have a working RAG chat system with scratchpad! 🎉
