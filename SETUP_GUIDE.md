# RAG System Setup Guide

Complete guide to set up and test the RAG chat system with all features.

## Prerequisites

- Docker Desktop installed and running
- Python 3.13+
- Node.js 18+
- uv (Python package manager) or pip

## Quick Start (5 Minutes)

### 1. Start Infrastructure Services

```bash
# Start PostgreSQL, Redis, and Qdrant
docker-compose up -d

# Verify services are running
docker ps
```

Expected output:
- `rag-system-postgres-1` on port 5432
- `rag-system-redis-1` on port 6379
- `rag-system-qdrant-1` on port 6333

### 2. Install Backend Dependencies

```bash
cd backend

# With uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### 3. Run Database Migrations

```bash
# Create initial migration
alembic revision --autogenerate -m "initial schema"

# Apply migration
alembic upgrade head

# Verify tables created
alembic current
```

Expected tables:
- users
- conversations
- messages
- scratchpad_entries
- knowledge_pools
- documents
- user_memories

### 4. Create Default User (Temporary)

```bash
# Connect to PostgreSQL
docker exec -it rag-system-postgres-1 psql -U user -d ragchat

# Insert default user
INSERT INTO users (id, email, username, hashed_password, is_active, created_at, updated_at)
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'demo@example.com',
  'demo',
  'not-a-real-hash',
  true,
  NOW(),
  NOW()
);

# Exit psql
\q
```

### 5. Start Backend Server

```bash
# From backend/ directory
uvicorn app.main:app --reload --port 8000
```

Backend will be available at: http://localhost:8000

Check API docs: http://localhost:8000/docs

### 6. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 7. Start Frontend Server

```bash
npm run dev
```

Frontend will be available at: http://localhost:3000

---

## Testing the RAG System

### Test 1: Scratchpad Functionality

1. Open http://localhost:3000
2. Toggle the scratchpad panel (left side)
3. **Test Todos:**
   - Add a todo: "Test RAG document upload"
   - Mark it as complete
   - Add another: "Search for uploaded content"
4. **Test Notes:**
   - Switch to Notes tab
   - Write some notes about the system
5. **Test Journal:**
   - Switch to Journal tab
   - Write today's journal entry

**Verify:** Refresh the page - all data should persist (now stored in PostgreSQL)

### Test 2: RAG Document Upload

Using the API (Swagger UI at http://localhost:8000/docs):

1. **Create a Knowledge Pool:**
   ```
   POST /api/rag/pools
   {
     "name": "Test Documents",
     "description": "Testing RAG functionality"
   }
   ```
   Note the returned `pool_id`.

2. **Upload a Test Document:**
   ```
   POST /api/rag/upload
   - file: [upload a PDF or TXT file]
   - pool_id: [paste pool_id from step 1]
   ```

   The document will be processed in the background:
   - Text extracted
   - Split into chunks
   - Embedded with OpenAI
   - Stored in Qdrant

3. **Check Document Status:**
   ```
   GET /api/rag/pools
   ```
   Look for `num_chunks` in the response to confirm processing completed.

### Test 3: Semantic Search

Using Swagger UI:

```
POST /api/rag/search
{
  "query": "What is this document about?",
  "knowledge_pool_ids": ["<your-pool-id>"],
  "limit": 5
}
```

**Expected Response:**
```json
{
  "query": "What is this document about?",
  "results": [
    {
      "document_id": "...",
      "filename": "test.pdf",
      "content": "relevant chunk of text...",
      "score": 0.85,
      "metadata": {...}
    }
  ],
  "num_results": 5
}
```

### Test 4: Chat with RAG Context

1. Open the chat interface at http://localhost:3000
2. Toggle "Use RAG" in the context pills
3. Ask a question about your uploaded document
4. The AI will retrieve relevant chunks and use them to answer

**Example conversation:**
```
You: What are the main topics in the uploaded document?
AI: Based on the uploaded document, the main topics are...
    [Response uses RAG search results as context]
```

### Test 5: Scratchpad Context Injection

1. Add some todos and notes in the scratchpad
2. Toggle "Use Scratchpad" in the context pills
3. Ask the AI about your tasks

**Example:**
```
You: What tasks do I need to complete today?
AI: Based on your scratchpad, you have the following tasks:
    - Test RAG document upload
    - Search for uploaded content
```

---

## Verifying Database Persistence

### Check Scratchpad Entries

```bash
docker exec -it rag-system-postgres-1 psql -U user -d ragchat

SELECT entry_type, content, is_completed
FROM scratchpad_entries
WHERE user_id = '00000000-0000-0000-0000-000000000001';
```

### Check Documents

```sql
SELECT filename, status, num_chunks, num_tokens
FROM documents
JOIN knowledge_pools ON documents.knowledge_pool_id = knowledge_pools.id
WHERE knowledge_pools.user_id = '00000000-0000-0000-0000-000000000001';
```

### Check Qdrant Collections

```bash
# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/{collection_name}
```

---

## Troubleshooting

### Database Connection Error

**Error:** `could not connect to server`

**Fix:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# If not running, start it
docker-compose up -d postgres

# Check logs
docker logs rag-system-postgres-1
```

### Alembic Migration Error

**Error:** `Target database is not up to date`

**Fix:**
```bash
# Check current version
alembic current

# Upgrade to latest
alembic upgrade head

# If stuck, stamp the database
alembic stamp head
```

### Qdrant Connection Error

**Error:** `Failed to connect to Qdrant`

**Fix:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant

# Verify Qdrant is healthy
curl http://localhost:6333/healthz
```

### Frontend Can't Reach Backend

**Error:** `Network error` or `CORS error`

**Fix:**
1. Check backend is running: http://localhost:8000/docs
2. Verify CORS settings in `backend/.env`:
   ```
   CORS_ORIGINS=["http://localhost:3000"]
   ```
3. Restart backend server

### OpenAI API Error

**Error:** `Authentication failed`

**Fix:**
1. Check your API key in `backend/.env`:
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
2. Verify key is valid at https://platform.openai.com/api-keys
3. Check you have credits available

---

## Production Checklist

Before deploying to production:

- [ ] Enable user authentication (replace `DEFAULT_USER_ID`)
- [ ] Set strong `SECRET_KEY` in environment
- [ ] Configure PostgreSQL connection pooling
- [ ] Set up Redis for session storage
- [ ] Enable HTTPS/TLS
- [ ] Configure Qdrant authentication
- [ ] Set up monitoring and logging
- [ ] Implement rate limiting
- [ ] Add error tracking (Sentry, etc.)
- [ ] Configure backups for PostgreSQL
- [ ] Set environment to production in config

---

## API Endpoints Summary

### Scratchpad
- `GET /api/scratchpad/` - Get scratchpad data
- `POST /api/scratchpad/` - Save scratchpad data

### RAG
- `POST /api/rag/pools` - Create knowledge pool
- `GET /api/rag/pools` - List knowledge pools
- `DELETE /api/rag/pools/{pool_id}` - Delete knowledge pool
- `POST /api/rag/upload` - Upload document
- `POST /api/rag/search` - Semantic search

### Chat
- `POST /api/chat/stream` - Streaming chat (SSE)

---

## Next Steps

1. **Add Authentication:**
   - Implement JWT token generation
   - Create login/register endpoints
   - Add user dependency to all routes

2. **Enhance RAG:**
   - Add document deletion endpoint
   - Implement web scraping loader
   - Add conversation memory extraction

3. **Improve UX:**
   - Add document upload UI in frontend
   - Show processing status in real-time
   - Add conversation history sidebar

4. **Scale:**
   - Configure connection pooling
   - Add caching layer with Redis
   - Implement background task queue (Celery)

---

## Architecture Overview

```
┌─────────────────┐
│   Frontend      │
│   (Next.js)     │──────┐
└─────────────────┘      │
                         │ HTTP/SSE
┌─────────────────┐      │
│   Backend       │◄─────┘
│   (FastAPI)     │
└────────┬────────┘
         │
         ├──► PostgreSQL (Users, Docs, Scratchpad)
         ├──► Qdrant (Vector Search)
         ├──► Redis (Sessions)
         └──► OpenAI (Embeddings, Chat)
```

## Key Files

### Backend
- `backend/app/models/database.py` - SQLAlchemy models
- `backend/app/services/rag_service.py` - RAG orchestration
- `backend/app/api/rag.py` - RAG endpoints
- `backend/app/api/scratchpad.py` - Scratchpad endpoints
- `backend/alembic/` - Database migrations

### Frontend
- `frontend/app/page.tsx` - Main chat UI
- `frontend/components/Scratchpad.tsx` - Scratchpad component
- `frontend/app/api/chat/route.ts` - Chat API route with context injection

---

**You've successfully set up a production-ready RAG system!**
