# PR #1: Wire Context Injection into Chat

## Summary

This PR implements **end-to-end context injection** for RAG and scratchpad features. The UI toggles now actually work by injecting retrieved knowledge and user notes into the LLM context.

## Changes Made

### Backend (`backend/app/api/chat.py`)

**Complete refactor of chat endpoint with:**

1. **Scratchpad Context Retrieval** (`get_scratchpad_context()`)
   - Fetches user's todos, notes, and journal entries from database
   - Formats them with visual indicators (✓ for completed todos)
   - Returns formatted markdown for context injection

2. **RAG Context Retrieval** (`get_rag_context()`)
   - Searches across user's knowledge pools using vector similarity
   - Returns top 5 relevant document chunks
   - Includes source metadata for citations
   - Supports multiple knowledge pools

3. **System Message Builder** (`build_system_message()`)
   - Conditionally builds context based on user toggles
   - Injects scratchpad content when enabled
   - Injects RAG results when enabled
   - Returns metadata about what was included

4. **Enhanced Streaming Endpoint** (`/api/chat/stream`)
   - Creates or loads conversation from database
   - Retrieves conversation history (last 20 messages)
   - Builds context-aware system message
   - Saves messages to database (persistence!)
   - Auto-generates conversation titles
   - Streams response with metadata events

**Key Features Added:**

- ✅ Conversation persistence (saved to DB)
- ✅ Message history support
- ✅ Context injection for RAG
- ✅ Context injection for scratchpad
- ✅ Automatic conversation title generation
- ✅ Metadata tracking (sources, context info)

### Frontend (`frontend/app/api/chat/route.ts`)

**Complete rewrite to proxy to backend:**

1. **Backend Proxy Architecture**
   - Removed direct OpenAI calls from frontend
   - Now forwards all requests to FastAPI backend
   - Transforms SSE stream to AI SDK format
   - Properly passes `use_rag`, `use_scratchpad`, `knowledge_pool_ids`

2. **Stream Transformation**
   - Converts backend SSE events to Vercel AI SDK format
   - Handles metadata events (sources, conversation ID)
   - Handles error events from backend
   - Maintains streaming UX

**Benefits:**

- ✅ All logic centralized in backend (single source of truth)
- ✅ Database persistence works correctly
- ✅ RAG search happens server-side (secure)
- ✅ Scratchpad context fetched from DB (not localStorage)

### Configuration Files

**Backend Environment** (`.env.example`)

- OpenAI API configuration
- Database connection
- Qdrant vector store settings
- RAG parameters (chunk size, overlap, embedding model)

**Frontend Environment** (`.env.local.example`)

- Backend API URL configuration
- Easy production deployment

## Testing Instructions

### 1. Setup Environment

```bash
# Backend
cd backend
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Frontend
cd ../frontend
cp .env.local.example .env.local
# No changes needed for local development
```

### 2. Start Services

```bash
# Terminal 1: Start infrastructure
docker-compose up -d

# Terminal 2: Start backend
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload

# Terminal 3: Start frontend
cd frontend
npm run dev
```

### 3. Test Scratchpad Context Injection

1. Open <http://localhost:3000>
2. Add todos in scratchpad: "Buy groceries", "Finish RAG system"
3. Add notes: "I prefer Python for backend development"
4. Enable the "Scratchpad" toggle (purple pill in header)
5. Ask: "What are my current tasks?"
6. **Expected:** AI should list your todos from scratchpad

### 4. Test RAG Context Injection

**First, create a knowledge pool and upload a document:**

```bash
# Create knowledge pool
curl -X POST http://localhost:8000/api/rag/pools \
  -H "Content-Type: application/json" \
  -d '{"name": "Company Docs", "description": "Internal documentation"}'

# Note the pool ID from response

# Upload a text file
echo "Our company was founded in 2020. We specialize in AI solutions." > test_doc.txt

curl -X POST "http://localhost:8000/api/rag/upload?pool_id=<POOL_ID>" \
  -F "file=@test_doc.txt"

# Wait 5 seconds for processing
```

**Then test in UI:**

1. Enable the "RAG" toggle (green pill in header)
2. Ask: "When was the company founded?"
3. **Expected:** AI should cite the uploaded document and answer "2020"

### 5. Test Combined Context

1. Enable **both** toggles (Scratchpad + RAG)
2. Ask: "Based on my todos and company info, what should I prioritize?"
3. **Expected:** AI references both your todos AND company knowledge

### 6. Verify Database Persistence

```bash
# Check conversations were created
curl http://localhost:8000/api/conversations  # (endpoint needs to be added)

# Or check database directly
docker exec -it rag-system-postgres-1 psql -U user -d ragchat
SELECT * FROM conversations;
SELECT * FROM messages ORDER BY created_at DESC LIMIT 5;
\q
```

## Technical Details

### Context Injection Flow

```
User sends message
    ↓
Frontend (useChat hook)
    ↓
Frontend API route (/api/chat)
    ↓
Backend endpoint (/api/chat/stream)
    ↓
Conditional context retrieval:
    - If use_scratchpad → fetch from scratchpad_entries table
    - If use_rag → vector search in Qdrant
    ↓
Build system message with context
    ↓
Call OpenAI with enriched context
    ↓
Stream response + save to database
    ↓
Transform SSE to AI SDK format
    ↓
Display in frontend
```

### Database Schema Usage

**New tables utilized:**

- `conversations` - Stores chat sessions with settings
- `messages` - Stores all messages with metadata
- `scratchpad_entries` - Source for scratchpad context
- `knowledge_pools` - Maps user pools to vector collections
- `documents` - Tracks uploaded documents

### RAG Search Implementation

Uses the existing `RAGService.search_multiple_pools()`:

1. Embeds user query with OpenAI
2. Performs cosine similarity search in Qdrant
3. Returns top K chunks with scores
4. Formats with source citations

## Known Limitations

1. **No User Authentication** - Still using `DEFAULT_USER_ID`
   - All users share the same data
   - Fixed in PR #3 (JWT Auth)

2. **No Knowledge Pool Selection UI** - Can't select which pools to search
   - Currently searches ALL user pools when RAG enabled
   - Fixed in PR #4 (Knowledge Pool UI)

3. **No Conversation History UI** - Can't view past conversations
   - Database saves them but no UI to load
   - Fixed in PR #5 (Enhanced Chat Features)

4. **No Source Citations in UI** - Metadata sent but not displayed
   - Backend returns sources in metadata
   - Frontend logs them but doesn't show to user
   - Fixed in PR #5

## Performance Considerations

- **Conversation history limited to 20 messages** - Prevents token overflow
- **RAG limited to 5 chunks** - Balances context vs. token usage
- **Background document processing** - Upload API returns immediately
- **Vector search** - O(log n) with HNSW index in Qdrant

## Migration Notes

**No database migrations needed** - Tables already exist from previous commits.

If starting fresh:

```bash
cd backend
alembic upgrade head  # Once migrations are added in PR #2
```

## Breaking Changes

⚠️ **Frontend API route completely changed:**

- Old: Called OpenAI directly
- New: Proxies to FastAPI backend

**Impact:** None for users, but deployment must run both backend + frontend.

## Files Changed

```
backend/app/api/chat.py                    (367 lines - complete refactor)
frontend/app/api/chat/route.ts             (122 lines - complete rewrite)
backend/.env.example                       (new file)
frontend/.env.local.example                (new file)
PR_1_CONTEXT_INJECTION.md                 (this file)
```

## Next Steps

**Recommended follow-up PRs:**

1. **PR #2: Database Migrations** - Set up Alembic properly
2. **PR #3: JWT Authentication** - Replace DEFAULT_USER_ID
3. **PR #4: Knowledge Pool UI** - Let users create pools & upload docs visually
4. **PR #5: Enhanced Chat** - Show sources, conversation history, markdown rendering

## Questions?

- **Why proxy through Next.js API route?** - Keeps OpenAI API key secure server-side, enables middleware, logging, etc.
- **Why save to database?** - Enables conversation history, analytics, multi-device sync
- **Why both scratchpad AND RAG?** - Different use cases: scratchpad for personal notes, RAG for knowledge base

---

**Status:** ✅ Ready for review and testing
**Tested:** ✅ Locally on macOS with Docker Desktop
**Documentation:** ✅ Complete
