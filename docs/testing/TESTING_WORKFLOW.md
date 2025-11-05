# Testing Workflow

## Overview

This document describes the testing strategy and workflow for the RAG system. The goal is to provide **fast feedback during development** through automated tests that catch bugs before they reach production.

## Quick Start

```bash
# Install test dependencies
cd backend
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/api/test_auth.py -v

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run tests in watch mode (install pytest-watch first)
uv run ptw
```

## Test Structure

```
backend/tests/
├── conftest.py              # Shared fixtures and test configuration
├── mocks/                   # Mock implementations for external services
│   ├── openai_mock.py      # Mock OpenAI API responses
│   └── qdrant_mock.py      # Mock Qdrant vector store
├── api/                     # API endpoint tests
│   ├── test_auth.py        # Authentication tests (22 tests)
│   ├── test_scratchpad.py  # Scratchpad CRUD tests (20 tests)
│   ├── test_chat.py        # Chat streaming tests (planned)
│   └── test_rag.py         # RAG functionality tests (planned)
├── services/                # Service layer tests (planned)
│   ├── test_rag_service.py
│   └── test_embeddings.py
└── utils/                   # Utility test helpers
    └── results_db.py       # Test results database for RAG evaluation
```

## Testing Philosophy

### Test Pyramid

```
     /\
    /E2E\          10% - Critical user flows only
   /------\
  / Integ  \       30% - API + Database integration
 /----------\
/   Unit     \     60% - Business logic, utilities
--------------
```

### Focus Areas

1. **API Integration Tests** (`tests/api/`)
   - Test complete request/response cycles
   - Mock external services (OpenAI, Qdrant)
   - Use in-memory SQLite for speed
   - Run in <5 seconds

2. **Service Layer Tests** (`tests/services/`)
   - Test business logic in isolation
   - Mock dependencies
   - Focus on edge cases

3. **RAG Quality Tests** (`scripts/e2e_rag_test.py`)
   - Automated evaluation with standard IR metrics
   - Git-correlated results tracking
   - Compare search quality across code changes

## Key Features

###  1. Fast Feedback

Tests run in **<10 seconds** using:
- In-memory SQLite database (no external dependencies)
- Mocked OpenAI API (deterministic responses)
- Mocked Qdrant vector store (no network calls)
- Pytest fixtures for efficient test data creation

### 2. Comprehensive Coverage

**Authentication Tests** (`test_auth.py`):
- ✅ User registration (duplicate email/username handling)
- ✅ Login with correct/incorrect credentials
- ✅ JWT token refresh
- ✅ Protected route authorization
- ✅ Data isolation between users

**Scratchpad Tests** (`test_scratchpad.py`):
- ✅ CRUD operations for todos/notes/journal
- ✅ Data persistence
- ✅ User data isolation
- ✅ Edge cases (long text, special characters, markdown)

### 3. Mock External Services

**OpenAI Mock** (`tests/mocks/openai_mock.py`):
- Deterministic embeddings (same text → same vector)
- Streaming and non-streaming chat completions
- Configurable responses for different queries
- No API costs, runs offline

**Qdrant Mock** (`tests/mocks/qdrant_mock.py`):
- In-memory vector storage
- Cosine similarity search
- Collection management
- Point CRUD operations

## Fixtures Reference

### Database Fixtures

```python
async def test_engine():
    """In-memory SQLite database engine."""

async def db_session(test_engine):
    """Database session with automatic rollback."""

async def client(db_session):
    """HTTP test client with overridden DB dependency."""
```

### Authentication Fixtures

```python
async def test_user(db_session):
    """Pre-created test user (email: test@example.com)."""

async def auth_headers(test_user):
    """JWT authentication headers for test_user."""

async def another_user(db_session):
    """Second test user for testing data isolation."""
```

### Mock Service Fixtures

```python
def mock_openai_client():
    """Mock OpenAI client with deterministic responses."""

def mock_qdrant_client():
    """Mock Qdrant client with in-memory storage."""

def mock_openai_embeddings():
    """Generate deterministic fake embeddings."""
```

## Writing New Tests

### Example: API Endpoint Test

```python
@pytest.mark.asyncio
async def test_create_knowledge_pool(client: AsyncClient, auth_headers: dict):
    """Test creating a new knowledge pool."""
    response = await client.post(
        "/api/rag/knowledge-pools",
        headers=auth_headers,
        json={
            "name": "My Documents",
            "description": "Personal document collection"
        }
    )

    assert response.status_code == 201
    data = response.json()

    assert data["name"] == "My Documents"
    assert "id" in data
    assert "created_at" in data
```

### Example: Service Layer Test

```python
@pytest.mark.asyncio
async def test_rag_search_with_reranking(mock_qdrant, mock_embeddings):
    """Test RAG search with reranking enabled."""
    rag_service = RAGService(
        embedding_provider=mock_embeddings,
        vector_store=mock_qdrant
    )

    results = await rag_service.search(
        query="What is machine learning?",
        limit=5,
        use_reranking=True
    )

    assert len(results) <= 5
    # Verify results are sorted by relevance
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
```

## RAG Quality Evaluation

The system includes an automated RAG evaluation framework at `backend/scripts/e2e_rag_test.py`.

### Running Evaluations

```bash
cd backend

# Run evaluation
make test-rag

# View history
make test-rag-history

# Compare two runs
make test-rag-compare RUNS='4 5'

# Force re-run (ignore cache)
make test-rag-force
```

### Metrics Tracked

- **Precision@K**: Proportion of retrieved documents that are relevant
- **Recall@K**: Proportion of relevant documents that are retrieved
- **MRR**: Mean Reciprocal Rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Latency**: p50, p95, avg response times

### Workflow

1. Baseline: Run evaluation on `main` branch
2. Feature branch: Make changes to RAG logic
3. Run evaluation again
4. Compare metrics to baseline
5. Commit if metrics improve or stay stable

**Example**:
```bash
# On main branch
git checkout main
make test-rag
# Result: P@5=0.85, R@5=0.90

# On feature branch
git checkout feature/improved-reranking
# Make changes to reranking logic
make test-rag
# Result: P@5=0.88 ↑, R@5=0.92 ↑

# Compare
make test-rag-compare RUNS='1 2'
# Shows +3.5% precision, +2.2% recall improvement
```

## Known Issues & Limitations

### Current Status

- ✅ Test infrastructure fully setup
- ✅ 42 API tests written (auth + scratchpad)
- ⚠️ 12/42 tests passing (async event loop issues)
- ❌ Chat streaming tests not yet written
- ❌ RAG API tests not yet written
- ❌ Frontend tests not yet setup

### Issues to Fix

1. **Async Event Loop Management**
   - Some tests fail with "Future attached to different loop"
   - Need to review pytest-asyncio fixture scoping
   - May need to use `pytest-asyncio` strict mode

2. **Database Type Compatibility**
   - PostgreSQL-specific types (JSONB, TSVECTOR) replaced with SQLite equivalents in tests
   - May cause subtle differences in behavior
   - Consider using PostgreSQL for tests via Docker

3. **Missing Test Coverage**
   - Chat streaming endpoints
   - RAG document upload and processing
   - Knowledge pool management
   - Conversation persistence

## CI/CD Integration

See `.github/workflows/test.yml` for GitHub Actions configuration.

**On every push**:
- Run all unit and integration tests
- Report failures in PR

**On main branch only**:
- Run RAG evaluation suite
- Store results for comparison

## Performance Goals

- ✅ API tests: <5 seconds
- ⏳ Full test suite: <30 seconds (when complete)
- ⏳ RAG evaluation: <2 minutes

## Next Steps

### Short Term (4-8 hours)

1. Fix async event loop issues in existing tests
2. Add chat streaming API tests
3. Add RAG upload/search API tests
4. Setup GitHub Actions CI/CD

### Medium Term (15-20 hours)

1. Add service layer unit tests
2. Setup frontend testing (Vitest + React Testing Library)
3. Add component tests for Scratchpad and Chat
4. Expand RAG evaluation test cases

### Long Term (40+ hours)

1. E2E tests with Playwright
2. Load testing
3. Security testing
4. Visual regression testing

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [HTTPX AsyncClient](https://www.python-httpx.org/async/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)

## Tips & Best Practices

1. **Run tests frequently**: Catch bugs early
2. **Write tests before fixing bugs**: Reproduce issue first
3. **Keep tests fast**: Mock external services
4. **Test edge cases**: Empty data, special characters, long text
5. **Use descriptive names**: Test name should explain what's being tested
6. **One assertion focus per test**: Makes failures easier to diagnose

## Getting Help

- Test failures? Check logs with `-v --tb=short`
- Slow tests? Use `pytest --duration=10` to find bottlenecks
- Coverage gaps? Run `pytest --cov=app --cov-report=html`
- Need examples? See existing tests in `tests/api/`
