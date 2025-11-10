# Memory System Test Documentation

This document describes the test suite for the User Memory System.

## Test Files

### 1. `test_memory_service.py`
**Unit tests for the MemoryService class**

Tests the core memory service functionality including:
- Memory extraction from conversations
- Memory extraction from journal entries
- Memory storage with deduplication
- Multi-factor retrieval (semantic + recency + importance)
- Memory CRUD operations
- User isolation and authorization
- Qdrant integration

**Test Coverage:**
- ✅ Extract memories from conversation (20+ messages)
- ✅ Extract memories from journal entries
- ✅ Handle empty extraction results
- ✅ Add new memories with embeddings
- ✅ Deduplicate similar memories
- ✅ Retrieve memories with multi-factor scoring
- ✅ User isolation (users can't access others' memories)
- ✅ Get all memories sorted by importance
- ✅ Update memory content and importance
- ✅ Delete individual memories
- ✅ Delete all user memories
- ✅ Format memories for LLM context

### 2. `tests/api/test_memory.py`
**Integration tests for Memory API endpoints**

Tests all REST API endpoints:
- `GET /api/memory/` - List all memories
- `POST /api/memory/` - Create memory
- `PUT /api/memory/{id}` - Update memory
- `DELETE /api/memory/{id}` - Delete memory
- `DELETE /api/memory/` - Delete all memories
- `POST /api/memory/extract/conversation/{id}` - Extract from conversation
- `POST /api/memory/extract/journal` - Extract from journal
- `GET /api/memory/search` - Search memories

**Test Coverage:**
- ✅ Authentication and authorization
- ✅ User data isolation
- ✅ Request validation
- ✅ Successful operations
- ✅ Error handling (not found, unauthorized, etc.)
- ✅ Full lifecycle integration test

## Running the Tests

### Prerequisites

Ensure you have the testing dependencies installed:

```bash
cd backend
uv pip install -e ".[dev]"  # Installs pytest, pytest-asyncio, httpx, etc.
```

### Run All Memory Tests

```bash
# Run both memory service and API tests
pytest tests/test_memory_service.py tests/api/test_memory.py -v

# With coverage
pytest tests/test_memory_service.py tests/api/test_memory.py --cov=app.services.memory_service --cov=app.api.memory --cov-report=html
```

### Run Specific Test Files

```bash
# Memory service tests only
pytest tests/test_memory_service.py -v

# Memory API tests only
pytest tests/api/test_memory.py -v
```

### Run Specific Test Functions

```bash
# Run a specific test
pytest tests/test_memory_service.py::test_extract_memories_from_conversation -v

# Run tests matching a pattern
pytest tests/ -k "memory" -v

# Run tests matching multiple patterns
pytest tests/ -k "memory and extract" -v
```

### Run with Different Verbosity

```bash
# Minimal output
pytest tests/test_memory_service.py

# Verbose output (shows each test name)
pytest tests/test_memory_service.py -v

# Very verbose (shows print statements)
pytest tests/test_memory_service.py -vv

# Show test output even for passing tests
pytest tests/test_memory_service.py -s
```

### Run with Coverage Report

```bash
# Terminal coverage report
pytest tests/test_memory_service.py tests/api/test_memory.py --cov=app.services.memory_service --cov=app.api.memory

# HTML coverage report (opens in browser)
pytest tests/test_memory_service.py tests/api/test_memory.py \
  --cov=app.services.memory_service \
  --cov=app.api.memory \
  --cov-report=html

# Then open htmlcov/index.html in your browser
```

### Run Tests in Parallel (Faster)

```bash
# Install pytest-xdist
uv pip install pytest-xdist

# Run tests in parallel using 4 workers
pytest tests/test_memory_service.py tests/api/test_memory.py -n 4
```

## Test Structure

### Fixtures

Tests use the following fixtures from `conftest.py`:

- `db_session` - Test database session (in-memory SQLite)
- `client` - Async HTTP client for API tests
- `test_user` - Authenticated test user
- `auth_headers` - Authentication headers with JWT token
- `another_user` - Second test user for isolation tests

Custom fixtures in test files:

- `mock_embeddings` - Mock OpenAI embeddings provider
- `mock_vector_store` - Mock Qdrant vector store
- `mock_llm_client` - Mock OpenAI LLM client
- `test_conversation` - Conversation with messages
- `test_journal_entries` - Sample journal entries
- `test_memory` - Single test memory
- `multiple_memories` - Multiple test memories

### Mocking Strategy

**External Services Mocked:**

1. **OpenAI Embeddings** - Uses deterministic hash-based embeddings
   - Consistent results for same text
   - No API calls or costs
   - Fast execution

2. **Qdrant Vector Store** - Uses in-memory mock implementation
   - Simulates vector search with cosine similarity
   - Supports CRUD operations
   - No external dependencies

3. **OpenAI LLM** - Returns predefined JSON responses
   - Simulates memory extraction
   - Predictable test outcomes
   - No API costs

**Why Mock?**

- **Speed:** Tests run in milliseconds instead of seconds
- **Reliability:** No network failures or API rate limits
- **Cost:** No OpenAI API charges during testing
- **Isolation:** Tests don't interfere with production data
- **Determinism:** Same inputs always produce same outputs

## Test Scenarios

### Happy Paths ✅

1. **Memory Extraction**
   - Extract from conversation with multiple messages
   - Extract from journal entries (7 days back)
   - Handle conversations with no extractable info

2. **Memory Storage**
   - Add new memory with embedding
   - Update existing memory content
   - Update memory importance score
   - Deduplicate similar memories

3. **Memory Retrieval**
   - Retrieve with semantic search
   - Multi-factor scoring (semantic + recency + importance)
   - Retrieve all memories sorted by importance
   - Search memories with query

4. **Memory Deletion**
   - Delete individual memory
   - Delete all user memories
   - Verify cascade deletion from Qdrant

### Edge Cases 🔍

1. **Authorization**
   - Unauthorized requests return 401
   - Users can't access other users' memories
   - Users can't delete other users' memories

2. **Validation**
   - Invalid importance values (< 0 or > 1) rejected
   - Missing required fields rejected
   - Invalid UUIDs return 404

3. **Empty Results**
   - No memories returns empty list
   - No extractable info returns empty list
   - Missing conversation returns 404

4. **Error Handling**
   - Update non-existent memory returns 404
   - Delete non-existent memory returns 404
   - Extraction from empty conversation handled gracefully

## Expected Test Results

When all tests pass, you should see:

```
tests/test_memory_service.py::test_extract_memories_from_conversation PASSED
tests/test_memory_service.py::test_extract_memories_from_journal PASSED
tests/test_memory_service.py::test_extract_memories_no_new_info PASSED
tests/test_memory_service.py::test_add_memory_success PASSED
tests/test_memory_service.py::test_add_memory_deduplication PASSED
tests/test_memory_service.py::test_retrieve_memories_multi_factor_scoring PASSED
tests/test_memory_service.py::test_retrieve_memories_user_isolation PASSED
tests/test_memory_service.py::test_get_all_memories PASSED
tests/test_memory_service.py::test_update_memory_content PASSED
tests/test_memory_service.py::test_update_memory_importance PASSED
tests/test_memory_service.py::test_update_memory_not_found PASSED
tests/test_memory_service.py::test_delete_memory_success PASSED
tests/test_memory_service.py::test_delete_all_memories PASSED
tests/test_memory_service.py::test_delete_memory_wrong_user PASSED
tests/test_memory_service.py::test_format_memories_for_context PASSED
tests/test_memory_service.py::test_format_memories_empty_list PASSED

tests/api/test_memory.py::test_get_memories_success PASSED
tests/api/test_memory.py::test_get_memories_empty PASSED
tests/api/test_memory.py::test_get_memories_with_limit PASSED
tests/api/test_memory.py::test_get_memories_unauthorized PASSED
tests/api/test_memory.py::test_get_memories_user_isolation PASSED
tests/api/test_memory.py::test_create_memory_success PASSED
tests/api/test_memory.py::test_create_memory_invalid_importance PASSED
tests/api/test_memory.py::test_create_memory_unauthorized PASSED
tests/api/test_memory.py::test_update_memory_content PASSED
tests/api/test_memory.py::test_update_memory_importance PASSED
tests/api/test_memory.py::test_update_memory_not_found PASSED
tests/api/test_memory.py::test_delete_memory_success PASSED
tests/api/test_memory.py::test_delete_memory_not_found PASSED
tests/api/test_memory.py::test_delete_memory_unauthorized PASSED
tests/api/test_memory.py::test_delete_all_memories_success PASSED
tests/api/test_memory.py::test_delete_all_memories_unauthorized PASSED
tests/api/test_memory.py::test_extract_from_conversation_success PASSED
tests/api/test_memory.py::test_extract_from_conversation_unauthorized PASSED
tests/api/test_memory.py::test_extract_from_journal_success PASSED
tests/api/test_memory.py::test_extract_from_journal_no_memories PASSED
tests/api/test_memory.py::test_extract_from_journal_unauthorized PASSED
tests/api/test_memory.py::test_search_memories_success PASSED
tests/api/test_memory.py::test_search_memories_no_results PASSED
tests/api/test_memory.py::test_search_memories_missing_query PASSED
tests/api/test_memory.py::test_search_memories_unauthorized PASSED
tests/api/test_memory.py::test_memory_lifecycle PASSED

======================== 42 passed in 2.5s =========================
```

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'app'
   ```
   **Solution:** Run pytest from the `backend/` directory

2. **Async Errors**
   ```
   RuntimeError: no running event loop
   ```
   **Solution:** Ensure `@pytest.mark.asyncio` decorator is present

3. **Database Errors**
   ```
   sqlite3.OperationalError: no such table
   ```
   **Solution:** Check that fixtures are properly configured in conftest.py

4. **Mock Not Working**
   ```
   AssertionError: Expected mock to be called
   ```
   **Solution:** Verify patch path matches actual import path

### Debug Mode

Run a single test with maximum verbosity:

```bash
pytest tests/test_memory_service.py::test_add_memory_success -vv -s --tb=long
```

Flags:
- `-vv` - Very verbose output
- `-s` - Show print statements
- `--tb=long` - Long traceback format

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Run Memory System Tests
  run: |
    cd backend
    pytest tests/test_memory_service.py tests/api/test_memory.py \
      --cov=app.services.memory_service \
      --cov=app.api.memory \
      --cov-report=xml \
      --cov-report=term

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./backend/coverage.xml
```

## Contributing

When adding new memory features:

1. Write tests FIRST (TDD approach)
2. Ensure tests cover:
   - Happy path
   - Error cases
   - Authorization
   - Edge cases
3. Maintain >80% code coverage
4. Run full test suite before committing

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- FastAPI Testing: [https://fastapi.tiangolo.com/tutorial/testing/](https://fastapi.tiangolo.com/tutorial/testing/)
