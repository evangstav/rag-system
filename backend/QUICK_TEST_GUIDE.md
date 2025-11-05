# Quick Test Guide

## ✅ Status: ALL WORKING!

**42/42 tests passing (100%)** in 8.64 seconds

## Quick Commands

```bash
# Install dependencies (one-time setup)
cd backend
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run with output
uv run pytest -v

# Run specific test file
uv run pytest tests/api/test_auth.py
uv run pytest tests/api/test_scratchpad.py

# Run single test
uv run pytest tests/api/test_auth.py::test_login_success -v

# Stop on first failure (fast debugging)
uv run pytest -x

# Run only failed tests from last run
uv run pytest --lf

# Coverage report
uv run pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## What's Tested

### ✅ Authentication (22 tests)
- Registration (duplicates, validation, edge cases)
- Login (valid/invalid, inactive users)
- Token refresh (valid/invalid/expired)
- Authorization & data isolation

### ✅ Scratchpad (20 tests)
- Get/save operations
- Todos, notes, journal CRUD
- User data isolation
- Edge cases (markdown, unicode, long text)

## Development Workflow

1. **Make code change**
2. **Run tests**: `uv run pytest` (8.64 seconds)
3. **If green**: Commit ✅
4. **If red**: Fix immediately with fast feedback

## Key Features

- ⚡ **8.64 seconds** for full test suite
- 🎯 **100%** pass rate
- 🔒 **Isolated** - Uses in-memory SQLite
- 🚀 **Mocked** - No external API calls
- 📊 **Comprehensive** - Auth, CRUD, edge cases

## Troubleshooting

**Tests failing after git pull?**
```bash
# Reinstall dependencies
uv pip install -e ".[dev]"
```

**Want to see SQL queries?**
```bash
# Edit backend/tests/conftest.py line 59
echo=True  # Set to True
```

**Tests running slow?**
```bash
# Check you're using SQLite, not PostgreSQL
uv run pytest -v --tb=short
# Should see: sqlite+aiosqlite:///:memory:
```

## Files

- **Tests**: `backend/tests/api/`
- **Fixtures**: `backend/tests/conftest.py`
- **Mocks**: `backend/tests/mocks/`
- **Config**: `backend/pyproject.toml`

## Documentation

- **Workflow**: `docs/testing/TESTING_WORKFLOW.md`
- **README**: `backend/tests/README.md`
- **Fixes**: `ASYNC_FIXES_SUMMARY.md`
- **Summary**: `TESTING_IMPLEMENTATION_SUMMARY.md`

## Next Steps

Want to add more tests?

1. Copy existing test pattern
2. Use fixtures: `client`, `auth_headers`, `test_user`
3. Write test following `test_<action>_<condition>` naming
4. Run: `uv run pytest tests/api/test_myfile.py -v`

**Example**:
```python
@pytest.mark.asyncio
async def test_create_item_success(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/items",
        headers=auth_headers,
        json={"name": "Test Item"}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Item"
```

---

**Need help?** Check the comprehensive guides in `docs/testing/`
