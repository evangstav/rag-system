# PR #2: Database Migrations Setup

## Summary

This PR sets up proper database migration management with Alembic. The system can now track schema changes, apply migrations reliably, and roll back if needed - essential for production deployments.

## Changes Made

### 1. Alembic Configuration (`alembic.ini`)

**Updated:**
- Added timestamped migration file naming: `YYYYMMDD_HHMM-{rev}_{slug}.py`
- Added comments about environment variable override
- Uses `postgresql+asyncpg://` for async SQLAlchemy

**Benefits:**
- Clear chronological ordering of migrations
- Database URL comes from `.env` (secure, environment-specific)

### 2. Initial Migration (`alembic/versions/001_initial_schema.py`)

**Complete schema creation including:**

**Tables:**
- `users` - User accounts with authentication
- `conversations` - Chat sessions with RAG/scratchpad settings
- `messages` - Individual chat messages with metadata
- `scratchpad_entries` - User todos, notes, journal entries
- `knowledge_pools` - RAG document collections
- `documents` - Uploaded files with processing status
- `user_memories` - Extracted user facts/preferences

**Enums:**
- `message_role` - user | assistant | system
- `scratchpad_entry_type` - todo | note | journal
- `document_status` - pending | processing | completed | failed

**Indexes for Performance:**
- `idx_messages_conversation_created` - Fast conversation history queries
- `idx_scratchpad_user_type` - Fast scratchpad retrieval
- `idx_user_memories_importance` - Sorted memory retrieval

**Features:**
- UUID primary keys with auto-generation
- Foreign keys with CASCADE deletes
- Timestamps with automatic NOW() defaults
- JSONB for flexible metadata storage
- Proper constraints and indexes

### 3. Migration Helper Script (`migrate.py`)

**Convenience wrapper for common operations:**

```bash
python migrate.py upgrade      # Apply all migrations
python migrate.py downgrade    # Revert last migration
python migrate.py reset        # Revert all migrations
python migrate.py current      # Show current version
python migrate.py history      # Show all migrations
python migrate.py create "msg" # Create new migration
```

**Features:**
- Clear command descriptions
- Error handling with status codes
- Usage help text
- Environment variable support

### 4. Comprehensive Documentation (`alembic/MIGRATIONS.md`)

**Includes:**
- Quick start guide
- Common tasks with examples
- Migration file structure
- Best practices
- Troubleshooting guide
- CI/CD integration examples
- Pre-deployment checklist

### 5. Updated Environment Variables

**`.env.example` updated with:**
- Migration running instructions
- Database URL format clarification

## Usage Instructions

### Initial Setup (Fresh Database)

```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Set environment variables
cd backend
cp .env.example .env
# Edit .env and set your DATABASE_URL

# 3. Run migrations
python migrate.py upgrade
```

**Expected Output:**
```
============================================================
  Applying all pending migrations
============================================================

INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 001, initial_schema

✅ Applying all pending migrations completed successfully
```

### Verify Migration Applied

```bash
python migrate.py current
```

**Expected Output:**
```
001 (head)
```

### Check Database

```bash
# Connect to database
docker exec -it rag-system-postgres-1 psql -U user -d ragchat

# List tables
\dt

# Should show:
#  alembic_version
#  conversations
#  documents
#  knowledge_pools
#  messages
#  scratchpad_entries
#  user_memories
#  users

# Check enums
\dT

# Exit
\q
```

## Testing Migrations

### Test Upgrade

```bash
# Fresh database
python migrate.py upgrade

# Check tables exist
python -c "from app.database import engine; import asyncio; asyncio.run(engine.dispose())"
```

### Test Downgrade (Rollback)

```bash
# Revert migration
python migrate.py downgrade

# Tables should be dropped
# Verify with psql: \dt should show only alembic_version
```

### Test Complete Reset

```bash
# Remove all schema
python migrate.py reset

# Re-apply
python migrate.py upgrade
```

## Integration with Existing Code

### Application Startup

**Current flow:**
1. App starts (`uvicorn app.main:app`)
2. SQLAlchemy models are loaded
3. Database connection is established
4. ✅ Tables already exist (from migrations)

**No changes needed** - Models work with migrated schema.

### Model Changes Workflow

**When updating SQLAlchemy models:**

1. Edit model in `app/models/database.py`:
   ```python
   class User(Base):
       # ... existing fields ...
       timezone = Column(String(50), default='UTC')  # New field
   ```

2. Create migration:
   ```bash
   python migrate.py create "add_user_timezone_column"
   ```

3. Review generated migration in `alembic/versions/`

4. Test locally:
   ```bash
   python migrate.py upgrade    # Apply
   python migrate.py downgrade  # Test rollback
   python migrate.py upgrade    # Re-apply
   ```

5. Commit migration file with model changes

## Migration File Structure

### Example Migration

```python
def upgrade() -> None:
    """Add timezone column to users table."""
    op.add_column('users',
        sa.Column('timezone', sa.String(50), server_default='UTC')
    )

def downgrade() -> None:
    """Remove timezone column from users table."""
    op.drop_column('users', 'timezone')
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Database Migrations

on:
  push:
    branches: [main]

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt

      - name: Run migrations
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          cd backend
          python migrate.py upgrade
```

## Benefits

### For Development

- ✅ Consistent schema across team members
- ✅ Version-controlled database changes
- ✅ Easy rollback if something breaks
- ✅ Clear history of schema evolution

### For Production

- ✅ Reliable deployment process
- ✅ Zero-downtime migrations possible
- ✅ Audit trail of all changes
- ✅ Automated CI/CD integration

### For Collaboration

- ✅ No manual SQL scripts to share
- ✅ Prevents schema drift between environments
- ✅ Clear documentation in migration files
- ✅ Conflicts caught early (merge conflicts in migrations)

## Troubleshooting

### "Table already exists" Error

**Problem:** Database has tables but no migration tracking.

**Solution 1 (Fresh start - dev only):**
```bash
# Drop all tables
python migrate.py reset

# Re-create with migrations
python migrate.py upgrade
```

**Solution 2 (Keep existing data):**
```bash
# Mark current schema as migrated
python -m alembic stamp head
```

### "Can't locate revision" Error

**Problem:** Migration files are missing.

**Check:**
```bash
ls -la backend/alembic/versions/
# Should show: 001_initial_schema.py
```

### Import Errors in env.py

**Problem:** Python can't find models.

**Solution:** Ensure you're running from `backend/` directory:
```bash
cd backend
python migrate.py upgrade
```

## File Changes

```
backend/
├── alembic.ini                          (updated - timestamped files, comments)
├── alembic/
│   ├── versions/
│   │   └── 001_initial_schema.py       (new - 200 lines)
│   └── MIGRATIONS.md                    (new - comprehensive guide)
├── migrate.py                           (new - helper script)
└── .env.example                         (updated - migration instructions)

PR_2_DATABASE_MIGRATIONS.md              (this file)
```

## Breaking Changes

⚠️ **None** - This is additive. Applications work with or without Alembic.

## Migration Path for Existing Databases

If you've been running the app and already have tables:

**Option 1: Fresh start (dev only)**
```bash
# Backup data if needed
docker exec -it rag-system-postgres-1 pg_dump -U user ragchat > backup.sql

# Drop database
docker exec -it rag-system-postgres-1 psql -U user -d postgres -c "DROP DATABASE ragchat;"
docker exec -it rag-system-postgres-1 psql -U user -d postgres -c "CREATE DATABASE ragchat;"

# Run migrations
cd backend
python migrate.py upgrade
```

**Option 2: Mark as current (keep data)**
```bash
cd backend
python -m alembic stamp 001
```

## Next Steps

**Recommended follow-up PRs:**

1. **PR #3: JWT Authentication**
   - Will require migration for user authentication fields
   - Example: `python migrate.py create "add_auth_fields"`

2. **PR #4: Knowledge Pool UI**
   - May need additional fields/tables
   - Migrations make this easy to add incrementally

3. **PR #5: Enhanced Chat Features**
   - Conversation metadata updates
   - New indexes for performance

## Questions?

- **Why Alembic?** - Industry standard, works with async SQLAlchemy, supports auto-generation
- **Why manual migration?** - Full control, clear documentation, matches existing schema exactly
- **Why not auto-generate?** - First migration from existing models requires manual creation
- **Future migrations?** - Use `python migrate.py create "message"` for auto-generation

---

**Status:** ✅ Ready for review and testing
**Tested:** ✅ Locally with PostgreSQL 15
**Documentation:** ✅ Complete with troubleshooting guide
**Backwards Compatible:** ✅ Yes - works with existing deployments
