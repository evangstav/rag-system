# Database Migrations Guide

This directory contains Alembic database migrations for the RAG Chat System.

## Quick Start

### 1. Setup

Ensure your `.env` file has the correct database URL:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat
```

### 2. Apply All Migrations

```bash
# Using the helper script (recommended)
python migrate.py upgrade

# Or directly with alembic
python -m alembic upgrade head
```

### 3. Check Current Version

```bash
python migrate.py current
```

## Common Tasks

### Apply Migrations

```bash
# Apply all pending migrations
python migrate.py upgrade

# Apply up to a specific version
python -m alembic upgrade <revision_id>
```

### Revert Migrations

```bash
# Revert last migration
python migrate.py downgrade

# Revert all migrations
python migrate.py reset

# Revert to specific version
python -m alembic downgrade <revision_id>
```

### View Migration History

```bash
# View all migrations
python migrate.py history

# View current version
python migrate.py current
```

### Create New Migration

```bash
# Auto-generate migration from model changes
python migrate.py create "add_new_column"

# Create empty migration (manual)
python -m alembic revision -m "custom_migration"
```

## Migration Files

Migrations are stored in `alembic/versions/` with timestamped filenames.

### Current Migrations

**001_initial_schema.py** - Initial database schema
- Creates all core tables: users, conversations, messages, scratchpad_entries, knowledge_pools, documents, user_memories
- Creates enum types for message_role, scratchpad_entry_type, document_status
- Creates indexes for performance

## Directory Structure

```
alembic/
├── versions/              # Migration files
│   └── 001_initial_schema.py
├── env.py                 # Alembic environment configuration
├── script.py.mako         # Template for new migrations
├── README.md              # This file
└── MIGRATIONS.md          # This guide
```

## How Migrations Work

1. **Upgrade**: Runs the `upgrade()` function in each migration file
2. **Downgrade**: Runs the `downgrade()` function to revert changes
3. **Version Tracking**: Alembic creates an `alembic_version` table to track the current migration

## Best Practices

### Before Creating Migrations

1. **Test model changes** - Ensure your SQLAlchemy models are correct
2. **Review auto-generated migrations** - Alembic may not catch everything
3. **Test both upgrade and downgrade** - Ensure migrations are reversible

### Creating Good Migrations

```python
def upgrade() -> None:
    """Clear description of what this migration does."""
    # Add table, column, index, etc.
    op.add_column('users', sa.Column('new_field', sa.String(100)))

def downgrade() -> None:
    """Reverse the changes."""
    # Remove what was added
    op.drop_column('users', 'new_field')
```

### Migration Naming

Use descriptive names:
- ✅ `add_user_timezone_column`
- ✅ `create_analytics_tables`
- ✅ `add_index_messages_created_at`
- ❌ `update`
- ❌ `fix_stuff`
- ❌ `migration_2`

## Troubleshooting

### "Target database is not up to date"

Your database is not at the latest migration. Run:

```bash
python migrate.py current
python migrate.py upgrade
```

### "Can't locate revision"

Migration files are missing or corrupted. Check `alembic/versions/` directory.

### "Concurrent transaction" errors

Another process is running migrations. Wait for it to complete or check for stuck processes.

### "Table already exists"

Your database has tables but no alembic_version tracking. Options:

1. **Fresh start** (development only):
   ```bash
   # Drop all tables and reapply migrations
   python migrate.py reset
   python migrate.py upgrade
   ```

2. **Mark as current** (if schema matches):
   ```bash
   python -m alembic stamp head
   ```

### Environment Variable Issues

Ensure `.env` file is in the `backend/` directory and contains:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat
```

## Database Connection

Alembic uses async PostgreSQL with asyncpg driver. The connection URL is automatically:
- Loaded from `DATABASE_URL` environment variable
- Converted from `postgres://` or `postgresql://` to `postgresql+asyncpg://`

## Advanced Usage

### Multiple Databases

For different environments:

```bash
# Development
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat_dev python migrate.py upgrade

# Testing
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat_test python migrate.py upgrade
```

### Offline SQL Generation

Generate SQL without running migrations:

```bash
python -m alembic upgrade head --sql > migrations.sql
```

### Branch Merging

If multiple migrations are created in parallel:

```bash
python -m alembic merge <rev1> <rev2> -m "merge_branches"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run database migrations
  env:
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
  run: |
    cd backend
    python migrate.py upgrade
```

### Pre-deployment Checklist

- [ ] Review all pending migrations
- [ ] Test migrations on staging database
- [ ] Backup production database
- [ ] Run migrations during maintenance window
- [ ] Verify application works after migration

## Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
