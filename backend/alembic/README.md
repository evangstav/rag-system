# Database Migrations

This directory contains Alembic database migrations for the RAG chat system.

## Prerequisites

- PostgreSQL running (via `docker-compose up -d postgres`)
- Backend dependencies installed (`uv pip install -e .` from backend directory)
- Database URL configured in `.env` file

## Common Commands

### Create a new migration (auto-generate from models)

```bash
# From backend/ directory
alembic revision --autogenerate -m "description of changes"
```

### Apply migrations (upgrade to latest)

```bash
alembic upgrade head
```

### Rollback one migration

```bash
alembic downgrade -1
```

### View current migration status

```bash
alembic current
```

### View migration history

```bash
alembic history --verbose
```

### Downgrade to a specific revision

```bash
alembic downgrade <revision_id>
```

## Initial Setup

To create the initial database schema:

```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Create initial migration (already done)
alembic revision --autogenerate -m "initial schema"

# 3. Apply migration
alembic upgrade head
```

## How It Works

1. **Models** - SQLAlchemy models defined in `app/models/database.py`
2. **Auto-detect** - Alembic compares models to current DB schema
3. **Generate** - Creates migration script in `alembic/versions/`
4. **Review** - Always review the generated migration before applying
5. **Apply** - Run `alembic upgrade head` to apply changes

## Environment Configuration

Alembic uses the `DATABASE_URL` from `.env`:

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/ragchat
```

The `env.py` file automatically converts this to `postgresql+asyncpg://` for async compatibility.

## Troubleshooting

### "Target database is not up to date"

```bash
# Check current version
alembic current

# Upgrade to latest
alembic upgrade head
```

### "Can't locate revision identified by 'xxxxx'"

```bash
# Stamp the database with a specific revision
alembic stamp head
```

### Reset database (CAUTION: deletes all data)

```bash
# Drop all tables
alembic downgrade base

# Recreate from scratch
alembic upgrade head
```
