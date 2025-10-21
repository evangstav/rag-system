#!/usr/bin/env python3
"""
Database migration helper script.

This script provides convenient commands for managing Alembic database migrations.
Uses DATABASE_URL from environment (.env file).
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        sys.exit(e.returncode)


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("""
Database Migration Helper
=========================

Usage:
    python migrate.py <command> [args]

Commands:
    upgrade     Apply all pending migrations (alembic upgrade head)
    downgrade   Revert last migration (alembic downgrade -1)
    reset       Revert all migrations (alembic downgrade base)
    current     Show current migration version (alembic current)
    history     Show migration history (alembic history)
    create      Create new migration (alembic revision --autogenerate -m "message")

Examples:
    python migrate.py upgrade              # Apply all pending migrations
    python migrate.py downgrade            # Revert last migration
    python migrate.py reset                # Revert all migrations
    python migrate.py current              # Show current version
    python migrate.py history              # Show all migrations
    python migrate.py create "add_column"  # Create new migration

Environment:
    Ensure DATABASE_URL is set in your .env file:
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragchat
        """)
        sys.exit(1)

    command = sys.argv[1].lower()

    # Change to backend directory (where alembic.ini is located)
    backend_dir = Path(__file__).parent

    commands = {
        'upgrade': (['python', '-m', 'alembic', 'upgrade', 'head'], "Applying all pending migrations"),
        'downgrade': (['python', '-m', 'alembic', 'downgrade', '-1'], "Reverting last migration"),
        'reset': (['python', '-m', 'alembic', 'downgrade', 'base'], "Reverting all migrations"),
        'current': (['python', '-m', 'alembic', 'current'], "Showing current migration version"),
        'history': (['python', '-m', 'alembic', 'history', '--verbose'], "Showing migration history"),
    }

    if command == 'create':
        if len(sys.argv) < 3:
            print("❌ Error: Migration message required")
            print("Usage: python migrate.py create \"migration message\"")
            sys.exit(1)

        message = sys.argv[2]
        cmd = ['python', '-m', 'alembic', 'revision', '--autogenerate', '-m', message]
        description = f"Creating new migration: {message}"
        return run_command(cmd, description)

    elif command in commands:
        cmd, description = commands[command]
        return run_command(cmd, description)

    else:
        print(f"❌ Unknown command: {command}")
        print("Run without arguments to see usage")
        sys.exit(1)


if __name__ == '__main__':
    main()
