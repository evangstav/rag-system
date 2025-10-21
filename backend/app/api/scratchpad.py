"""
Scratchpad API endpoints.

Handles user scratchpad entries (todos, notes, journal).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, date
from uuid import UUID
import uuid

from app.dependencies import get_db
from app.models.database import ScratchpadEntry, ScratchpadEntryType
from app.models.schemas import ScratchpadData, TodoItem

router = APIRouter()

# TODO: Replace with actual user authentication
# For now, use a hardcoded default user ID
DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


@router.get("/", response_model=ScratchpadData)
async def get_scratchpad(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Get current scratchpad data for the authenticated user.

    Returns todos, notes, and today's journal entry.
    """
    # Fetch all scratchpad entries for the user
    result = await db.execute(
        select(ScratchpadEntry).where(ScratchpadEntry.user_id == user_id)
    )
    entries = result.scalars().all()

    # Separate entries by type
    todos = []
    notes_content = ""
    journal_content = ""

    today = date.today()

    for entry in entries:
        if entry.entry_type == ScratchpadEntryType.TODO:
            todos.append(
                TodoItem(
                    id=str(entry.id),
                    text=entry.content,
                    completed=entry.is_completed,
                )
            )
        elif entry.entry_type == ScratchpadEntryType.NOTE:
            # Combine all notes (or use the latest one)
            notes_content = entry.content
        elif entry.entry_type == ScratchpadEntryType.JOURNAL:
            # Only return today's journal
            if entry.entry_date and entry.entry_date.date() == today:
                journal_content = entry.content

    return ScratchpadData(
        todos=todos,
        notes=notes_content,
        journal=journal_content,
    )


@router.post("/")
async def save_scratchpad(
    data: ScratchpadData,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = DEFAULT_USER_ID,
):
    """
    Save scratchpad data for the authenticated user.

    Updates todos, notes, and journal entry.
    """
    today = datetime.now()

    # 1. Handle todos
    # Get existing todos
    result = await db.execute(
        select(ScratchpadEntry).where(
            and_(
                ScratchpadEntry.user_id == user_id,
                ScratchpadEntry.entry_type == ScratchpadEntryType.TODO,
            )
        )
    )
    existing_todos = {str(entry.id): entry for entry in result.scalars().all()}

    # Update or create todos
    todo_ids_in_request = set()
    for todo in data.todos:
        todo_ids_in_request.add(todo.id)

        if todo.id in existing_todos:
            # Update existing todo
            entry = existing_todos[todo.id]
            entry.content = todo.text
            entry.is_completed = todo.completed
        else:
            # Create new todo
            new_todo = ScratchpadEntry(
                id=UUID(todo.id) if todo.id else uuid.uuid4(),
                user_id=user_id,
                entry_type=ScratchpadEntryType.TODO,
                content=todo.text,
                is_completed=todo.completed,
            )
            db.add(new_todo)

    # Delete todos that are no longer in the request
    for todo_id, entry in existing_todos.items():
        if todo_id not in todo_ids_in_request:
            await db.delete(entry)

    # 2. Handle notes (single entry per user)
    result = await db.execute(
        select(ScratchpadEntry).where(
            and_(
                ScratchpadEntry.user_id == user_id,
                ScratchpadEntry.entry_type == ScratchpadEntryType.NOTE,
            )
        )
    )
    notes_entry = result.scalar_one_or_none()

    if data.notes:
        if notes_entry:
            notes_entry.content = data.notes
        else:
            notes_entry = ScratchpadEntry(
                user_id=user_id,
                entry_type=ScratchpadEntryType.NOTE,
                content=data.notes,
            )
            db.add(notes_entry)
    elif notes_entry:
        # Delete notes if empty
        await db.delete(notes_entry)

    # 3. Handle journal (today's entry)
    result = await db.execute(
        select(ScratchpadEntry).where(
            and_(
                ScratchpadEntry.user_id == user_id,
                ScratchpadEntry.entry_type == ScratchpadEntryType.JOURNAL,
                ScratchpadEntry.entry_date >= today.replace(hour=0, minute=0, second=0),
            )
        )
    )
    journal_entry = result.scalar_one_or_none()

    if data.journal:
        if journal_entry:
            journal_entry.content = data.journal
        else:
            journal_entry = ScratchpadEntry(
                user_id=user_id,
                entry_type=ScratchpadEntryType.JOURNAL,
                content=data.journal,
                entry_date=today,
            )
            db.add(journal_entry)
    elif journal_entry:
        # Delete journal if empty
        await db.delete(journal_entry)

    # Commit transaction
    await db.commit()

    return {"status": "saved"}
