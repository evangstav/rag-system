"""
Conversation management API endpoints.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.dependencies import get_current_user, get_db
from app.models.database import Conversation, Message, MessageRole, User
from app.models.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from app.services.title_service import TitleGenerationService

router = APIRouter()

# Initialize title service
title_service = TitleGenerationService()


@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
):
    """
    List all conversations for the current user.
    Ordered by most recently updated first.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
        .offset(offset)
    )
    conversations = result.scalars().all()
    return conversations


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a new conversation for the current user.
    """
    new_conversation = Conversation(
        user_id=user.id,
        title=conversation_data.title,
        use_rag=conversation_data.use_rag,
        use_scratchpad=conversation_data.use_scratchpad,
    )

    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)

    return new_conversation


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get a specific conversation by ID.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0,
):
    """
    Get all messages for a specific conversation.
    Ordered by creation time (oldest first).
    """
    # First verify the conversation belongs to the user
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()

    return messages


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    updates: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Update a conversation (e.g., change title, toggle RAG/scratchpad).
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Update fields if provided
    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(conversation, field, value)

    await db.commit()
    await db.refresh(conversation)

    return conversation


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Delete a conversation and all its messages.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()

    return {"message": "Conversation deleted successfully", "id": str(conversation_id)}


@router.post("/{conversation_id}/regenerate-title", response_model=ConversationResponse)
async def regenerate_conversation_title(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Regenerate the title for a conversation based on its first message exchange.

    This endpoint allows users to manually trigger title generation for conversations
    that may have generic titles or no title at all.
    """
    # Fetch the conversation
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Fetch the first two messages (user and assistant)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .limit(2)
    )
    messages = result.scalars().all()

    if len(messages) < 2:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate title: conversation needs at least one exchange (user message + assistant response)"
        )

    # Ensure we have a user message and assistant response
    if messages[0].role != MessageRole.USER or messages[1].role != MessageRole.ASSISTANT:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate title: first two messages must be user question and assistant response"
        )

    user_message = messages[0].content
    assistant_response = messages[1].content

    try:
        # Generate new title
        new_title = await title_service.generate_title(
            user_message=user_message,
            assistant_response=assistant_response,
            max_length=100,  # Allow longer titles for manual regeneration
            style="descriptive"  # Use descriptive style for manual regeneration
        )

        # Update conversation
        conversation.title = new_title
        await db.commit()
        await db.refresh(conversation)

        return conversation

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate title: {str(e)}"
        )
