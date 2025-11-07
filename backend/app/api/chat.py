"""
Chat API endpoints with context injection support.

Handles streaming chat with RAG and scratchpad context injection.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI
from uuid import UUID
import json
import time
from typing import AsyncGenerator

from app.dependencies import get_db, get_current_active_user
from app.models.database import (
    Conversation,
    Message as DBMessage,
    MessageRole,
    ScratchpadEntry,
    ScratchpadEntryType,
    KnowledgePool,
    User,
)
from app.models.schemas import ChatRequest
from app.services.rag_service import RAGService
from app.services.rag.text_splitter import TokenAwareSplitter
from app.services.title_service import TitleGenerationService
from app.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize clients
client = AsyncOpenAI(api_key=settings.openai_api_key)
rag_service = RAGService(text_splitter=TokenAwareSplitter())
title_service = TitleGenerationService(openai_client=client)


async def get_scratchpad_context(db: AsyncSession, user_id: UUID) -> str:
    """
    Retrieve and format scratchpad content for context injection.

    Returns a formatted string of todos, notes, and journal entries.
    """
    # Fetch all scratchpad entries
    result = await db.execute(
        select(ScratchpadEntry).where(ScratchpadEntry.user_id == user_id)
    )
    entries = result.scalars().all()

    if not entries:
        return ""

    context_parts = []

    # Format todos
    todos = [e for e in entries if e.entry_type == ScratchpadEntryType.TODO]
    if todos:
        todo_lines = ["**User's Current Todos:**"]
        for todo in todos:
            status = "✓" if todo.is_completed else "○"
            todo_lines.append(f"  {status} {todo.content}")
        context_parts.append("\n".join(todo_lines))

    # Format notes
    notes = [e for e in entries if e.entry_type == ScratchpadEntryType.NOTE]
    if notes:
        # Combine all notes (or take the latest)
        notes_content = notes[-1].content if notes else ""
        if notes_content.strip():
            context_parts.append(f"**User's Notes:**\n{notes_content}")

    # Format journal (today's entry)
    from datetime import date

    today = date.today()
    journals = [
        e
        for e in entries
        if e.entry_type == ScratchpadEntryType.JOURNAL
        and e.entry_date
        and e.entry_date.date() == today
    ]
    if journals:
        journal_content = journals[-1].content
        if journal_content.strip():
            context_parts.append(
                f"**User's Journal Entry (Today):**\n{journal_content}"
            )

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


async def get_rag_context(
    db: AsyncSession,
    user_id: UUID,
    query: str,
    knowledge_pool_ids: list[UUID] | None = None,
    limit: int = 5,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant documents from knowledge pools.

    Returns:
        - Formatted context string
        - List of source documents (for metadata)
    """
    logger.info(
        "rag_context_requested",
        query=query,
        user_id=str(user_id),
        pool_count=len(knowledge_pool_ids) if knowledge_pool_ids else None,
        limit=limit,
    )

    # Get collection names for the requested pools
    if knowledge_pool_ids:
        result = await db.execute(
            select(KnowledgePool).where(
                KnowledgePool.id.in_(knowledge_pool_ids),
                KnowledgePool.user_id == user_id,
            )
        )
    else:
        # Use all user's pools
        result = await db.execute(
            select(KnowledgePool).where(KnowledgePool.user_id == user_id)
        )

    pools = result.scalars().all()

    if not pools:
        logger.info("rag_context_no_pools", user_id=str(user_id))
        return "", []

    collection_names = [pool.collection_name for pool in pools]

    # Search across pools with reranking and deduplication
    results = await rag_service.search_multiple_pools_with_reranking(
        query=query,
        collection_names=collection_names,
        final_k=limit,
    )

    if not results:
        logger.info("rag_context_no_results", query=query, pools=collection_names)
        return "", []

    # Log retrieval results
    logger.debug(
        f"Retrieved {len(results)} results for query: {query[:50]}... from pools: {collection_names}"
    )

    # Format context
    context_parts = ["**Retrieved Knowledge:**"]
    sources = []

    for i, result in enumerate(results, 1):
        context_parts.append(f"\n[Source {i}: {result.filename}]\n{result.content}")
        sources.append(
            {
                "filename": result.filename,
                "score": result.score,
                "content_preview": result.content[:200] + "..."
                if len(result.content) > 200
                else result.content,
            }
        )

    logger.info(
        "rag_context_built",
        num_sources=len(sources),
        context_length=len("\n".join(context_parts)),
    )

    return "\n".join(context_parts), sources


async def build_system_message(
    db: AsyncSession,
    user_id: UUID,
    user_query: str,
    use_rag: bool = False,
    use_scratchpad: bool = False,
    knowledge_pool_ids: list[UUID] | None = None,
) -> tuple[str, dict]:
    """
    Build system message with context injection.

    Returns:
        - System message string
        - Metadata dict (sources, context info)
    """
    logger.info(
        "building_system_message",
        use_rag=use_rag,
        use_scratchpad=use_scratchpad,
        user_id=str(user_id),
    )

    base_message = "You are a helpful AI assistant. You provide clear, accurate, and thoughtful responses."

    context_parts = []
    metadata = {}

    # Add scratchpad context
    if use_scratchpad:
        scratchpad_context = await get_scratchpad_context(db, user_id)
        if scratchpad_context:
            context_parts.append(scratchpad_context)
            metadata["scratchpad_included"] = True
            logger.info("scratchpad_context_added", length=len(scratchpad_context))

    # Add RAG context
    if use_rag:
        rag_context, sources = await get_rag_context(
            db, user_id, user_query, knowledge_pool_ids
        )
        if rag_context:
            context_parts.append(rag_context)
            metadata["rag_sources"] = sources
            metadata["num_sources"] = len(sources)

    if context_parts:
        context_text = "\n".join(context_parts)
        system_message = f"""{base_message}

You have been provided with additional context to help answer the user's question. Use this context when relevant, but also apply your general knowledge.

{context_text}

Remember to cite sources when using information from the retrieved knowledge."""
    else:
        system_message = base_message

    logger.info(
        "system_message_built",
        message_length=len(system_message),
        has_rag=bool(metadata.get("rag_sources")),
        has_scratchpad=metadata.get("scratchpad_included", False),
    )

    return system_message, metadata


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Stream chat response with Server-Sent Events.

    Supports:
    - RAG context injection (when use_rag=True)
    - Scratchpad context injection (when use_scratchpad=True)
    - Conversation persistence
    - Message history

    Requires authentication via JWT token.
    """
    user_id = current_user.id

    logger.info(
        "chat_request_received",
        user_id=str(user_id),
        conversation_id=str(request.conversation_id)
        if request.conversation_id
        else None,
        use_rag=request.use_rag,
        use_scratchpad=request.use_scratchpad,
    )

    async def generate() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        try:
            # Get or create conversation
            conversation_id = request.conversation_id
            conversation = None

            if conversation_id:
                # Load existing conversation
                result = await db.execute(
                    select(Conversation).where(
                        and_(
                            Conversation.id == conversation_id,
                            Conversation.user_id == user_id,
                        )
                    )
                )
                conversation = result.scalar_one_or_none()

                if not conversation:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )
            else:
                # Create new conversation
                conversation = Conversation(
                    user_id=user_id,
                    use_rag=request.use_rag,
                    use_scratchpad=request.use_scratchpad,
                )
                db.add(conversation)
                await db.commit()
                await db.refresh(conversation)
                conversation_id = conversation.id

            # Get conversation history
            result = await db.execute(
                select(DBMessage)
                .where(DBMessage.conversation_id == conversation_id)
                .order_by(DBMessage.created_at.desc())
                .limit(20)  # Keep last 20 messages for context
            )
            history_messages = result.scalars().all()[
                ::-1
            ]  # Reverse to chronological order

            # Extract user's last message
            user_message = request.messages[-1] if request.messages else None
            if not user_message or user_message.role != "user":
                yield f"data: {json.dumps({'type': 'error', 'error': 'Last message must be from user'})}\n\n"
                return

            user_query = user_message.content

            # Build system message with context
            system_message, context_metadata = await build_system_message(
                db=db,
                user_id=user_id,
                user_query=user_query,
                use_rag=request.use_rag,
                use_scratchpad=request.use_scratchpad,
                knowledge_pool_ids=request.knowledge_pool_ids,
            )

            # Save user message to database
            user_db_message = DBMessage(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_query,
            )
            db.add(user_db_message)
            await db.commit()

            # Build message list for OpenAI
            messages = [{"role": "system", "content": system_message}]

            # Add conversation history
            for msg in history_messages:
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

            # Add current user message
            messages.append(
                {
                    "role": "user",
                    "content": user_query,
                }
            )

            # Stream response from OpenAI
            model = settings.default_llm_model

            logger.info(
                "llm_request_starting",
                model=model,
                message_count=len(messages),
                system_message_length=len(messages[0]["content"]),
            )

            llm_start_time = time.perf_counter()

            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.temperature,
                max_completion_tokens=settings.max_tokens,
                stream=True,
                stream_options={"include_usage": True},  # Request token usage
            )

            # Collect full response
            full_response = ""
            usage_data = None

            # Send conversation ID first
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': str(conversation_id)})}\n\n"

            # Send context metadata
            if context_metadata:
                yield f"data: {json.dumps({'type': 'metadata', 'metadata': context_metadata})}\n\n"

            # Stream content
            async for chunk in stream:
                # Capture usage data from final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage

                # Check if choices exist and have content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content = delta.content
                        full_response += content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

            llm_duration_ms = (time.perf_counter() - llm_start_time) * 1000

            # Log LLM call with token usage
            if usage_data:
                logger.info(
                    f"LLM call completed - Model: {model}, "
                    f"Prompt tokens: {usage_data.prompt_tokens}, "
                    f"Completion tokens: {usage_data.completion_tokens}, "
                    f"Total tokens: {usage_data.total_tokens}, "
                    f"Duration: {llm_duration_ms:.2f}ms"
                )
            else:
                logger.warning(f"LLM usage data unavailable for model: {model}")

            # Save assistant message to database
            assistant_db_message = DBMessage(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                extra_metadata=context_metadata if context_metadata else None,
            )
            db.add(assistant_db_message)

            # Update conversation title if it's the first exchange
            # Check if title is None, 'None', or empty, and this is the first exchange (no history before this message)
            if (
                not conversation.title
                or conversation.title.strip() == ""
                or conversation.title == "None"
            ) and len(history_messages) == 0:
                # Generate a concise title from the conversation using the title service
                try:
                    generated_title = await title_service.generate_title(
                        user_message=user_query,
                        assistant_response=full_response,
                        max_length=60,
                        style="concise"
                    )
                    conversation.title = generated_title
                    logger.info(f"Generated title for conversation {conversation_id}: {generated_title}")
                except Exception as e:
                    logger.error(f"Error generating title: {e}")
                    # The title service has its own fallback, so this shouldn't happen
                    # But just in case, provide a final fallback
                    conversation.title = user_query[:50] + (
                        "..." if len(user_query) > 50 else ""
                    )

            await db.commit()

            # Calculate total duration
            total_duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Chat stream completed - Conversation: {conversation_id}, "
                f"Duration: {total_duration_ms:.2f}ms, "
                f"Response length: {len(full_response)} chars"
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conversation_id)})}\n\n"

        except HTTPException as e:
            logger.error(
                f"Chat HTTP error - Status: {e.status_code}, Detail: {e.detail}"
            )
            yield f"data: {json.dumps({'type': 'error', 'error': e.detail})}\n\n"
        except Exception as e:
            logger.error(
                f"Chat stream error - Type: {type(e).__name__}, Error: {str(e)}"
            )
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
