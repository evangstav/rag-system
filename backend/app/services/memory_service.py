"""
User Memory Service for extracting, storing, and retrieving user preferences and context.

Implements memory extraction from conversations and journal entries, with multi-factor
scoring (semantic + recency + importance) for retrieval.
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy import select, and_, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database import (
    UserMemory,
    Message,
    ScratchpadEntry,
    ScratchpadEntryType,
)
from app.services.rag.embeddings import OpenAIEmbeddings
from app.services.rag.vector_store import QdrantVectorStore


# Memory extraction prompt
MEMORY_EXTRACTION_PROMPT = """You are a memory extraction assistant. Analyze the conversation and extract important, persistent information about the user that should be remembered for future conversations.

Extract the following types of information:
1. **Preferences:** Communication style, likes/dislikes, preferred tools, formats, workflows
2. **Facts:** Personal details, professional info, role, company, ongoing projects
3. **Relationships:** People, organizations, locations frequently mentioned
4. **Goals:** Objectives, aspirations, tasks they're working on, learning interests
5. **Context:** Domain expertise, technical background, areas of interest, experience level

For each memory, provide:
- content: Clear, standalone statement in third person (e.g., "User prefers concise technical responses")
- category: One of [preference, fact, relationship, goal, context]
- importance: 0.0-1.0 (0.8+ for critical info, 0.5-0.7 for useful, 0.3-0.4 for minor details)

Only extract information that:
- Would be useful in future conversations
- Is persistent (not just relevant to current discussion)
- Is clearly stated or strongly implied
- Represents lasting user characteristics or circumstances

DO NOT extract:
- Ephemeral information (one-time requests, specific dates unless recurring)
- Conversation-specific details
- Generic statements that apply to everyone
- Sensitive personal information (passwords, API keys, financial data)

Conversation History:
{conversation_history}

Current Memories (avoid duplicates):
{existing_memories}

Extract new or updated memories in JSON format. If no new persistent information is found, return an empty array.

Response format:
{{
  "memories": [
    {{
      "content": "User is a Python developer working on a RAG system with FastAPI and Next.js",
      "category": "context",
      "importance": 0.8
    }},
    {{
      "content": "User prefers detailed technical explanations with code examples",
      "category": "preference",
      "importance": 0.7
    }}
  ]
}}
"""


# Journal extraction prompt (optimized for journal entries)
JOURNAL_EXTRACTION_PROMPT = """You are a memory extraction assistant analyzing journal entries. Extract persistent information about the user's goals, preferences, context, and important life details.

Focus on:
1. **Long-term goals and aspirations** (career, learning, projects)
2. **Ongoing projects or initiatives** mentioned repeatedly
3. **Preferences and patterns** in how they work or think
4. **Important context** about their role, expertise, or circumstances
5. **Key relationships** or collaborations mentioned

DO NOT extract:
- Day-specific events or activities (unless part of recurring pattern)
- Temporary moods or feelings
- One-time occurrences
- Overly personal or sensitive information

Journal Entries:
{journal_content}

Current Memories (avoid duplicates):
{existing_memories}

Extract persistent, long-term memories in JSON format:
{{
  "memories": [
    {{
      "content": "User is learning about AI memory systems and implementing one for their RAG project",
      "category": "goal",
      "importance": 0.8
    }}
  ]
}}
"""


class MemoryService:
    """
    Service for managing user memories.

    Handles:
    - Extraction from conversations and journal entries
    - Storage with embeddings in Qdrant + PostgreSQL
    - Multi-factor retrieval (semantic + recency + importance)
    - Deduplication and consolidation
    """

    def __init__(
        self,
        db: AsyncSession,
        embeddings: Optional[OpenAIEmbeddings] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        llm_client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize memory service.

        Args:
            db: Database session
            embeddings: Embedding provider (defaults to OpenAI)
            vector_store: Vector store (defaults to Qdrant)
            llm_client: OpenAI client for extraction (defaults to new client)
        """
        self.db = db
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vector_store = vector_store or QdrantVectorStore()
        self.llm_client = llm_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.collection_name = settings.memory_collection_name

    async def ensure_collection(self):
        """Ensure the memory collection exists in Qdrant."""
        try:
            await self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=self.embeddings.dimensions,
                distance="cosine",
            )
        except Exception:
            # Collection likely already exists
            pass

    async def extract_memories_from_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit_messages: int = 20,
    ) -> List[UserMemory]:
        """
        Extract memories from a conversation using LLM.

        Args:
            conversation_id: Conversation to extract from
            user_id: User ID
            limit_messages: Number of recent messages to analyze (default 20)

        Returns:
            List of newly created UserMemory objects
        """
        # Get recent conversation messages
        messages = await self._fetch_conversation_messages(
            conversation_id, limit_messages
        )
        if not messages:
            return []

        # Format and extract
        conversation_text = self._format_messages(messages)
        existing_text = await self._get_existing_memories_text(user_id)

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation_history=conversation_text,
            existing_memories=existing_text,
        )

        extracted = await self._call_llm_for_extraction(prompt, "conversation")
        if not extracted:
            return []

        print(f"Extracted memories: \n {extracted}")

        # Store extracted memories
        return await self._store_extracted_memories(
            user_id=user_id,
            extracted_data=extracted,
            source_conversation_id=conversation_id,
        )

    async def extract_memories_from_journal(
        self,
        user_id: UUID,
        entry_date: Optional[datetime] = None,
        days_back: int = 7,
    ) -> List[UserMemory]:
        """
        Extract memories from journal entries.

        Args:
            user_id: User ID
            entry_date: Specific date to extract from (None = recent entries)
            days_back: If entry_date is None, how many days back to analyze

        Returns:
            List of newly created UserMemory objects
        """
        # Fetch journal entries
        entries = await self._fetch_journal_entries(user_id, entry_date, days_back)
        if not entries:
            return []

        # Format entries
        journal_text = "\n\n".join(
            [
                f"[{entry.entry_date.strftime('%Y-%m-%d')}]\n{entry.content}"
                for entry in entries
            ]
        )

        # Get existing memories and extract
        existing_text = await self._get_existing_memories_text(user_id)

        prompt = JOURNAL_EXTRACTION_PROMPT.format(
            journal_content=journal_text,
            existing_memories=existing_text,
        )

        extracted = await self._call_llm_for_extraction(prompt, "journal")
        if not extracted:
            return []

        # Store extracted memories (no source_conversation_id for journal entries)
        return await self._store_extracted_memories(
            user_id=user_id,
            extracted_data=extracted,
            source_conversation_id=None,
        )

    async def add_memory(
        self,
        user_id: UUID,
        content: str,
        importance: float = 0.5,
        category: str = "context",
        source_conversation_id: Optional[UUID] = None,
    ) -> Optional[UserMemory]:
        """
        Add or update a memory.

        Checks for similar existing memories and updates if found,
        otherwise creates new memory.

        Args:
            user_id: User ID
            content: Memory content
            importance: Importance score (0.0-1.0)
            category: Memory category
            source_conversation_id: Source conversation (optional)

        Returns:
            UserMemory object (new or updated), or None if skipped
        """
        await self.ensure_collection()

        # Check for similar existing memory
        similar = await self._find_similar_memory(
            user_id, content, threshold=settings.memory_similarity_threshold
        )

        if similar:
            # Update existing memory
            similar.content = content
            similar.importance = max(similar.importance, importance)
            similar.updated_at = func.now()
            await self.db.commit()
            await self.db.refresh(similar)
            return similar

        # Generate embedding
        try:
            embedding = await self.embeddings.embed_text(content)
        except Exception as e:
            print(f"Failed to generate embedding for memory: {e}")
            return None

        # Store in Qdrant using direct client API with UUID
        # Generate a unique UUID for this memory point
        from uuid import uuid4
        qdrant_id = str(uuid4())

        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=qdrant_id,
                vector=embedding,
                payload={
                    "user_id": str(user_id),
                    "content": content,
                    "importance": importance,
                    "category": category,
                },
            )

            await self.vector_store.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
        except Exception as e:
            print(f"Failed to store memory in Qdrant: {e}")
            return None

        # Store in PostgreSQL
        memory = UserMemory(
            user_id=user_id,
            content=content,
            importance=importance,
            source_conversation_id=source_conversation_id,
            qdrant_id=qdrant_id,
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)

        return memory

    async def retrieve_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = 10,
    ) -> List[UserMemory]:
        """
        Retrieve relevant memories with multi-factor scoring.

        Combines:
        - Semantic similarity (vector search)
        - Recency (time-based decay)
        - Importance (extracted score)

        Args:
            user_id: User ID
            query: Query to search for relevant memories
            limit: Maximum number of memories to return

        Returns:
            List of UserMemory objects, sorted by combined score
        """
        await self.ensure_collection()

        # Generate query embedding
        try:
            query_embedding = await self.embeddings.embed_text(query)
        except Exception as e:
            print(f"Failed to generate query embedding: {e}")
            return []

        # Semantic search in Qdrant
        try:
            results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Get more for filtering
            )
        except Exception as e:
            print(f"Failed to search memories in Qdrant: {e}")
            return []

        # Filter to user's memories and get from DB
        qdrant_ids = [
            r.id
            for r in results
            if r.metadata and r.metadata.get("user_id") == str(user_id)
        ]

        if not qdrant_ids:
            return []

        stmt = select(UserMemory).where(
            and_(
                UserMemory.user_id == user_id,
                UserMemory.qdrant_id.in_(qdrant_ids),
            )
        )
        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        # Multi-factor scoring
        scored_memories = []
        now = datetime.utcnow()

        for memory in memories:
            # Get semantic score
            semantic_score = next(
                (r.score for r in results if r.id == memory.qdrant_id),
                0.0,
            )

            # Recency score (time decay)
            time_delta_hours = (now - memory.updated_at).total_seconds() / 3600
            recency_score = 1.0 / (1.0 + 0.01 * time_delta_hours)

            # Combined score
            final_score = (
                settings.memory_semantic_weight * semantic_score
                + settings.memory_recency_weight * recency_score
                + settings.memory_importance_weight * memory.importance
            )

            scored_memories.append((final_score, memory))

        # Sort by score and return top-k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_memories[:limit]]

    async def get_all_memories(
        self,
        user_id: UUID,
        limit: int = 100,
        category: Optional[str] = None,
    ) -> List[UserMemory]:
        """
        Get all memories for a user, ordered by importance.

        Args:
            user_id: User ID
            limit: Maximum number of memories
            category: Filter by category (optional)

        Returns:
            List of UserMemory objects
        """
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)

        if category:
            # Note: category is not in the current schema, but kept for future use
            pass

        stmt = stmt.order_by(
            UserMemory.importance.desc(),
            UserMemory.updated_at.desc(),
        ).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_memory(
        self,
        memory_id: UUID,
        user_id: UUID,
        content: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> Optional[UserMemory]:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID
            user_id: User ID (for authorization)
            content: New content (optional)
            importance: New importance (optional)

        Returns:
            Updated UserMemory object, or None if not found
        """
        stmt = select(UserMemory).where(
            and_(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id,
            )
        )
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()

        if not memory:
            return None

        # Update fields
        if content is not None:
            memory.content = content

            # Re-generate embedding and update in Qdrant
            try:
                embedding = await self.embeddings.embed_text(content)
                from qdrant_client.models import PointStruct

                point = PointStruct(
                    id=memory.qdrant_id,
                    vector=embedding,
                    payload={
                        "user_id": str(user_id),
                        "content": content,
                        "importance": memory.importance,
                    },
                )

                await self.vector_store.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
            except Exception as e:
                print(f"Failed to update memory in Qdrant: {e}")

        if importance is not None:
            memory.importance = importance

        # Explicitly set updated_at to current UTC time
        from datetime import datetime, timezone

        memory.updated_at = datetime.now(timezone.utc)

        # Commit changes
        await self.db.commit()

        # Force load all attributes by using execute with options
        from sqlalchemy.orm import selectinload, undefer

        stmt = (
            select(UserMemory).where(UserMemory.id == memory_id).options(undefer("*"))
        )
        result = await self.db.execute(stmt)
        refreshed_memory = result.scalar_one()

        return refreshed_memory

    async def delete_memory(self, memory_id: UUID, user_id: UUID) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: Memory ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted, False if not found
        """
        stmt = select(UserMemory).where(
            and_(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id,
            )
        )
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()

        if not memory:
            return False

        # Delete from Qdrant
        if memory.qdrant_id:
            try:
                from qdrant_client.models import PointIdsList

                await self.vector_store.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=[memory.qdrant_id]),
                )
            except Exception as e:
                print(f"Failed to delete memory from Qdrant: {e}")

        # Delete from DB
        await self.db.delete(memory)
        await self.db.commit()

        return True

    async def delete_all_memories(self, user_id: UUID) -> int:
        """
        Delete all memories for a user.

        Args:
            user_id: User ID

        Returns:
            Number of memories deleted
        """
        # Get all memory IDs
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)
        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        if not memories:
            return 0

        # Delete from Qdrant
        qdrant_ids = [m.qdrant_id for m in memories if m.qdrant_id]
        if qdrant_ids:
            try:
                from qdrant_client.models import PointIdsList

                await self.vector_store.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=qdrant_ids),
                )
            except Exception as e:
                print(f"Failed to delete memories from Qdrant: {e}")

        # Delete from DB
        count = len(memories)
        stmt = delete(UserMemory).where(UserMemory.user_id == user_id)
        await self.db.execute(stmt)
        await self.db.commit()

        return count

    def format_memories_for_context(
        self,
        memories: List[UserMemory],
        max_length: int = 1000,
    ) -> str:
        """
        Format memories into context string for LLM.

        Args:
            memories: List of memories
            max_length: Maximum character length

        Returns:
            Formatted string
        """
        if not memories:
            return "No relevant memories."

        memory_lines = []
        current_length = 0

        for memory in memories:
            line = f"- {memory.content}\n"

            if current_length + len(line) > max_length:
                break

            memory_lines.append(line)
            current_length += len(line)

        if not memory_lines:
            return "No relevant memories."

        return "".join(memory_lines).strip()

    async def _find_similar_memory(
        self,
        user_id: UUID,
        content: str,
        threshold: float = 0.85,
    ) -> Optional[UserMemory]:
        """
        Find similar existing memory.

        Args:
            user_id: User ID
            content: Memory content to check
            threshold: Similarity threshold

        Returns:
            Similar UserMemory if found, else None
        """
        try:
            # Generate embedding
            embedding = await self.embeddings.embed_text(content)

            # Search for similar
            results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=5,
            )

            # Find most similar for this user
            for result in results:
                if result.score >= threshold:
                    if result.metadata and result.metadata.get("user_id") == str(
                        user_id
                    ):
                        # Get from DB
                        stmt = select(UserMemory).where(
                            and_(
                                UserMemory.user_id == user_id,
                                UserMemory.qdrant_id == result.id,
                            )
                        )
                        db_result = await self.db.execute(stmt)
                        memory = db_result.scalar_one_or_none()
                        if memory:
                            return memory

        except Exception as e:
            print(f"Failed to find similar memory: {e}")

        return None

    async def _fetch_conversation_messages(
        self, conversation_id: UUID, limit: int
    ) -> List[Message]:
        """Fetch recent conversation messages in chronological order."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def _fetch_journal_entries(
        self,
        user_id: UUID,
        entry_date: Optional[datetime],
        days_back: int,
    ) -> List[ScratchpadEntry]:
        """Fetch journal entries based on date criteria."""
        stmt = select(ScratchpadEntry).where(
            and_(
                ScratchpadEntry.user_id == user_id,
                ScratchpadEntry.entry_type == ScratchpadEntryType.JOURNAL,
                ScratchpadEntry.entry_date.isnot(None),
            )
        )

        if entry_date:
            # Specific date range
            start_of_day = entry_date.replace(hour=0, minute=0, second=0)
            end_of_day = start_of_day + timedelta(days=1)
            stmt = stmt.where(
                and_(
                    ScratchpadEntry.entry_date >= start_of_day,
                    ScratchpadEntry.entry_date < end_of_day,
                )
            )
        else:
            # Recent entries
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            stmt = stmt.where(ScratchpadEntry.entry_date >= cutoff_date)

        stmt = stmt.order_by(ScratchpadEntry.entry_date.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def _get_existing_memories_text(self, user_id: UUID) -> str:
        """Get formatted text of existing memories for context."""
        existing_memories = await self.get_all_memories(user_id, limit=20)
        if not existing_memories:
            return "None"
        return "\n".join([f"- {m.content}" for m in existing_memories])

    async def _call_llm_for_extraction(
        self, prompt: str, source_type: str
    ) -> Optional[dict]:
        """
        Call LLM to extract memories.

        Args:
            prompt: Formatted extraction prompt
            source_type: Type of source (for error messages)

        Returns:
            Parsed JSON dict with memories, or None if extraction fails
        """
        try:
            response = await self.llm_client.chat.completions.create(
                model=settings.memory_extraction_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for consistent extraction
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"{source_type.capitalize()} memory extraction failed: {e}")
            return None

    async def _store_extracted_memories(
        self,
        user_id: UUID,
        extracted_data: dict,
        source_conversation_id: Optional[UUID] = None,
    ) -> List[UserMemory]:
        """
        Store extracted memories in database.

        Args:
            user_id: User ID
            extracted_data: Parsed JSON from LLM extraction
            source_conversation_id: Source conversation (None for journal)

        Returns:
            List of newly created UserMemory objects
        """
        new_memories = []
        for mem_data in extracted_data.get("memories", []):
            memory = await self.add_memory(
                user_id=user_id,
                content=mem_data["content"],
                importance=mem_data.get("importance", 0.5),
                category=mem_data.get("category", "context"),
                source_conversation_id=source_conversation_id,
            )
            if memory:
                new_memories.append(memory)
        return new_memories

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages into readable conversation."""
        lines = []
        for msg in messages:
            role = msg.role.value.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n\n".join(lines)
