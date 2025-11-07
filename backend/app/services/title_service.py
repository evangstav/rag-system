"""
Title generation service for conversations.

Provides intelligent conversation title generation using LLM with fallbacks.
"""

import logging
from typing import Optional
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)


class TitleGenerationService:
    """
    Service for generating conversation titles using LLM.

    Features:
    - Context-aware title generation
    - Multiple fallback strategies
    - Configurable title length and style
    """

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        """
        Initialize the title generation service.

        Args:
            openai_client: Optional AsyncOpenAI client. If not provided, creates a new one.
        """
        self.client = openai_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.default_llm_model

    async def generate_title(
        self,
        user_message: str,
        assistant_response: str,
        max_length: int = 50,
        style: str = "concise"
    ) -> str:
        """
        Generate a conversation title based on the first exchange.

        Args:
            user_message: The user's first message
            assistant_response: The assistant's first response
            max_length: Maximum title length in characters
            style: Title style - "concise", "descriptive", or "topic"

        Returns:
            Generated title string
        """
        try:
            logger.info(
                f"Generating {style} title for conversation",
                extra={
                    "user_message_length": len(user_message),
                    "response_length": len(assistant_response),
                    "max_length": max_length
                }
            )

            # Build system prompt based on style
            system_prompt = self._build_system_prompt(style, max_length)

            # Call LLM to generate title
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response},
                ],
                max_completion_tokens=50,
                temperature=0.7,
            )

            generated_title = response.choices[0].message.content.strip()

            # Clean up the title (remove quotes, extra whitespace)
            generated_title = self._clean_title(generated_title)

            # Ensure title doesn't exceed max length
            if len(generated_title) > max_length:
                generated_title = generated_title[:max_length-3] + "..."

            # Validate title is not empty
            if not generated_title or len(generated_title) < 3:
                logger.warning("Generated title too short, using fallback")
                return self._fallback_title(user_message, max_length)

            logger.info(f"Successfully generated title: {generated_title}")
            return generated_title

        except Exception as e:
            logger.error(f"Error generating title: {type(e).__name__}: {str(e)}")
            return self._fallback_title(user_message, max_length)

    def _build_system_prompt(self, style: str, max_length: int) -> str:
        """
        Build the system prompt for title generation based on style.

        Args:
            style: Title style preference
            max_length: Maximum title length

        Returns:
            System prompt string
        """
        base_instruction = (
            f"Generate a clear, engaging title for this conversation. "
            f"Maximum length: {max_length} characters. "
            "Return ONLY the title text, no quotes, no extra text."
        )

        style_guidelines = {
            "concise": (
                "Create a brief, punchy title (3-5 words) that captures "
                "the main topic or question."
            ),
            "descriptive": (
                "Create a descriptive title that clearly explains what the "
                "conversation is about. Use complete phrases."
            ),
            "topic": (
                "Create a topic-based title that categorizes the conversation. "
                "Focus on the subject matter."
            ),
        }

        guideline = style_guidelines.get(style, style_guidelines["concise"])

        return f"{base_instruction}\n\n{guideline}\n\nExamples:\n" \
               "- 'Python List Comprehension Help'\n" \
               "- 'Understanding React Hooks'\n" \
               "- 'SQL Query Optimization Tips'\n" \
               "- 'Debugging API Connection Issues'"

    def _clean_title(self, title: str) -> str:
        """
        Clean up generated title by removing quotes, extra whitespace, etc.

        Args:
            title: Raw generated title

        Returns:
            Cleaned title string
        """
        # Remove surrounding quotes
        title = title.strip('"\'`')

        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "Title: ",
            "title: ",
            "Conversation: ",
            "conversation: ",
            "Topic: ",
            "topic: ",
        ]

        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):]

        # Clean up whitespace
        title = " ".join(title.split())

        return title

    def _fallback_title(self, user_message: str, max_length: int = 50) -> str:
        """
        Generate a fallback title from the user message.

        Args:
            user_message: The user's message
            max_length: Maximum title length

        Returns:
            Fallback title string
        """
        # Clean the message
        clean_message = " ".join(user_message.split())

        # If message is short enough, use it as-is
        if len(clean_message) <= max_length:
            return clean_message

        # Otherwise, truncate intelligently
        # Try to break at a sentence or word boundary
        truncated = clean_message[:max_length-3]

        # Find last space to avoid cutting mid-word
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # Only use if we don't lose too much
            truncated = truncated[:last_space]

        return truncated + "..."

    async def regenerate_title(
        self,
        user_message: str,
        assistant_response: str,
        current_title: Optional[str] = None,
    ) -> str:
        """
        Regenerate a conversation title (for manual title updates).

        Similar to generate_title but can take into account the current title
        to try generating something different if requested.

        Args:
            user_message: The user's first message
            assistant_response: The assistant's first response
            current_title: The current title (if any)

        Returns:
            New generated title
        """
        # For now, just use the standard generation
        # In the future, could use current_title to generate alternatives
        return await self.generate_title(user_message, assistant_response)
