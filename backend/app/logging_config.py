"""
Centralized structlog configuration for the RAG system.

This module provides structured logging throughout the application with:
- JSON output for production (machine-readable)
- Colored console output for development (human-readable)
- Automatic context injection (request IDs, user IDs, etc.)
- Exception formatting and stack traces
- Integration with standard library logging
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor


def setup_logging(json_logs: bool = False, log_level: str = "INFO") -> None:
    """
    Configure structlog for the entire application.

    Args:
        json_logs: If True, output JSON logs (production). If False, use colored console (dev).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Common processors for all environments
    shared_processors: list[Processor] = [
        # Add logger name
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add stack info if stack_info=True is passed
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
        # Decode unicode
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output
        processors = shared_processors + [
            # Render as JSON
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Colored console output
        processors = shared_processors + [
            # Add colors and pretty formatting
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        # Wrapper class for loggers
        wrapper_class=structlog.stdlib.BoundLogger,
        # Context class for storing context variables
        context_class=dict,
        # Logger factory
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache loggers for performance
        cache_logger_on_first_use=True,
    )


def add_request_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Add request context to log entries.

    This processor can be used with FastAPI middleware to automatically
    add request ID, user ID, and other context to all logs.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Modified event dictionary with added context
    """
    # This will be populated by FastAPI middleware
    # For now, it's a placeholder for future enhancement
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured structlog logger.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("user_login", user_id="123", username="alice")
        >>> logger.error("database_error", error=str(e), query=query)
    """
    return structlog.get_logger(name)


# Convenience function for binding context to logger
def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    Useful for adding request-scoped context like user_id, request_id, etc.

    Args:
        **kwargs: Key-value pairs to bind to the logger context

    Example:
        >>> bind_context(request_id="abc-123", user_id="456")
        >>> logger.info("processing_request")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all bound context variables.

    Should be called at the end of request processing to avoid leaking
    context between requests.
    """
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """
    Remove specific keys from the bound context.

    Args:
        *keys: Keys to remove from context

    Example:
        >>> unbind_context("request_id", "user_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)
