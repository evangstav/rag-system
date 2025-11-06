from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, conversations, scratchpad, rag
from app.config import settings
from app.logging_config import setup_logging, get_logger

# Configure structured logging on startup
json_logs = settings.environment == "production"
log_level = "DEBUG" if settings.debug else "INFO"
setup_logging(json_logs=json_logs, log_level=log_level)

logger = get_logger(__name__)

app = FastAPI(title="RAG Chat System")

# Log startup information
logger.info(
    "application_starting",
    environment=settings.environment,
    debug=settings.debug,
    log_format="json" if json_logs else "console",
    log_level=log_level,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(scratchpad.router, prefix="/api/scratchpad", tags=["scratchpad"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])


@app.get("/health")
async def health():
    logger.debug("health_check_requested")
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info("application_ready", message="RAG Chat System is ready to accept requests")
