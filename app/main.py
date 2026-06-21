"""
Main application entrypoint for the Healthcare AI Assistant FastAPI application.

Defines API routes, global configurations, CORS settings, and startup lifecycles.
"""

import json
import uuid
import logging
import contextvars
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from app.config import get_settings
from app.embeddings import ingest_documents
from app.agent import route_and_answer

# Initialize settings
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Request-scoped correlation ID (Phase 3A)
# Stored in a ContextVar so every log line within a request automatically
# includes the same short UUID, enabling end-to-end tracing in logs.
# ─────────────────────────────────────────────────────────────────────────────
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    """Inject the current request_id into every log record for this process."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()  # type: ignore[attr-defined]
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Optional structured JSON formatter (Phase 3B)
# Activated when LOG_FORMAT=json in .env — useful for production observability
# platforms (Datadog, ELK, Splunk) that ingest structured JSON log streams.
# ─────────────────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        return json.dumps(log_obj)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
_log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

# Use a LogRecord factory that always pre-populates request_id="-" on every
# new record. This is the belt-and-suspenders default; the RequestIdFilter
# middleware overwrites it with the actual UUID for application log lines.
# Without this, uvicorn's own loggers (uvicorn.access, uvicorn.error) that
# bypass our root logger filter would raise KeyError: 'request_id' when
# Python's % formatter tries to expand %(request_id)s.
_original_factory = logging.getLogRecordFactory()

def _record_factory(*args, **kwargs):
    record = _original_factory(*args, **kwargs)
    record.request_id = "-"  # default; overwritten per-request by RequestIdFilter
    return record

logging.setLogRecordFactory(_record_factory)

if settings.LOG_FORMAT == "json":
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=_log_level, handlers=[_handler])
else:
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - [req:%(request_id)s] - %(message)s",
    )

# Also add RequestIdFilter to root logger and all its handlers so the ContextVar
# value (set by add_request_id middleware) is written onto each record during
# an active request.
_request_id_filter = RequestIdFilter()
_root_logger = logging.getLogger()
_root_logger.addFilter(_request_id_filter)
for _h in _root_logger.handlers:
    _h.addFilter(_request_id_filter)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager for startup and shutdown procedures.
    """
    logger.info("Healthcare AI Assistant starting up...")
    logger.info(
        "Configuration loaded successfully: Host=%s, Port=%d, LLM=%s",
        settings.APP_HOST, settings.APP_PORT, settings.GROQ_MODEL,
    )
    yield
    logger.info("Healthcare AI Assistant shutting down...")


app = FastAPI(
    title="Healthcare AI Assistant",
    description="A RAG-powered healthcare AI assistant built with FastAPI, LangChain, ChromaDB, and Groq.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Request correlation ID middleware (Phase 3A)
# Generates a short UUID per request, sets it on the ContextVar so all log
# lines within the request carry [req:<id>], and returns it as X-Request-ID.
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    logger.info(
        "Incoming request: %s %s [request_id=%s]",
        request.method, request.url.path, request_id,
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─────────────────────────────────────────────────────────────────────────────
# CORS (Phase 2B)
# allow_credentials is False (correct) — this app uses no cookies or auth
# headers. The previous allow_origins=["*"] + allow_credentials=True was a
# CORS/Fetch spec violation: browsers reject wildcard origin with credentials.
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.allowed_methods_list,
    allow_headers=settings.allowed_headers_list,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health of the API")
    version: str = Field(..., description="Version of the application")
    model: str = Field(..., description="Name of the LLM model in use")
    embedding_model: str = Field(..., description="Name of the embedding model in use")
    vector_store_status: str = Field(..., description="Status of the ChromaDB connection")


class IngestRequest(BaseModel):
    reset: bool = Field(default=False, description="Whether to reset the collection before ingestion")


class IngestResponse(BaseModel):
    status: str = Field(..., description="Operation status (success/error)")
    documents_loaded: int = Field(default=0, description="Number of documents loaded")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    collection_name: str = Field(default="", description="Name of the Chroma collection")
    message: str = Field(..., description="Ingestion result summary status")


class AskRequest(BaseModel):
    question: str = Field(..., description="Healthcare question")

    @field_validator("question")
    @classmethod
    def validate_length(cls, v: str) -> str:
        # Phase 2C: length limits come from config so they can be tuned
        # without code changes. Pydantic Field min_length/max_length cannot
        # be set dynamically, so a @field_validator is used instead.
        cfg = get_settings()
        if len(v) < cfg.QUESTION_MIN_LENGTH:
            raise ValueError(
                f"Question must be at least {cfg.QUESTION_MIN_LENGTH} characters"
            )
        if len(v) > cfg.QUESTION_MAX_LENGTH:
            raise ValueError(
                f"Question must be at most {cfg.QUESTION_MAX_LENGTH} characters"
            )
        return v


class AskResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    answer: str = Field(..., description="The generated answer")
    sources: list[dict] = Field(default=[], description="Source document chunks used")
    confidence: str = Field(..., description="Confidence level: none/low/medium/high/conversational")
    question: str = Field(..., description="The original user question")
    model_used: str = Field(..., description="Name of the LLM model used")
    tool_used: str | None = Field(default=None, description="Name of the tool used (appointment routing only)")
    tool_response: dict | None = Field(default=None, description="Raw tool response (appointment routing only)")


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception raised on path %s: %s",
        request.url.path, str(exc), exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the administrator."},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health Checks"])
async def health_check() -> HealthResponse:
    try:
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=settings.GROQ_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            vector_store_status="ready",
        )
    except Exception as e:
        logger.error("Failed to compile health check status: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check execution failed: {str(e)}",
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_docs(payload: IngestRequest) -> IngestResponse:
    """
    Ingest documents from the data directory into the vector store.
    """
    try:
        result = ingest_documents(reset_collection=payload.reset)
        return IngestResponse(**result)
    except FileNotFoundError as e:
        logger.error("Data directory not found: %s", str(e))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "No documents found in data directory"},
        )
    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": str(e)},
        )
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ingestion.",
        )


@app.post("/ask", response_model=AskResponse, tags=["Querying"])
async def ask_assistant(payload: AskRequest) -> AskResponse:
    """
    Query the healthcare RAG assistant with a natural-language question.

    Returns an answer grounded in the ingested documents, along with
    source citations and a confidence label.
    """
    logger.info("Received /ask request: %s", payload.question)
    try:
        result = route_and_answer(question=payload.question)
        return AskResponse(**result)
    except ValueError as exc:
        error_msg = str(exc)
        if "empty" in error_msg.lower() or "ingest" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "error", "message": error_msg},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": error_msg},
        )
    except Exception as exc:
        logger.error("Unexpected error in /ask: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your question.",
        )
