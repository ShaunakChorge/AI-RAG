"""
Main application entrypoint for the Healthcare AI Assistant FastAPI application.

Defines API routes, global configurations, CORS settings, and startup lifecycles.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.config import get_settings
from app.embeddings import ingest_documents
from app.agent import route_and_answer

# Initialize settings and configuration
settings = get_settings()

# Setup logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager for startup and shutdown procedures.
    """
    logger.info("Healthcare AI Assistant starting up...")
    logger.info("Configuration loaded successfully: Host=%s, Port=%d, LLM=%s", 
                settings.APP_HOST, settings.APP_PORT, settings.GROQ_MODEL)
    yield
    logger.info("Healthcare AI Assistant shutting down...")

app = FastAPI(
    title="Healthcare AI Assistant",
    description="A RAG-powered healthcare AI assistant built with FastAPI, LangChain, ChromaDB, and Groq.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for hackathon/demo frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response and Request Pydantic Schemas
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
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Healthcare question (3–500 characters)"
    )

class AskResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    answer: str = Field(..., description="The generated answer")
    sources: list[dict] = Field(default=[], description="Source document chunks used")
    confidence: str = Field(..., description="Confidence level: none/low/medium/high")
    question: str = Field(..., description="The original user question")
    model_used: str = Field(..., description="Name of the LLM model used")
    tool_used: str | None = Field(default=None, description="Name of the tool used (appointment routing only)")
    tool_response: dict | None = Field(default=None, description="Raw tool response (appointment routing only)")

# Global exception handler for unexpected API errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception raised on path %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the administrator."}
    )

# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health Checks"])
async def health_check() -> HealthResponse:
    try:
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=settings.GROQ_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            vector_store_status="ready"
        )
    except Exception as e:
        logger.error("Failed to compile health check status: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check execution failed: {str(e)}"
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
            content={"status": "error", "message": "No documents found in data directory"}
        )
    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ingestion."
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
        # Empty knowledge base — client must ingest first
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
