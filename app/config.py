from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    All configurable values live here — never hardcode them in application
    logic. This makes tuning, auditing, and deployment changes code-free.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Required ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""

    # ── LLM Settings ──────────────────────────────────────────────────────────
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Comma-separated fallback models tried in order if the primary fails.
    # Edit this list without code changes if Groq deprecates a model.
    # (llama-3.3-70b-versatile and llama-3.1-8b-instant deprecated June 2026;
    #  openai/gpt-oss-20b and openai/gpt-oss-120b are the recommended replacements.)
    GROQ_FALLBACK_MODELS: str = "llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b"

    # ── Embedding Settings ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── ChromaDB Settings ─────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./vector_store"
    CHROMA_COLLECTION_NAME: str = "healthcare_docs"

    # ── Document Ingestion Settings ───────────────────────────────────────────
    DATA_DIR: str = "./data"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_K: int = 3

    # ── Retrieval Threshold ───────────────────────────────────────────────────
    # Minimum cosine similarity score for a chunk to be used in RAG context.
    # Raise to be stricter (fewer hallucinations, more "not found" replies);
    # lower to be more permissive (more answers, higher hallucination risk).
    SIMILARITY_SCORE_THRESHOLD: float = 0.20

    # ── CORS Settings ─────────────────────────────────────────────────────────
    # Comma-separated list. For local dev, point at your Streamlit port.
    # For production, set this to your deployed frontend's exact URL(s).
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"
    ALLOWED_METHODS: str = "GET,POST"
    ALLOWED_HEADERS: str = "Content-Type"
    # IMPORTANT: allow_credentials=False is correct — this app uses no cookies
    # or auth headers. The previous True + wildcard origin was a CORS spec
    # violation: browsers reject wildcard origin combined with credentials.
    CORS_ALLOW_CREDENTIALS: bool = False

    # ── Question Length Validation ────────────────────────────────────────────
    QUESTION_MIN_LENGTH: int = 3
    QUESTION_MAX_LENGTH: int = 500

    # ── Server Settings ───────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── Logging Format ────────────────────────────────────────────────────────
    # "text" for human-readable plain-text logs (default, good for development).
    # "json" for structured JSON logs (production observability platforms like
    #  Datadog, ELK, Splunk — each log line is a parseable JSON object).
    LOG_FORMAT: str = "text"

    # ── Parsed helper properties ──────────────────────────────────────────────

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_methods_list(self) -> list[str]:
        return [m.strip() for m in self.ALLOWED_METHODS.split(",") if m.strip()]

    @property
    def allowed_headers_list(self) -> list[str]:
        return [h.strip() for h in self.ALLOWED_HEADERS.split(",") if h.strip()]

    @property
    def groq_fallback_models_list(self) -> list[str]:
        return [m.strip() for m in self.GROQ_FALLBACK_MODELS.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Retrieve and cache application settings.
    """
    return Settings()
