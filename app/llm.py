"""
LLM Module for Healthcare AI Assistant.

Provides LLM configuration and initialization using langchain-groq.
"""

import os
import logging
from langchain_groq import ChatGroq
from app.config import get_settings

logger = logging.getLogger(__name__)

def get_llm() -> ChatGroq:
    """
    Initialize and return a ChatGroq LLM instance.

    Reads API key and model from the cached application settings.
    Ensures that the GROQ_API_KEY is set in the environment variables.

    Returns:
        ChatGroq: A configured instance of ChatGroq.

    Raises:
        ValueError: If GROQ_API_KEY is missing or contains the default placeholder.
    """
    settings = get_settings()
    api_key = settings.GROQ_API_KEY

    if not api_key or api_key == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY is missing or placeholder value is used.")
        raise ValueError("GROQ_API_KEY is not configured in environment or .env file.")

    # Expose to environment for components that read it directly
    os.environ["GROQ_API_KEY"] = api_key

    logger.info("Initializing ChatGroq with model: %s", settings.GROQ_MODEL)
    return ChatGroq(
        groq_api_key=api_key,
        model=settings.GROQ_MODEL,
        temperature=0.1,
        max_tokens=1024,
    )
