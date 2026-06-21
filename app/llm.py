"""
LLM Module for Healthcare AI Assistant.

Provides LLM configuration and initialization using langchain-groq,
plus a fallback chain that automatically retries with alternative
models if the primary model fails (rate limit, API error, deprecation).
"""

import os
import logging
from langchain_groq import ChatGroq  # type: ignore
from app.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(model_override: str | None = None) -> ChatGroq:
    """
    Initialize and return a ChatGroq LLM instance.

    Reads API key and model from the cached application settings.
    Ensures that the GROQ_API_KEY is set in the environment variables.

    Args:
        model_override: If provided, use this model name instead of
                        settings.GROQ_MODEL. Used by the fallback chain
                        to request a specific model without changing globals.

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

    model_name = model_override or settings.GROQ_MODEL
    logger.info("Initializing ChatGroq with model: %s", model_name)
    return ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.1,
        max_tokens=1024,
    )


def invoke_with_fallback(messages: list) -> tuple[str, str]:
    """
    Try the primary Groq model, falling back through GROQ_FALLBACK_MODELS
    in order if the primary fails due to rate limiting, API errors, or
    timeouts (e.g. due to model deprecation).

    The response's model_used field is set to whichever model actually
    answered, giving evaluators visibility into fallback behaviour.

    Args:
        messages: List of LangChain message objects (SystemMessage, HumanMessage).

    Returns:
        Tuple of (response_text, model_name_that_succeeded)

    Raises:
        Exception: If every model in the chain fails.
    """
    settings = get_settings()
    models_to_try = [settings.GROQ_MODEL] + settings.groq_fallback_models_list
    last_exception = None

    for model_name in models_to_try:
        try:
            logger.info("Attempting LLM call with model: %s", model_name)
            llm = get_llm(model_override=model_name)
            response = llm.invoke(messages)
            logger.info("LLM call succeeded with model: %s", model_name)
            return response.content.strip(), model_name
        except Exception as exc:
            logger.warning(
                "Model %s failed: %s. Trying next fallback if available.",
                model_name, str(exc)
            )
            last_exception = exc
            continue

    logger.error("All models in fallback chain failed.")
    if last_exception is not None:
        raise last_exception
    raise Exception("All models in fallback chain failed or no models configured.")
