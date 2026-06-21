"""
RAG Module for Healthcare AI Assistant.

Handles retrieval, prompt construction, LLM invocation, and
source formatting for the healthcare question-answering pipeline.
"""

import os
import logging
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings
from app.embeddings import get_embedding_function, get_chroma_client, get_or_create_collection
from app.llm import get_llm, invoke_with_fallback
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

# System prompt is now loaded from app/prompts/healthcare_rag_system_prompt.txt
# via the load_prompt() utility. See app/prompts.py for the rationale.


def get_retriever(vectorstore):
    """Return a similarity-search retriever from the given vectorstore."""
    settings = get_settings()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_K},
    )


def get_relevant_docs_with_threshold(vectorstore, question: str, k: int = 3, score_threshold: float = 0.35) -> list:
    # Threshold value lives in config for easy tuning without code changes;
    # the retrieval function itself stays here because config files cannot
    # contain executable logic.
    """
    Retrieve top-k chunks but only return those above a minimum similarity
    score to prevent hallucination on completely unrelated queries.

    Args:
        vectorstore:     The Chroma vectorstore.
        question:        User's question.
        k:               Maximum number of chunks to retrieve.
        score_threshold: Minimum relevance score (0.0–1.0, higher = stricter).

    Returns:
        Filtered list of Document objects.
    """
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    filtered = [doc for doc, score in docs_with_scores if score >= score_threshold]
    logger.info(
        "Retrieved %d/%d chunks above score threshold %.2f",
        len(filtered), len(docs_with_scores), score_threshold,
    )
    return filtered


def format_sources(source_documents: list) -> list:
    """
    Convert a list of LangChain Document objects into serialisable source dicts.

    Deduplicates by document basename (keeps first occurrence).

    Args:
        source_documents: List of LangChain Document objects.

    Returns:
        List of dicts with keys 'document' and 'chunk'.
    """
    seen = set()
    sources = []
    for doc in source_documents:
        raw_source = doc.metadata.get("source", "unknown")
        doc_name = os.path.basename(raw_source)

        if doc_name in seen:
            continue
        seen.add(doc_name)

        content = doc.page_content
        chunk_preview = content[:200] + "..." if len(content) > 200 else content

        sources.append({
            "document": doc_name,
            "chunk": chunk_preview,
        })
    return sources


def calculate_confidence(answer: str, sources: list) -> str:
    """
    Derive a simple confidence label from the answer text and source count.

    Args:
        answer:  The LLM-generated answer string.
        sources: Deduplicated list of source dicts.

    Returns:
        One of "none", "low", "medium", or "high".
    """
    if "could not find" in answer.lower():
        return "none"
    if len(sources) >= 2:
        return "high"
    if len(sources) == 1:
        return "medium"
    return "low"


def query_rag(question: str) -> dict:
    """
    Orchestrate the full RAG pipeline for a single question.

    Steps:
        1. Load the persistent vectorstore.
        2. Guard against an empty collection (prompt user to /ingest).
        3. Retrieve top-k relevant document chunks.
        4. Inject chunks into the system prompt and call the LLM.
        5. Build and return the structured response dict.

    Args:
        question: The user's healthcare question.

    Returns:
        Dict with keys: answer, sources, confidence, question, model_used.

    Raises:
        ValueError: If the knowledge base is empty.
        Exception:  Re-raised after logging for any other error.
    """
    settings = get_settings()
    logger.info("RAG query received: %s", question)

    try:
        # ── 1. Load vectorstore ──────────────────────────────────────────────
        embedding_fn = get_embedding_function()
        client = get_chroma_client()
        vectorstore = get_or_create_collection(client, embedding_fn)

        # ── 2. Guard: empty collection ───────────────────────────────────────
        raw_collection = client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        doc_count = raw_collection.count()
        if doc_count == 0:
            logger.warning("Knowledge base is empty. User must run /ingest first.")
            raise ValueError(
                "Knowledge base is empty. Please call POST /ingest first."
            )

        # ── 3. Retrieve relevant chunks with score threshold ─────────────────
        # Threshold value comes from settings.SIMILARITY_SCORE_THRESHOLD
        # (config-driven) — see app/config.py and .env.example to tune it.
        retrieved_docs = get_relevant_docs_with_threshold(
            vectorstore, question,
            k=settings.RETRIEVAL_K,
            score_threshold=settings.SIMILARITY_SCORE_THRESHOLD,
        )
        logger.info("Using %d chunks after score filtering", len(retrieved_docs))

        # If no chunks pass the threshold, the question is out-of-scope
        if not retrieved_docs:
            logger.info("No relevant chunks found above threshold — returning not-found response.")
            return {
                "answer": (
                    "I could not find this information in the provided documents. "
                    "Please contact the healthcare facility directly for assistance."
                ),
                "sources": [],
                "confidence": "none",
                "question": question,
                "model_used": settings.GROQ_MODEL,
            }

        # ── 4. Build context and call LLM ────────────────────────────────────
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        system_prompt_template = load_prompt("healthcare_rag_system_prompt.txt")
        filled_prompt = system_prompt_template.format(context=context)

        messages = [
            SystemMessage(content=filled_prompt),
            HumanMessage(content=question),
        ]
        # Use fallback chain — returns (answer_text, model_name_that_succeeded)
        answer, model_used = invoke_with_fallback(messages)
        logger.info("Answer generated successfully via model: %s", model_used)

        # ── 5. Build response ────────────────────────────────────────────────
        sources = format_sources(retrieved_docs)
        confidence = calculate_confidence(answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "question": question,
            "model_used": model_used,
        }

    except ValueError:
        # Re-raise ValueError so the endpoint can return a 503
        raise
    except Exception as exc:
        logger.error("RAG pipeline error: %s", str(exc), exc_info=True)
        raise
