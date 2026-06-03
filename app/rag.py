"""
RAG Module for Healthcare AI Assistant.

Handles retrieval, prompt construction, LLM invocation, and
source formatting for the healthcare question-answering pipeline.
"""

import logging
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings
from app.embeddings import get_embedding_function, get_chroma_client, get_or_create_collection
from app.llm import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level system prompt constant
# ---------------------------------------------------------------------------
HEALTHCARE_SYSTEM_PROMPT = """
You are a healthcare information assistant for a medical facility. 
Your role is to answer questions based ONLY on the provided context 
from official healthcare documents.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use any external 
   knowledge or make assumptions beyond what is explicitly stated.
2. If the context does not contain enough information to answer the 
   question, respond with exactly: 
   "I could not find this information in the provided documents. 
   Please contact the healthcare facility directly for assistance."
3. Never provide medical diagnoses, treatment recommendations, or 
   medical advice beyond what is explicitly stated in the documents.
4. Always maintain a professional, clear, and empathetic tone.
5. If citing information, refer to it as coming from official 
   facility documentation.
6. Do not speculate, extrapolate, or fill gaps with general medical 
   knowledge.

Context from documents:
{context}

Remember: Only answer from the context above. If uncertain, say so.
"""


def get_retriever(vectorstore):
    """Return a similarity-search retriever from the given vectorstore."""
    settings = get_settings()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_K},
    )


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
        import os
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

        # ── 3. Retrieve relevant chunks ──────────────────────────────────────
        retriever = get_retriever(vectorstore)
        retrieved_docs = retriever.invoke(question)
        logger.info("Retrieved %d chunks for question", len(retrieved_docs))

        # ── 4. Build context and call LLM ────────────────────────────────────
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        filled_prompt = HEALTHCARE_SYSTEM_PROMPT.format(context=context)

        llm = get_llm()
        messages = [
            SystemMessage(content=filled_prompt),
            HumanMessage(content=question),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()
        logger.info("Answer generated successfully")

        # ── 5. Build response ────────────────────────────────────────────────
        sources = format_sources(retrieved_docs)
        confidence = calculate_confidence(answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "question": question,
            "model_used": settings.GROQ_MODEL,
        }

    except ValueError:
        # Re-raise ValueError so the endpoint can return a 503
        raise
    except Exception as exc:
        logger.error("RAG pipeline error: %s", str(exc), exc_info=True)
        raise
