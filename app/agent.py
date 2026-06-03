"""
Agent Module for Healthcare AI Assistant.

Coordinates intent detection, tool dispatching, and routing between
the scheduling mock tool, the RAG pipeline, and the conversational
fallback for the /ask endpoint.

Routing priority (checked in order):
  1. APPOINTMENT_INTENT  — booking-action keywords  → mock scheduling tool
  2. RAG_INTENT          — healthcare-topic keywords → RAG pipeline
  3. CONVERSATIONAL_INTENT (default)                → polite fallback reply
"""

import re
import logging
from app.rag import query_rag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent constants
# ---------------------------------------------------------------------------

APPOINTMENT_INTENT    = "appointment"
RAG_INTENT            = "rag"
CONVERSATIONAL_INTENT = "conversational"

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Booking-action words → mock scheduling tool
APPOINTMENT_KEYWORDS = [
    "book",
    "schedule",
    "available slots",
    "see a doctor",
    "consult",
    "when can i",
    "next available",
    "slot",
    "cardiology",
    "orthopedics",
    "neurology",
    "general medicine",
]

# Healthcare / document-domain words → RAG pipeline.
# If NONE of these appear in the query, we skip the vector store entirely.
RAG_TRIGGER_KEYWORDS = [
    # Medical topics
    "medication", "medicine", "prescription", "drug", "refill", "dosage",
    "treatment", "diagnosis", "symptom", "condition", "disease", "illness",
    "surgery", "procedure", "recovery", "discharge", "follow-up",
    "pain", "allergy", "vaccine", "vaccination", "lab", "test", "result",
    # Facility / admin
    "insurance", "coverage", "claim", "billing", "payment", "copay",
    "deductible", "prior authorization", "formulary",
    "appointment", "clinic", "hospital", "doctor", "physician",
    "nurse", "patient", "portal", "record", "medical record",
    "department", "specialist", "referral", "emergency",
    "telehealth", "virtual visit", "telemedicine",
    # Policy / compliance
    "hipaa", "privacy", "phi", "data", "consent", "rights",
    "policy", "guideline", "procedure", "instruction",
    "hours", "open", "closed", "when", "how long", "how do i",
    "what is the", "what are", "can i", "do i need",
]

# ---------------------------------------------------------------------------
# Department / date helpers (for appointment extraction)
# ---------------------------------------------------------------------------

_DEPARTMENT_MAP = {
    "cardiology":  ("cardiology", "heart"),
    "orthopedics": ("orthopedic", "bone", "joint"),
    "neurology":   ("neuro", "brain"),
}

_DATE_TOKENS = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "today", "tomorrow", "next week",
]


# ---------------------------------------------------------------------------
# 1. Intent detection
# ---------------------------------------------------------------------------

def detect_intent(question: str) -> str:
    """
    Classify the user's question into one of three intents.

    Priority order:
      1. APPOINTMENT_INTENT  — booking-action keyword found
      2. RAG_INTENT          — at least one healthcare keyword found
      3. CONVERSATIONAL_INTENT — no domain keywords; treat as chit-chat

    Args:
        question: The raw user question.

    Returns:
        One of APPOINTMENT_INTENT, RAG_INTENT, CONVERSATIONAL_INTENT.
    """
    lowered = question.lower().strip()

    # ── 1. Appointment check (highest priority) ──────────────────────────────
    for keyword in APPOINTMENT_KEYWORDS:
        if keyword in lowered:
            logger.info(
                "Intent detected: %s (matched keyword: '%s')",
                APPOINTMENT_INTENT, keyword,
            )
            return APPOINTMENT_INTENT

    # ── 2. RAG check — opt-in: only if a domain keyword is present ───────────
    for keyword in RAG_TRIGGER_KEYWORDS:
        if keyword in lowered:
            logger.info(
                "Intent detected: %s (matched RAG keyword: '%s')",
                RAG_INTENT, keyword,
            )
            return RAG_INTENT

    # ── 3. Default: conversational / off-topic ───────────────────────────────
    logger.info("Intent detected: %s (no domain keywords found)", CONVERSATIONAL_INTENT)
    return CONVERSATIONAL_INTENT


# ---------------------------------------------------------------------------
# 2. Mock scheduling tool
# ---------------------------------------------------------------------------

def check_available_slots(department: str, date: str) -> dict:
    """
    Pure mock scheduling tool — no external calls.

    Args:
        department: Title-cased department name (e.g. "Cardiology").
        date:       Date string extracted from the question.

    Returns:
        Dict with mock slot data and booking instructions.
    """
    return {
        "department": department.title(),
        "date": date,
        "available_slots": ["9:00 AM", "11:30 AM", "2:00 PM", "4:00 PM"],
        "booking_instructions": (
            "Call 1-800-HEALTH-1 or visit patient portal to confirm"
        ),
        "note": (
            "Slot availability is illustrative. "
            "Contact scheduling for real-time availability."
        ),
        "next_available_date": "next business day if date unavailable",
    }


# ---------------------------------------------------------------------------
# 3. Appointment handler
# ---------------------------------------------------------------------------

def handle_appointment_question(question: str) -> dict:
    """
    Handle appointment / scheduling questions using the mock tool.

    Args:
        question: The user's scheduling-related question.

    Returns:
        Dict matching the AskResponse schema, plus tool_used and tool_response.
    """
    lowered = question.lower()

    # --- Department extraction ---
    department = "General Medicine"
    for dept, synonyms in _DEPARTMENT_MAP.items():
        if any(syn in lowered for syn in synonyms):
            department = dept.title()
            break

    # --- Date extraction ---
    date = "Next Available"
    for token in _DATE_TOKENS:
        if token in lowered:
            date = token.title()
            break

    logger.info(
        "Appointment handler: department='%s', date='%s'", department, date
    )

    tool_response = check_available_slots(department, date)

    slots_str = ", ".join(tool_response["available_slots"])
    answer = (
        f"I can check appointment availability for {department}. "
        f"Available slots on {date}: {slots_str}. "
        f"To confirm a booking, {tool_response['booking_instructions']}. "
        f"Note: {tool_response['note']}"
    )

    return {
        "answer": answer,
        "sources": [
            {
                "document": "appointment_policy.txt",
                "chunk": "Mock scheduling tool response",
            }
        ],
        "confidence": "high",
        "question": question,
        "model_used": "scheduling_tool_v1",
        "tool_used": "check_available_slots",
        "tool_response": tool_response,
    }


# ---------------------------------------------------------------------------
# 4. Conversational handler
# ---------------------------------------------------------------------------

def handle_conversational(question: str) -> dict:
    """
    Handle greetings, small-talk, and any off-topic messages without
    touching the vector store.

    Produces a warm, on-brand response that redirects the user toward
    supported healthcare topics.
    """
    lowered = question.lower().strip()

    # Personalised response for identity questions
    if any(phrase in lowered for phrase in ["your name", "who are you", "what are you"]):
        answer = (
            "I'm the Healthcare AI Assistant for this medical facility. "
            "I can answer questions about our policies, medications, insurance, "
            "discharge instructions, telehealth guidelines, and appointment availability. "
            "What can I help you with today?"
        )
    elif any(phrase in lowered for phrase in ["how are you", "you okay", "you good"]):
        answer = (
            "I'm doing great and ready to help! "
            "Feel free to ask me anything about our healthcare services, "
            "policies, or appointments."
        )
    else:
        # Generic catch-all for anything off-topic
        answer = (
            "Hello! I'm the Healthcare AI Assistant. "
            "I'm designed to answer questions about our facility's healthcare policies, "
            "medications, insurance, telehealth, and appointment scheduling. "
            "How can I assist you today?"
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": "high",
        "question": question,
        "model_used": "rule_based_routing",
        "tool_used": None,
        "tool_response": None,
    }


# ---------------------------------------------------------------------------
# 5. Main router
# ---------------------------------------------------------------------------

def route_and_answer(question: str) -> dict:
    """
    Detect intent and dispatch to the appropriate handler.

    - APPOINTMENT_INTENT    → handle_appointment_question (mock scheduling tool)
    - RAG_INTENT            → query_rag (vector-search + LLM pipeline)
    - CONVERSATIONAL_INTENT → handle_conversational (no vector store)

    Args:
        question: The user's question.

    Returns:
        Structured response dict compatible with AskResponse schema.
    """
    intent = detect_intent(question)

    if intent == APPOINTMENT_INTENT:
        logger.info("Routing to appointment handler for question: %s", question)
        return handle_appointment_question(question)

    if intent == RAG_INTENT:
        logger.info("Routing to RAG pipeline for question: %s", question)
        return query_rag(question)

    logger.info("Routing to conversational handler for question: %s", question)
    return handle_conversational(question)
