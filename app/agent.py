"""
Agent Module for Healthcare AI Assistant.

Coordinates intent detection, tool dispatching, and routing between
the scheduling mock tool and the RAG pipeline for the /ask endpoint.
"""

import logging
from app.rag import query_rag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent constants and keyword lists
# ---------------------------------------------------------------------------

RAG_INTENT = "rag"
APPOINTMENT_INTENT = "appointment"
GREETING_INTENT = "greeting"

# Only booking-action words trigger the appointment tool.
# Informational words like "appointment" alone route to RAG.
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

# Department synonyms for extraction
_DEPARTMENT_MAP = {
    "cardiology": ("cardiology", "heart"),
    "orthopedics": ("orthopedic", "bone", "joint"),
    "neurology": ("neuro", "brain"),
}

# Day / relative-date tokens for extraction
_DATE_TOKENS = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "today", "tomorrow", "next week",
]

_GREETING_TOKENS = [
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "yo", "what is your name", "who are you", "how are you"
]



# ---------------------------------------------------------------------------
# 1. Intent detection
# ---------------------------------------------------------------------------

def detect_intent(question: str) -> str:
    """
    Classify the user's question as APPOINTMENT_INTENT or RAG_INTENT.

    Uses a keyword list focused on booking-action words rather than
    general appointment mentions, so policy questions still route to RAG.

    Args:
        question: The raw user question.

    Returns:
        APPOINTMENT_INTENT or RAG_INTENT string constant.
    """
    lowered = question.lower().strip()
    for keyword in APPOINTMENT_KEYWORDS:
        if keyword in lowered:
            logger.info("Intent detected: %s (matched keyword: '%s')", APPOINTMENT_INTENT, keyword)
            return APPOINTMENT_INTENT

    # Simple heuristic: if the question is very short and starts with or equals a greeting
    import re
    if len(lowered) < 30:
        for greeting in _GREETING_TOKENS:
            if re.search(r'\b' + re.escape(greeting) + r'\b', lowered):
                logger.info("Intent detected: %s", GREETING_INTENT)
                return GREETING_INTENT

    logger.info("Intent detected: %s", RAG_INTENT)
    return RAG_INTENT


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

    Extracts department and date from the question via simple keyword
    matching, calls check_available_slots, and formats a structured response.

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
# 4. Greeting handler
# ---------------------------------------------------------------------------

def handle_greeting(question: str) -> dict:
    """
    Handle generic greetings directly to avoid unnecessary RAG searches.
    """
    return {
        "answer": "Hello! I am the Healthcare AI Assistant. How can I help you with our facility's policies, guidelines, or appointments today?",
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

    - APPOINTMENT_INTENT → handle_appointment_question (mock scheduling tool)
    - RAG_INTENT         → query_rag (vector-search + LLM pipeline)

    Args:
        question: The user's question.

    Returns:
        Structured response dict compatible with AskResponse schema.
    """
    intent = detect_intent(question)

    if intent == APPOINTMENT_INTENT:
        logger.info("Routing to appointment handler for question: %s", question)
        return handle_appointment_question(question)
        
    if intent == GREETING_INTENT:
        logger.info("Routing to greeting handler for question: %s", question)
        return handle_greeting(question)

    logger.info("Routing to RAG pipeline for question: %s", question)
    return query_rag(question)
