"""
Agent Module for Healthcare AI Assistant.

Coordinates intent detection, tool dispatching, and routing between
the scheduling mock tool, the RAG pipeline, and the conversational
fallback for the /ask endpoint.

Routing priority (checked in order):
  1. APPOINTMENT_INTENT    — booking-action keywords      → mock scheduling tool
  2. CONVERSATIONAL_INTENT — pure greeting/small-talk     → polite fallback reply
  3. RAG_INTENT            — default for ALL other input  → RAG pipeline

FIX v2.0 (2026-06-04):
  - Expanded RAG_TRIGGER_KEYWORDS with 60+ missing terms covering wound care,
    appointment policy, billing, HIPAA, refill, and telehealth domains.
  - Changed default routing fallback from CONVERSATIONAL to RAG.
    Rationale: When in doubt, the RAG pipeline is always safer than a
    generic greeting. The vector store's score_threshold (0.35) already
    handles truly out-of-scope questions by returning "could not find".
  - Moved CONVERSATIONAL check to priority 2 (only after appointment check),
    so pure greetings are still handled gracefully without touching the LLM.
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

# Booking-action words → mock scheduling tool.
# Kept tight: must be explicit booking verbs, NOT generic medical words.
APPOINTMENT_KEYWORDS = [
    "book an appointment",
    "schedule an appointment",
    "make an appointment",
    "book a",           # "book a cardiology", "book a slot"
    "available slots",
    "see a doctor",
    "when can i see",
    "next available",
    "book me",
    "i want to book",
    "i want to schedule",
    "i need an appointment",
]

# Pure greeting/small-talk patterns (regex, matched against full lowered question).
# ONLY exact or near-exact greetings — any healthcare word escapes this check.
PURE_GREETING_PATTERNS = [
    r"^hi+$",
    r"^hello+$",
    r"^hey+$",
    r"^hi there$",
    r"^hello there$",
    r"^good morning$",
    r"^good afternoon$",
    r"^good evening$",
    r"^thank you$",
    r"^thank you so much$",
    r"^thanks+$",
    r"^thx$",
    r"^ty$",
    r"^bye$",
    r"^goodbye$",
    r"^see you$",
    r"^cya$",
    r"^take care$",
    r"^who are you\??$",
    r"^what are you\??$",
    r"^what is your name\??$",
    r"^what do you do\??$",
    r"^are you an? ai\??$",
    r"^are you a robot\??$",
    r"^are you real\??$",
    r"^how are you\??$",
    r"^you okay\??$",
    r"^what can you (help me with|do)\??$",
    r"^can you help me\??$",
    r"^what kind of questions can i ask\??$",
    r"^i need help$",
]

# ---------------------------------------------------------------------------
# Department / date helpers (for appointment extraction)
# ---------------------------------------------------------------------------

_DEPARTMENT_MAP = {
    "cardiology":       ("cardiology", "heart", "cardiac"),
    "orthopedics":      ("orthopedic", "orthopedics", "bone", "joint", "spine"),
    "neurology":        ("neurology", "neuro", "brain", "nerve"),
    "general medicine": ("general medicine", "general practitioner", "gp"),
    "dermatology":      ("dermatology", "skin", "dermatologist"),
    "pediatrics":       ("pediatric", "pediatrics", "child", "children"),
}

_DATE_TOKENS = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "today", "tomorrow", "next week",
    "this week", "next monday", "next tuesday", "next wednesday",
    "next thursday", "next friday",
]


# ---------------------------------------------------------------------------
# 1. Intent detection
# ---------------------------------------------------------------------------

def detect_intent(question: str) -> str:
    """
    Classify the user's question into one of three intents.

    Priority order (v2.0):
      1. APPOINTMENT_INTENT    — explicit booking-action phrase found
      2. CONVERSATIONAL_INTENT — pure greeting/small-talk (regex, exact match)
      3. RAG_INTENT            — everything else (DEFAULT fallback)

    The RAG fallback is intentionally the catch-all. The RAG pipeline's
    score_threshold already handles out-of-scope questions by returning
    "I could not find this information..." without hallucinating.

    Args:
        question: The raw user question.

    Returns:
        One of APPOINTMENT_INTENT, CONVERSATIONAL_INTENT, RAG_INTENT.
    """
    lowered = question.lower().strip()

    # ── 1. Appointment booking check (highest priority) ──────────────────────
    for keyword in APPOINTMENT_KEYWORDS:
        if keyword in lowered:
            logger.info(
                "Intent detected: %s (booking keyword: '%s')",
                APPOINTMENT_INTENT, keyword,
            )
            return APPOINTMENT_INTENT

    # ── 2. Pure greeting check (only short, exact-match phrases) ─────────────
    for pattern in PURE_GREETING_PATTERNS:
        if re.fullmatch(pattern, lowered):
            logger.info(
                "Intent detected: %s (greeting pattern: '%s')",
                CONVERSATIONAL_INTENT, pattern,
            )
            return CONVERSATIONAL_INTENT

    # ── 3. Default: RAG pipeline ─────────────────────────────────────────────
    # All medical, policy, billing, HIPAA, appointment-info, telehealth,
    # and any ambiguous questions land here. The vector store score_threshold
    # handles out-of-scope queries safely without hallucination.
    logger.info("Intent detected: %s (default fallback)", RAG_INTENT)
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
            "Call 1-800-HEALTH-1 or visit the patient portal to confirm"
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
                "chunk": "Mock scheduling tool response — slots are illustrative only.",
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

    Only reached for PURE greetings that match PURE_GREETING_PATTERNS.
    All healthcare, policy, and ambiguous questions bypass this handler.
    """
    lowered = question.lower().strip()

    if any(phrase in lowered for phrase in ["your name", "who are you", "what are you", "what do you do"]):
        answer = (
            "I'm the Healthcare AI Assistant for this medical facility. "
            "I can answer questions about our policies, medications, insurance, "
            "discharge instructions, telehealth guidelines, and appointment availability. "
            "What can I help you with today?"
        )
    elif any(phrase in lowered for phrase in ["how are you", "you okay", "you good", "are you well"]):
        answer = (
            "I'm doing great and ready to help! "
            "Feel free to ask me anything about our healthcare services, "
            "policies, or appointments."
        )
    elif any(phrase in lowered for phrase in ["thank", "thanks", "thank you", "thx"]):
        answer = (
            "You're welcome! If you have any more healthcare questions, "
            "feel free to ask anytime."
        )
    elif any(phrase in lowered for phrase in ["bye", "goodbye", "see you", "cya"]):
        answer = (
            "Goodbye! Take care and stay healthy. "
            "Come back anytime you have healthcare questions."
        )
    else:
        answer = (
            "Hello! I'm the Healthcare AI Assistant. "
            "I'm designed to answer questions about our facility's healthcare policies, "
            "medications, insurance, telehealth, and appointment scheduling. "
            "How can I assist you today?"
        )

    return {
        "answer": answer,
        "sources": [],
        "confidence": "conversational",
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

    v2.0 routing priority:
    - APPOINTMENT_INTENT    → handle_appointment_question (mock scheduling tool)
    - CONVERSATIONAL_INTENT → handle_conversational (pure greetings only)
    - RAG_INTENT (default)  → query_rag (vector-search + LLM pipeline)

    Args:
        question: The user's question.

    Returns:
        Structured response dict compatible with AskResponse schema.
    """
    intent = detect_intent(question)

    if intent == APPOINTMENT_INTENT:
        logger.info("Routing to appointment handler: %s", question)
        return handle_appointment_question(question)

    if intent == CONVERSATIONAL_INTENT:
        logger.info("Routing to conversational handler: %s", question)
        return handle_conversational(question)

    # RAG is the default — catches all healthcare questions and ambiguous input
    logger.info("Routing to RAG pipeline: %s", question)
    return query_rag(question)
