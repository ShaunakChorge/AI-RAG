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

# Booking-action words → mock scheduling tool.
# Kept tight: must be explicit booking verbs, NOT generic medical words.
APPOINTMENT_KEYWORDS = [
    "book an appointment",
    "schedule an appointment",
    "make an appointment",
    "book a",          # "book a cardiology", "book a slot"
    "available slots",
    "see a doctor",
    "when can i see",
    "next available",
    "book me",
    "i want to book",
    "i want to schedule",
    "i need an appointment",
]

# Healthcare / document-domain words → RAG pipeline.
# Only specific clinical, admin, and policy terms — NOT generic English words.
RAG_TRIGGER_KEYWORDS = [
    # Clinical / medical
    "medication", "medicine", "prescription", "drug", "drugs", "refill",
    "dosage", "dose", "treatment", "diagnosis", "symptom", "symptoms",
    "condition", "disease", "illness", "surgery", "procedure", "recovery",
    "discharge", "follow-up", "follow up", "allergy", "allergies",
    "vaccine", "vaccination", "immunization", "x-ray", "mri", "ct scan",
    "blood pressure", "blood test", "cholesterol", "diabetes",
    # Facility / admin
    "insurance", "coverage", "co-pay", "copay", "claim", "billing",
    "deductible", "prior authorization", "formulary", "network",
    "appointment policy", "clinic hours", "office hours",
    "hospital", "physician", "nurse", "patient portal", "medical record",
    "specialist", "referral", "emergency room", "urgent care",
    "telehealth", "virtual visit", "telemedicine", "video visit",
    "department hours",
    # Policy / compliance
    "hipaa", "privacy policy", "phi", "patient rights", "consent form",
    "policy", "guideline", "guidelines", "instruction", "instructions",
    "facility policy", "healthcare policy",
    # Natural question starters about the documents
    "what is the policy",
    "what are the guidelines",
    "how do i request",
    "how long does",
    "how many days",
    "can a patient",
    "am i allowed",
    "do i need a referral",
    "is it covered",
    "what documents",
    "how to prepare",
    "after discharge",
    "before surgery",
    "during recovery",
]

# ---------------------------------------------------------------------------
# Department / date helpers (for appointment extraction)
# ---------------------------------------------------------------------------

_DEPARTMENT_MAP = {
    "cardiology":  ("cardiology", "heart", "cardiac"),
    "orthopedics": ("orthopedic", "orthopedics", "bone", "joint", "spine"),
    "neurology":   ("neurology", "neurology", "neuro", "brain", "nerve"),
    "general medicine": ("general medicine", "general practitioner", "gp"),
    "dermatology": ("dermatology", "skin", "dermatologist"),
    "pediatrics":  ("pediatric", "pediatrics", "child", "children"),
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

    Priority order:
      1. APPOINTMENT_INTENT  — explicit booking-action phrase found
      2. RAG_INTENT          — at least one specific healthcare keyword found
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

    # ── 2. RAG check — opt-in: only if a specific domain keyword is present ──
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

    Produces a warm, on-brand response that redirects the user toward
    supported healthcare topics.
    """
    lowered = question.lower().strip()

    # Personalised response for identity questions
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
