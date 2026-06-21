"""
Agent Module for Healthcare AI Assistant.

Coordinates intent detection, tool dispatching, and routing between
the scheduling mock tool, the RAG pipeline, and the LLM-grounded
conversational persona handler for the /ask endpoint.

Routing priority (checked in order):
  1. APPOINTMENT_INTENT    — booking-action keywords      -> mock scheduling tool
  2. CONVERSATIONAL_INTENT — greetings/small-talk/off-topic -> LLM persona handler
  3. RAG_INTENT            — default for ALL other input  -> RAG pipeline

ARCHITECTURE NOTE (Phase 4, 2026-06-21):
  The conversational handler previously returned hardcoded if/elif string
  responses ("Hello! I'm the Healthcare AI Assistant..."). This was replaced
  with an LLM call grounded in conversational_persona_prompt.txt so that:
  - Responses vary naturally (no identical canned text every time)
  - The LLM is explicitly instructed NEVER to invent facility facts
  - Off-topic, general-medical, and gibberish inputs are handled gracefully
    in-character without hallucinating
  The PURE_GREETING_PATTERNS regex list was removed — no longer needed since
  the persona handler uses a lightweight heuristic instead of exact-match regex.
"""

import re
import logging
from app.rag import query_rag
from app.config import get_settings
from app.prompts import load_prompt
from app.llm import invoke_with_fallback

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

# Booking-action words -> mock scheduling tool.
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

# Greeting / small-talk trigger words — used for the lightweight CONVERSATIONAL
# heuristic in detect_intent(). These are plain substrings, not regex, and only
# trigger CONVERSATIONAL_INTENT when the question is short AND contains none of
# the healthcare_hint_terms below. This way "how are telehealth visits billed?"
# stays in RAG while "how are you?" goes to the persona handler.
_GREETING_TRIGGERS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thank you", "thanks", "thx", "ty", "bye", "goodbye", "see you", "cya",
    "take care", "who are you", "what are you", "what is your name",
    "what do you do", "are you an ai", "are you a robot", "are you real",
    "how are you", "you okay", "what can you help", "can you help me",
    "what kind of questions", "i need help",
]

# Clearly off-topic patterns — these are very unlikely to be healthcare queries.
# Match as substrings in the lowered question.
_OFF_TOPIC_TRIGGERS = [
    "capital of", "what is the weather", "tell me a joke",
    "what is 2 plus", "what is 2+", "solve for x",
    "write me a poem", "write a poem", "write code", "help me code",
    "python code", "javascript", "factorial of",
]

# Healthcare hint terms — if any of these appear, the question almost certainly
# belongs in RAG even if it looks like a greeting/small-talk pattern.
# Example: "how are telehealth appointments billed?" contains "telehealth".
_HEALTHCARE_HINTS = [
    "appointment", "medication", "prescription", "refill", "discharge",
    "insurance", "hipaa", "phi", "record", "billing", "payment", "copay",
    "deductible", "claim", "telehealth", "telemedicine", "doctor", "nurse",
    "patient", "procedure", "wound", "follow-up", "followup", "surgery",
    "hospital", "clinic", "facility", "health", "medical", "medicine",
    "diagnosis", "treatment", "drug", "dose", "dosage", "schedule",
    "slots", "policy", "coverage", "network", "prior auth", "authorization",
    "emergency", "urgent", "consent", "privacy", "breach", "hipaa",
    "portal", "lab", "test", "result", "blood", "vitals",
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

    Priority order:
      1. APPOINTMENT_INTENT    — explicit booking-action phrase found
      2. CONVERSATIONAL_INTENT — lightweight heuristic: short question
                                  with a greeting/small-talk/off-topic
                                  signal AND no healthcare hint terms
      3. RAG_INTENT            — everything else (DEFAULT fallback)

    The heuristic for CONVERSATIONAL_INTENT is intentionally conservative:
    when uncertain, we prefer routing to RAG because the vector store's
    SIMILARITY_SCORE_THRESHOLD safely handles out-of-scope queries by
    returning "I could not find..." without hallucinating — much safer
    than skipping RAG for something that might actually have a document
    answer. We only skip RAG when the question is clearly non-healthcare.

    Args:
        question: The raw user question.

    Returns:
        One of APPOINTMENT_INTENT, CONVERSATIONAL_INTENT, RAG_INTENT.
    """
    lowered = question.lower().strip()

    # -- 1. Appointment booking check (highest priority) ----------------------
    for keyword in APPOINTMENT_KEYWORDS:
        if keyword in lowered:
            logger.info(
                "Intent detected: %s (booking keyword: '%s')",
                APPOINTMENT_INTENT, keyword,
            )
            return APPOINTMENT_INTENT

    # -- 2. Conversational heuristic ------------------------------------------
    # Only route to the persona handler if BOTH conditions are true:
    #   a) The question contains a clear greeting/small-talk/off-topic signal
    #   b) The question contains NO healthcare domain terms
    # This prevents "how are telehealth visits billed?" from being treated as
    # small talk just because it starts with "how are".

    # Check condition (b) first — fast exit if any healthcare hint found
    has_healthcare_hint = any(hint in lowered for hint in _HEALTHCARE_HINTS)

    if not has_healthcare_hint:
        # Check condition (a) — greeting/small-talk/off-topic signal present
        is_greeting = any(trigger in lowered for trigger in _GREETING_TRIGGERS)
        is_off_topic = any(trigger in lowered for trigger in _OFF_TOPIC_TRIGGERS)

        if is_greeting or is_off_topic:
            logger.info(
                "Intent detected: %s (no healthcare hints; greeting=%s, off_topic=%s)",
                CONVERSATIONAL_INTENT, is_greeting, is_off_topic,
            )
            return CONVERSATIONAL_INTENT

    # -- 3. Default: RAG pipeline ---------------------------------------------
    # All healthcare, policy, billing, HIPAA, telehealth, ambiguous, and
    # "how do I / can I / should I" healthcare questions land here.
    # The vector store score_threshold handles out-of-scope queries safely.
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
# 4. LLM-grounded conversational persona handler
# ---------------------------------------------------------------------------

def handle_conversational(question: str) -> dict:
    """
    Handle greetings, small talk, off-topic questions, and gibberish
    using an LLM call grounded in the conversational persona prompt.

    Unlike query_rag(), this path has NO document context — the LLM
    is instructed via the persona prompt to never invent facility
    facts and to redirect off-topic or general-medical questions
    back to its documented scope. This keeps the bot in character
    for every possible input without hallucinating.

    Uses invoke_with_fallback() so that if the primary model is
    unavailable, the same fallback chain used in RAG applies here too.

    Args:
        question: The user's message.

    Returns:
        Dict matching the AskResponse schema with confidence="conversational".
    """
    from langchain.schema import HumanMessage, SystemMessage

    logger.info("Routing to persona-grounded conversational handler")

    persona_prompt_template = load_prompt("conversational_persona_prompt.txt")
    filled_prompt = persona_prompt_template.format(message=question)

    messages = [
        SystemMessage(content=filled_prompt),
        HumanMessage(content=question),
    ]

    # invoke_with_fallback returns (answer_text, model_name_that_succeeded)
    answer, model_used = invoke_with_fallback(messages)

    return {
        "answer": answer,
        "sources": [],
        "confidence": "conversational",
        "question": question,
        "model_used": model_used,
        "tool_used": None,
        "tool_response": None,
    }


# ---------------------------------------------------------------------------
# 5. Main router
# ---------------------------------------------------------------------------

def route_and_answer(question: str) -> dict:
    """
    Detect intent and dispatch to the appropriate handler.

    Routing priority:
    - APPOINTMENT_INTENT    -> handle_appointment_question (mock scheduling tool)
    - CONVERSATIONAL_INTENT -> handle_conversational (LLM persona, no vector search)
    - RAG_INTENT (default)  -> query_rag (vector-search + LLM pipeline)

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
        logger.info("Routing to persona-grounded conversational handler: %s", question)
        return handle_conversational(question)

    # RAG is the default — catches all healthcare questions and ambiguous input
    logger.info("Routing to RAG pipeline: %s", question)
    return query_rag(question)
