---
title: Healthcare AI Assistant Backend
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

<div align="center">

# 🏥 Healthcare AI Assistant

**A production-grade RAG-powered AI assistant for healthcare document Q&A**
<!--
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://ai-rag-shaunakchorge.streamlit.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface)](https://awesomedevworks-healthcare-ai-backend.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/ShaunakChorge/AI-RAG)
-->
</div>

---

## 📋 Table of Contents

1. [Live Demo](#-live-demo)
2. [Problem Statement](#-problem-statement)
3. [Architecture](#-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Knowledge Base Documents](#-knowledge-base-documents)
7. [Setup & Installation](#-setup--installation)
8. [Running the Application](#-running-the-application)
9. [Docker Setup](#-docker-setup)
10. [API Reference](#-api-reference)
11. [Agent & Tool Workflow](#-agent--tool-workflow)
12. [Prompt Engineering Strategy](#-prompt-engineering-strategy)
13. [Sample Questions & Responses](#-sample-questions--responses)
14. [LLM, Embedding & Vector DB Choices](#-llm-embedding--vector-db-choices)
15. [Hallucination Prevention](#-hallucination-prevention)
16. [Healthcare Compliance Note](#-healthcare-compliance-note)
17. [Limitations & Future Improvements](#-limitations--future-improvements)

---

## 🚀 Live Demo

| Component | URL |
|-----------|-----|
| **Interactive Frontend** | [ai-rag-shaunakchorge.streamlit.app](https://ai-rag-shaunakchorge.streamlit.app) |
| **FastAPI Backend Docs** | [healthcare-ai-backend.hf.space/docs](https://awesomedevworks-healthcare-ai-backend.hf.space/docs) |

> **Tip:** Open the frontend, click **Ingest Documents** in the sidebar first, then start asking healthcare questions.

---

## 🎯 Problem Statement

Mindbowser works with healthcare clients who need AI assistants capable of answering questions from internal clinical, operational, and compliance documents. This prototype demonstrates a fully working **Retrieval-Augmented Generation (RAG)** system that:

1. **Ingests** healthcare documents from a local folder
2. **Stores** searchable vector embeddings in ChromaDB
3. **Retrieves** the most relevant context chunks for any user question
4. **Generates** grounded, citation-backed answers using a hosted LLM
5. **Cites** the source document and exact chunk used in each response
6. **Refuses to hallucinate** — returns a clear fallback if no relevant information exists
7. **Exposes** all functionality through a documented REST API
8. **Routes** appointment-booking queries to a mock scheduling tool (agentic workflow)

> ⚠️ **No real patient data or PHI is used.** All documents are synthetic, facility-policy-style text created specifically for this prototype.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User (Browser)                                │
│               Streamlit Frontend UI                              │
│         (ai-rag-shaunakchorge.streamlit.app)                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │  HTTPS POST /ask, POST /ingest, GET /health
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Docker on HF Spaces)               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Agent Router (agent.py)                 │   │
│   │                                                          │   │
│   │   Intent Detection (keyword-based opt-in routing)        │   │
│   │        │                                                 │   │
│   │        ├── APPOINTMENT keywords ──► Mock Scheduling Tool │   │
│   │        │      check_available_slots(dept, date)          │   │
│   │        │                                                 │   │
│   │        ├── HEALTHCARE keywords ────► RAG Pipeline        │   │
│   │        │      ChromaDB Vector Search                     │   │
│   │        │      all-MiniLM-L6-v2 Embeddings                │   │
│   │        │      Similarity Score Threshold (≥ 0.35)        │   │
│   │        │      Groq LLM (llama-3.1-8b-instant)            │   │
│   │        │      Structured Response + Citations            │   │
│   │        │                                                 │   │
│   │        └── CONVERSATIONAL (default) ──► Rule-based reply  │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                      │  Embeddings API
                      ▼
        ┌─────────────────────────────┐
        │  Groq Cloud API             │
        │  llama-3.1-8b-instant       │
        │  (sub-second inference)     │
        └─────────────────────────────┘
```

### Data Flow — RAG Pipeline

```
User Question
     │
     ▼
Embed question with all-MiniLM-L6-v2
     │
     ▼
Similarity search in ChromaDB
     │
     ▼
Score threshold filter (≥ 0.35) ──► No match? Return "not found" instantly
     │
     ▼
Inject top-k chunks into system prompt
     │
     ▼
Groq LLM generates grounded answer
     │
     ▼
Return: { answer, sources, confidence }
```

---

## 🛠️ Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| **LLM** | Groq `llama-3.1-8b-instant` | Sub-second latency, strong instruction following, generous free tier |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Lightweight CPU-friendly model, excellent semantic similarity |
| **Vector DB** | ChromaDB (persistent) | Local disk persistence, native LangChain integration, no external service |
| **RAG Framework** | LangChain | Retriever abstractions, document loaders, text splitters |
| **API Framework** | FastAPI | Async, auto OpenAPI docs, Pydantic validation, production-grade |
| **Frontend** | Streamlit | Rapid prototyping, rich widget ecosystem, Streamlit Cloud deployment |
| **Configuration** | Pydantic Settings | Type-safe env variable loading, `.env` file support |
| **Containerization** | Docker + Docker Compose | Reproducible builds, one-command deployment |
| **Hosting (Backend)** | Hugging Face Spaces | Free Docker hosting, 16GB RAM, persistent containers |
| **Hosting (Frontend)** | Streamlit Community Cloud | Free Streamlit hosting, GitHub integration |

---

## 📁 Project Structure

```
AI-RAG/
├── app/
│   ├── __init__.py          # Package marker
│   ├── agent.py             # Intent detection & 3-way router (appointment / RAG / conversational)
│   ├── config.py            # Pydantic Settings — all config from .env
│   ├── embeddings.py        # ChromaDB client, HuggingFace embeddings, ingestion pipeline
│   ├── llm.py               # Groq LLM wrapper
│   ├── main.py              # FastAPI app — routes, middleware, error handlers
│   └── rag.py               # RAG pipeline — retrieval, score filtering, prompt, LLM call
├── data/                    # Synthetic healthcare policy documents (TXT)
│   ├── appointment_policy.txt
│   ├── discharge_instructions.txt
│   ├── hipaa_guidelines.txt
│   ├── insurance_faq.txt
│   ├── medication_refill_policy.txt
│   └── telehealth_guidelines.txt
├── tests/
│   ├── test_api.py          # pytest smoke tests for all endpoints
│   └── sample_questions.md  # Manual test questions with expected answers
├── vector_store/            # Auto-created — ChromaDB persistent storage
├── .env.example             # Template for environment variables
├── .gitignore
├── Dockerfile               # Multi-stage production Docker image
├── docker-compose.yml       # One-command local orchestration
├── requirements.txt         # Python dependencies with pinned versions
├── streamlit_app.py         # Streamlit frontend (deployed separately)
└── README.md
```

---

## 📚 Knowledge Base Documents

All documents are **synthetic** — written to simulate realistic healthcare facility policies. No real patient data or PHI is included.

| File | Topic | Key Content |
|------|-------|-------------|
| `appointment_policy.txt` | Scheduling & booking | How to book, cancellation policy, no-show rules, arrival requirements, department-specific requirements |
| `discharge_instructions.txt` | Post-procedure care | Activity restrictions, wound care, medication instructions, follow-up requirements, emergency warning signs |
| `hipaa_guidelines.txt` | Privacy & compliance | PHI handling, patient rights, data retention (7 years for adults), breach notification (60-day window) |
| `insurance_faq.txt` | Billing & coverage | Claims process, billing disputes (30-day window), financial assistance eligibility, EOB explanation |
| `medication_refill_policy.txt` | Prescription management | Refill eligibility (75% usage threshold), controlled substance restrictions, portal vs. phone refills |
| `telehealth_guidelines.txt` | Virtual visit standards | Platform requirements, privacy rules (no public spaces), controlled substance restrictions, consent |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.11+**
- A free **Groq API key** from [console.groq.com](https://console.groq.com)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ShaunakChorge/AI-RAG.git
cd AI-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

All other values have sensible defaults and do not need to be changed for local development.

---

## ▶️ Running the Application

### Option A — FastAPI backend only (default)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive API docs (Swagger UI): `http://localhost:8000/docs`

### Option B — Backend + Streamlit frontend

Open two terminals:

**Terminal 1 — FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run streamlit_app.py
```

Frontend opens at `http://localhost:8501`.

### First-time setup — ingest documents

After the backend is running, ingest the knowledge base (required once, or after adding new documents):

```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"reset": true}'
```

You are now ready to ask questions.

---

## 🐳 Docker Setup

### Quick start with Docker Compose (recommended)

```bash
# 1. Build and start the container
docker-compose up --build

# 2. (In a separate terminal) Ingest documents
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"reset": true}'

# 3. Ask a question
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the medication refill policy?"}'
```

### Manual Docker build

```bash
docker build -t healthcare-ai .
docker run -p 8000:8000 --env-file .env healthcare-ai
```

### Notes on Docker

- The `vector_store/` directory is mounted as a volume so embeddings **persist** between container restarts.
- The `data/` directory is mounted as **read-only** — documents are never modified by the container.
- The embedding model (`all-MiniLM-L6-v2`) is **downloaded during the Docker build** and baked into the image, so no internet access is needed at runtime for embeddings.

---

## 📡 API Reference

### `GET /health`

Returns the current health status of the API and loaded models.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model": "llama-3.1-8b-instant",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "vector_store_status": "ready"
}
```

**cURL:**
```bash
curl http://localhost:8000/health
```

---

### `POST /ingest`

Reads all `.txt` files from the `data/` folder, splits them into chunks, generates embeddings, and stores them in ChromaDB.

**Request:**
```json
{ "reset": false }
```

Set `reset: true` to wipe the existing collection before ingesting (useful when documents change).

**Response:**
```json
{
  "status": "success",
  "documents_loaded": 6,
  "chunks_created": 94,
  "collection_name": "healthcare_docs",
  "message": "Successfully ingested 6 documents into 94 chunks"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"reset": true}'
```

---

### `POST /ask`

Accepts a natural-language healthcare question and returns a grounded answer with source citations.

**Request:**
```json
{ "question": "Can a patient request a medication refill through telehealth?" }
```

**Constraints:** `question` must be 3–500 characters.

**Response:**
```json
{
  "answer": "Yes, patients can request medication refills through telehealth if the medication is already prescribed and does not require an in-person evaluation.",
  "sources": [
    {
      "document": "telehealth_guidelines.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits provided the medication is non-controlled and has been previously prescribed..."
    }
  ],
  "confidence": "high",
  "question": "Can a patient request a medication refill through telehealth?",
  "model_used": "llama-3.1-8b-instant",
  "tool_used": null,
  "tool_response": null
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```

**Confidence levels:**

| Level | Meaning |
|-------|---------|
| `high` | 2+ source documents matched |
| `medium` | 1 source document matched |
| `low` | Sources matched but answer uncertain |
| `none` | No relevant information found in documents |
| `conversational` | Off-topic / greeting — no RAG used |

---

### Error Responses

| HTTP Code | Scenario |
|-----------|----------|
| `422 Unprocessable Entity` | Missing or invalid `question` field |
| `503 Service Unavailable` | Knowledge base is empty — call `/ingest` first |
| `500 Internal Server Error` | Unexpected server error (logged for debugging) |

---

## 🤖 Agent & Tool Workflow

The agent (`app/agent.py`) acts as a **3-way intent router** before any LLM is called:

```
Question
   │
   ├─ Matches APPOINTMENT_KEYWORDS? (e.g. "book an appointment", "available slots")
   │       └─► Mock Scheduling Tool: check_available_slots(department, date)
   │               Returns: available slots, booking instructions
   │
   ├─ Matches RAG_TRIGGER_KEYWORDS? (e.g. "medication", "hipaa", "discharge", "insurance")
   │       └─► RAG Pipeline
   │               1. Embed question with all-MiniLM-L6-v2
   │               2. Similarity search ChromaDB (top-3 chunks)
   │               3. Score threshold filter (≥ 0.35)
   │               4. Inject context into system prompt
   │               5. Call Groq LLM
   │               6. Return answer + citations + confidence
   │
   └─ No domain keywords matched? (e.g. "hello", "yo", "how are you")
           └─► Conversational Handler (no vector store, no LLM call)
                   Returns: friendly redirect to healthcare topics
```

**Key design decision:** RAG is **opt-in**. A query must contain at least one specific healthcare-domain keyword to trigger the vector search. This prevents wasted LLM calls and nonsensical citations for off-topic messages.

### Appointment Tool Example

**Question:** `"Can I book a cardiology appointment for Monday?"`

**Tool called:** `check_available_slots(department="Cardiology", date="Monday")`

**Response:**
```json
{
  "answer": "I can check appointment availability for Cardiology. Available slots on Monday: 9:00 AM, 11:30 AM, 2:00 PM, 4:00 PM. To confirm a booking, call 1-800-HEALTH-1 or visit the patient portal.",
  "tool_used": "check_available_slots",
  "tool_response": {
    "department": "Cardiology",
    "date": "Monday",
    "available_slots": ["9:00 AM", "11:30 AM", "2:00 PM", "4:00 PM"],
    "booking_instructions": "Call 1-800-HEALTH-1 or visit the patient portal to confirm"
  }
}
```

> Note: This is a **mock tool** with illustrative slot data. It does not connect to a real scheduling system.

---

## 🧠 Prompt Engineering Strategy

The system prompt (`app/rag.py`) is designed to enforce **strict knowledge-only answering**:

```
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
```

**Why this works:**
- Rule 1 prevents external knowledge injection
- Rule 2 provides an exact fallback string that can be detected programmatically (used by `calculate_confidence()` to set `confidence: "none"`)
- Rule 3 adds a healthcare-specific safety layer against harmful medical advice
- Rules 4–6 ensure professional tone and citation hygiene

**Additional hallucination guard:** Before the LLM is even called, retrieved chunks are filtered by a **similarity score threshold of 0.35**. If no chunk scores high enough, the "not found" response is returned instantly without consuming any LLM tokens.

---

## 💬 Sample Questions & Responses

### 1. RAG — Medication policy

**Q:** `"Can I refill an opioid prescription via the online portal?"`

**A:** `"No, controlled substances like opioid pain relievers cannot be refilled via the online portal. Refill requests for Schedule II controlled substances must be handled by calling the pharmacy directly or visiting in person."`

**Sources:** `medication_refill_policy.txt`  
**Confidence:** `high`

---

### 2. RAG — HIPAA

**Q:** `"How long are adult medical records kept?"`

**A:** `"Adult medical records are securely retained for a minimum of seven years after the last patient encounter."`

**Sources:** `hipaa_guidelines.txt`  
**Confidence:** `medium`

---

### 3. RAG — Discharge

**Q:** `"What is the weight limit for lifting after discharge?"`

**A:** `"You should not lift anything heavier than 10 pounds for the first two weeks following your discharge."`

**Sources:** `discharge_instructions.txt`  
**Confidence:** `medium`

---

### 4. Appointment Tool (agentic routing)

**Q:** `"Book a cardiology appointment for Monday"`

**A:** `"I can check appointment availability for Cardiology. Available slots on Monday: 9:00 AM, 11:30 AM, 2:00 PM, 4:00 PM. To confirm a booking, call 1-800-HEALTH-1 or visit the patient portal."`

**Tool Used:** `check_available_slots`  
**Confidence:** `high`

---

### 5. Out-of-scope — correctly refused

**Q:** `"What is the cure for diabetes?"`

**A:** `"I could not find this information in the provided documents. Please contact the healthcare facility directly for assistance."`

**Sources:** none  
**Confidence:** `none`

---

### 6. Conversational — no RAG triggered

**Q:** `"Hello"`

**A:** `"Hello! I'm the Healthcare AI Assistant. I'm designed to answer questions about our facility's healthcare policies, medications, insurance, telehealth, and appointment scheduling. How can I assist you today?"`

**Model:** `rule_based_routing` (no vector search, no LLM call)

---

## 🔬 LLM, Embedding & Vector DB Choices

### LLM — Groq `llama-3.1-8b-instant`

- **Why Groq over OpenAI:** Groq's LPU inference hardware delivers sub-second response times even on 8B parameter models. The free tier is generous enough for a prototype.
- **Why llama-3.1-8b-instant:** Provides strong instruction-following at a small memory footprint. The "instant" variant is tuned for fast inference, making the chatbot feel real-time.
- **Limitation:** Groq is a cloud service — requires internet and an API key. For fully on-premise deployment, this could be replaced with a local Ollama instance running Llama or Mistral.

### Embedding Model — `all-MiniLM-L6-v2`

- **Why this model:** 384-dimensional sentence embeddings with excellent semantic similarity performance on domain-specific texts. The model runs entirely on CPU with under 100MB memory.
- **Normalization:** Embeddings are L2-normalized at inference time, making cosine similarity equivalent to dot product for ChromaDB's relevance scoring.
- **Baked into Docker image:** Downloaded during `docker build` so there is no network dependency at container runtime.

### Vector Database — ChromaDB

- **Why ChromaDB:** Pure-Python, zero-infrastructure vector store that persists to disk. For a prototype with 6 documents, it offers excellent performance with no external service or cloud dependency.
- **Persistence:** Data is stored in `./vector_store/` which is Docker-volume-mounted, so embeddings survive container restarts.
- **Scale path:** For production with millions of documents, ChromaDB can be replaced with Weaviate, Pinecone, or Qdrant with minimal code changes (only `app/embeddings.py` needs updating).

---

## 🛡️ Hallucination Prevention

This system uses **three independent layers** to prevent the LLM from hallucinating:

| Layer | Mechanism | Where |
|-------|-----------|-------|
| **1. Intent gating** | RAG only activates if the query contains a specific healthcare keyword | `app/agent.py` |
| **2. Score threshold** | Chunks with relevance score < 0.35 are discarded before the LLM sees them | `app/rag.py` |
| **3. Prompt enforcement** | System prompt strictly forbids external knowledge and mandates the exact fallback string | `app/rag.py` |

---

## ⚕️ Healthcare Compliance Note

This prototype uses **synthetic, publicly available policy-style documents** and never processes real Protected Health Information (PHI).

For production HIPAA compliance, the following would be required:
- All data stored on encrypted, access-controlled infrastructure (AES-256 at rest, TLS in transit)
- Audit logging and role-based access control (RBAC) for all data access
- Regular risk assessments and vulnerability scanning
- Signed Business Associate Agreements (BAAs) with all cloud providers (Groq, HF Spaces, etc.)
- Robust incident-response and breach notification procedures (60-day window per HIPAA)
- Replacement of Groq Cloud with a HIPAA-compliant or on-premise LLM provider

---

## 🚧 Limitations & Future Improvements

| # | Current Limitation | Suggested Improvement |
|---|-------------------|-----------------------|
| 1 | **Mock scheduling tool** — returns hardcoded slots | Integrate with a real EHR/scheduling API (e.g. HL7 FHIR) |
| 2 | **Keyword-based intent routing** — may misclassify edge cases | Replace with a lightweight intent classification model (e.g. zero-shot with BART) |
| 3 | **Static document set** — no automated re-ingestion on file change | Add a file-watcher that triggers re-ingestion when documents in `data/` change |
| 4 | **No authentication** — all endpoints are public | Implement OAuth2 / JWT bearer token authentication |
| 5 | **Single-language support** — English only | Use multilingual embedding models (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) |
| 6 | **No conversation memory** — each query is independent | Add LangChain `ConversationBufferMemory` for multi-turn context |
| 7 | **Groq cloud dependency** — requires internet | Add Ollama local LLM fallback for fully offline / on-premise deployment |
| 8 | **No evaluation metrics** — answer quality not measured | Implement RAGAS evaluation (faithfulness, answer relevancy, context recall) |
| 9 | **Chunk size is fixed** — no dynamic chunking | Experiment with semantic chunking based on sentence boundaries |
| 10 | **No document versioning** — old and new chunks co-exist after update | Implement document-level hash tracking and selective re-ingestion |

---

## 🧪 Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest

# Run all smoke tests
pytest tests/test_api.py -v
```

The test suite covers:
- `GET /health` — returns 200 with all expected fields
- `POST /ask` — rejects missing `question` field (422)
- `POST /ask` — rejects questions shorter than 3 characters (422)
- `POST /ask` — returns 503 (not 500) when knowledge base is empty
- `POST /ingest` — successfully ingests all 6 documents
- `POST /ingest` — returns 400 when data directory has no documents

---

## 📄 License

This project was created as a hackathon-style assignment prototype. All synthetic documents are original content created for demonstration purposes only.
