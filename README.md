# Healthcare AI Assistant

## Overview
The Healthcare AI Assistant is a Retrieval‑Augmented Generation (RAG) system that answers patient‑facing questions using only the official healthcare documents of a medical facility. It routes appointment‑booking queries to a mock scheduling tool and all other queries to a LLM‑driven knowledge‑base pipeline.

## Architecture
```
User Request
     │
     ▼
FastAPI Backend
     │
     ▼
Agent Router (Intent Detection)
     │
     ├──── Appointment Keywords? ──► Mock Scheduling Tool
     │                                      │
     └──── Knowledge Question? ──► RAG Pipeline
                                            │
                                 ChromaDB Vector Search
                                 (all-MiniLM-L6-v2 embeddings)
                                            │
                                 Retrieved Context (top 3 chunks)
                                            │
                                 Groq LLM (llama-3.1-8b-instant)
                                            │
                                 Structured Response + Sources
```

## Tech Stack
| Component   | Technology                                   | Reason                                                                        |
|------------|----------------------------------------------|-------------------------------------------------------------------------------|
| LLM        | Groq **llama‑3.1‑8b‑instant**                | Fast, high‑quality instruction following, free tier for quick prototyping      |
| Embeddings | **all‑MiniLM‑L6‑v2** (sentence‑transformers) | Lightweight CPU‑friendly model with good semantic similarity performance       |
| Vector DB  | **ChromaDB**                                 | Local persistence, no external service, native LangChain integration            |
| Framework  | **FastAPI**                                  | Modern async API framework, automatic OpenAPI docs, high performance           |
| RAG Library| **LangChain**                                | Provides abstractions for retrievers, document loaders, and prompt templates    |
| Config     | **Pydantic Settings**                        | Type‑safe environment configuration, easy defaults and validation               |

## System Prompt
```python
"""
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
```

## Setup Instructions
### Prerequisites
- Python **3.11+**
- Groq API key (free at `console.groq.com`)

### Installation
```bash
git clone https://github.com/yourusername/healthcare-ai-assistant.git
cd healthcare-ai-assistant
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running Locally
```bash
uvicorn app.main:app --reload --port 8000
```

### Running with Docker
```bash
docker-compose up --build
```

### First Time Setup
1. **Ingest documents**
   ```bash
   curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"reset": true}'
   ```
2. **Ask a question**
   ```bash
   curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the policy for medication refills?"}'
   ```
3. **Launch Streamlit UI** (optional frontend)
   ```bash
   # In a second terminal, while the FastAPI server is running:
   streamlit run streamlit_app.py
   # Opens at http://localhost:8501
   ```

## API Reference
### `GET /health`
**Response** (`application/json`):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model": "llama-3.1-8b-instant",
  "embedding_model": "all-MiniLM-L6-v2",
  "vector_store_status": "ready"
}
```
---
### `POST /ingest`
**Request** (`application/json`):
```json
{ "reset": false }
```
**Response** (`application/json`):
```json
{
  "status": "success",
  "documents_loaded": 42,
  "chunks_created": 350,
  "collection_name": "healthcare_docs",
  "message": "Ingestion completed"
}
```
---
### `POST /ask`
**Request** (`application/json`):
```json
{ "question": "When are the cardiology clinic hours?" }
```
**Response** (`application/json`):
```json
{
  "answer": "The cardiology clinic operates Monday‑Friday, 9 AM‑5 PM.",
  "sources": [
    { "document": "appointment_policy.txt", "chunk": "Cardiology clinic hours are Monday‑Friday 9 AM‑5 PM..." }
  ],
  "confidence": "high",
  "question": "When are the cardiology clinic hours?",
  "model_used": "llama-3.1-8b-instant",
  "tool_used": null,
  "tool_response": null
}
```
---
## Sample Questions and Responses
### 1. Normal RAG question
**Q:** *"What are the guidelines for telehealth consultations?"*
**A:** "Telehealth consultations must be conducted via the secure patient portal, last no longer than 30 minutes, and require prior authorization."
**Sources:** `telehealth_guidelines.txt`
---
### 2. Normal RAG question
**Q:** *"How should a patient prepare for discharge after hip replacement?"*
**A:** "Patients should arrange for physical therapy, ensure pain medication is prescribed, and schedule a follow‑up appointment within two weeks."
**Sources:** `discharge_instructions.txt`
---
### 3. Appointment booking question (tool routing)
**Q:** *"Can I book a cardiology appointment for next Tuesday?"*
**A:** "I can check appointment availability for Cardiology. Available slots on Tuesday: 9:00 AM, 11:30 AM, 2:00 PM, 4:00 PM. To confirm a booking, call 1‑800‑HEALTH‑1 or visit patient portal to confirm. Note: Slot availability is illustrative. Contact scheduling for real‑time availability."
**Tool Used:** `check_available_slots`
---
### 4. Cross‑document question
**Q:** *"What insurance plans cover medication refills?"*
**A:** "Medication refill coverage varies by plan. Refer to the insurance FAQ for specific formulary details."
**Sources:** `insurance_faq.txt`
---
### 5. Out‑of‑scope question
**Q:** *"What is the dosage for ibuprofen for a child?"*
**A:** "I could not find this information in the provided documents. Please contact the healthcare facility directly for assistance."
---
## Project Structure
```
Healthcare AI Assistant/
├─ app/
│   ├─ __init__.py                 # Package marker
│   ├─ agent.py                    # Intent detection & routing logic
│   ├─ config.py                   # Pydantic settings loader
│   ├─ embeddings.py               # Chroma client & embedding utilities
│   ├─ llm.py                      # Groq LLM wrapper
│   ├─ main.py                     # FastAPI entrypoint & routes
│   └─ rag.py                      # Retrieval‑augmented generation pipeline
├─ data/                           # Raw policy documents (TXT)
│   ├─ appointment_policy.txt
│   ├─ discharge_instructions.txt
│   ├─ hipaa_guidelines.txt
│   ├─ insurance_faq.txt
│   ├─ medication_refill_policy.txt
│   └─ telehealth_guidelines.txt
├─ vector_store/                   # Persisted Chroma DB files
├─ tests/                          # pytest suite
├─ .env.example                    # Example environment variables
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md                       # <-- this documentation
```
*Each file’s purpose is summarized inline above.*

## Knowledge Base Documents
| Filename                     | Topic                              | Key Policies Covered                                 |
|-----------------------------|------------------------------------|------------------------------------------------------|
| appointment_policy.txt      | Scheduling & booking               | Department hours, booking workflow, cancellation      |
| discharge_instructions.txt  | Post‑procedure care                | Follow‑up, medication, activity restrictions          |
| hipaa_guidelines.txt        | Privacy & security                 | PHI handling, data retention, breach reporting        |
| insurance_faq.txt           | Billing & coverage                 | Coverage limits, formularies, pre‑authorization       |
| medication_refill_policy.txt| Prescription management            | Refill intervals, documentation required             |
| telehealth_guidelines.txt   | Virtual visit standards            | Platform requirements, consent, documentation        |

## LLM and Model Choices
- **Groq over local LLM:** Groq provides a hosted inference API with sub‑second latency and a generous free tier, eliminating the need for GPU hardware and simplifying deployment.
- **`llama‑3.1‑8b‑instant`:** Offers strong instruction‑following capabilities at a small footprint, ideal for fast RAG responses.
- **`all‑MiniLM‑L6‑v2`:** CPU‑friendly embedding model that balances speed and semantic quality, enabling real‑time vector search on modest hardware.
- **ChromaDB:** Pure‑Python vector store that persists to disk, works seamlessly with LangChain, and avoids external service costs.

## Agent Workflow
1. **Intent Detection** (`detect_intent`) scans the user question for booking‑action keywords.
2. **If appointment intent** → `handle_appointment_question` extracts department/date, calls the mock `check_available_slots`, and formats a response.
3. **Otherwise** → `query_rag` retrieves the top‑k chunks from Chroma, injects them into the `HEALTHCARE_SYSTEM_PROMPT`, calls the LLM, and returns answer + sources.
4. The router (`route_and_answer`) returns a unified response schema used by the FastAPI `/ask` endpoint.

## Prompt Engineering Strategy
- **System‑prompt** enforces *knowledge‑only* answers, preventing hallucinations.
- **Strict rules** guarantee compliance with healthcare communication standards (no diagnosis, no external knowledge).
- **Context placeholder** `{context}` injects the retrieved text, keeping the prompt concise.
- **Confidence labeling** derives from source count, giving downstream clients a reliability signal.

## Limitations and Future Improvements
1. **Mock scheduling tool** – replace with real EHR integration.
2. **Static document set** – add automated ingestion pipelines for new policies.
3. **No authentication** – implement OAuth2/JWT for protected endpoints.
4. **Single‑language support** – extend to multilingual embeddings/models.
5. **Limited hallucination guard** – explore retrieval‑augmented generation with self‑critique.

## Healthcare Compliance Note
The current prototype uses **synthetic, publicly available policy documents** and never processes real Protected Health Information (PHI). For production compliance (HIPAA):
- Store all data on encrypted, access‑controlled infrastructure.
- Enforce audit logging and role‑based access control.
- Perform regular risk assessments and vulnerability scanning.
- Sign Business Associate Agreements (BAAs) with any cloud providers.
- Implement robust incident‑response procedures for data breaches.

---

