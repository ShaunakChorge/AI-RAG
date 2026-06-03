"""
Smoke tests for the Healthcare AI Assistant API.

Run with:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """
    Verify the health check endpoint returns 200 OK and includes all expected keys.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data
    assert "embedding_model" in data
    assert "vector_store_status" in data


def test_ask_rejects_missing_question_field() -> None:
    """
    Verify the /ask endpoint returns 422 Unprocessable Entity when
    the required 'question' field is missing from the request body.
    """
    response = client.post("/ask", json={"query": "What is diabetes?"})
    assert response.status_code == 422


def test_ask_rejects_too_short_question() -> None:
    """
    Verify the /ask endpoint returns 422 when the question is shorter than 3 chars.
    """
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_without_ingest_returns_controlled_error() -> None:
    """
    Verify the /ask endpoint returns 503 (not a 500 crash) when the
    knowledge base is empty and a genuine RAG question is asked.
    """
    response = client.post("/ask", json={"question": "What is the medication refill policy?"})
    # Must not crash with 500 Internal Server Error
    assert response.status_code != 500
    # Empty KB returns 503 Service Unavailable
    assert response.status_code == 503


def test_ingest_endpoint() -> None:
    """
    Verify the /ingest endpoint successfully ingests documents and returns 200 OK.
    """
    response = client.post("/ingest", json={"reset": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_loaded"] >= 1       # at least 1 doc loaded
    assert data["chunks_created"] >= 1         # at least 1 chunk created
    assert data["collection_name"] == "healthcare_docs"


def test_ingest_empty_directory(monkeypatch) -> None:
    """
    Verify that /ingest returns HTTP 400 and appropriate error if no documents are found.
    """
    from app.embeddings import load_documents_from_folder
    def mock_load(folder_path):
        raise ValueError("No documents found in data directory")

    monkeypatch.setattr("app.embeddings.load_documents_from_folder", mock_load)

    response = client.post("/ingest", json={"reset": True})
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "No documents found" in data["message"]
