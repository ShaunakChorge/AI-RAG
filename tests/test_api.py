"""
Basic smoke tests for the Healthcare AI Assistant API.
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

def test_ask_without_ingest() -> None:
    """
    Verify the /ask endpoint returns a controlled error status (not 500 crash).
    Currently, since it is a stub, it should return 501 Not Implemented.
    """
    response = client.post("/ask", json={"query": "What is diabetes?"})
    # Must not crash with 500 Internal Server Error
    assert response.status_code != 500
    # Currently expected to return 501 Not Implemented
    assert response.status_code == 501

def test_ingest_endpoint() -> None:
    """
    Verify the /ingest endpoint successfully ingests documents and returns 200 OK.
    """
    response = client.post("/ingest", json={"reset": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_loaded"] == 6
    assert data["chunks_created"] == 94
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
    assert data["message"] == "No documents found in data directory"

