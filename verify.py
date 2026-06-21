import requests
import time
import json

BASE_URL = "http://localhost:8000"

def run_tests():
    print("--- 7B: Health Endpoint ---")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(f"Headers: {dict(resp.headers)}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")
    
    print("--- 7J: Short Question (<3 chars) ---")
    resp = requests.post(f"{BASE_URL}/ask", json={"question": "hi"})
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

    print("--- 7H: X-Request-ID Header ---")
    req_id = resp.headers.get("x-request-id")
    print(f"Header present: {req_id is not None}, Value: {req_id}\n")

    print("--- 7C: Ingest Endpoint ---")
    print("Triggering ingest... this will take a moment")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/ingest", json={"reset": True})
    print(f"Status: {resp.status_code} (took {time.time()-start:.1f}s)")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

    print("--- 7D: RAG Question (How do I) ---")
    resp = requests.post(f"{BASE_URL}/ask", json={"question": "how do I refill my medication via the portal?"})
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

    print("--- 7E: Appointment Routing ---")
    resp = requests.post(f"{BASE_URL}/ask", json={"question": "book a cardiology appointment for next week"})
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

    print("--- 7F: Conversational Routing ---")
    resp = requests.post(f"{BASE_URL}/ask", json={"question": "Hello there, who are you?"})
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

    print("--- 7G: Out-of-scope Refusal ---")
    resp = requests.post(f"{BASE_URL}/ask", json={"question": "What is the capital of France?"})
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}\n")

if __name__ == '__main__':
    run_tests()
