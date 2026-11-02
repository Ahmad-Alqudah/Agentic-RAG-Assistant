import pytest
from fastapi.testclient import TestClient

import app.api.main as api_module


class FakeGraph:

    def invoke(self, state):
        question = state["question"].lower()

        if "security incident" in question:
            return {
                **state,
                "status": "human_review",
                "route": "human_review",
                "retry_count": 0,
                "validation_passed": False,
                "answer": (
                    "This request requires human review "
                    "before an answer can be provided."
                ),
                "retrieved_sources": ["security_policy.txt"],
            }

        if "ceo" in question:
            return {
                **state,
                "status": "abstained",
                "route": "abstain",
                "retry_count": 1,
                "validation_passed": False,
                "answer": (
                    "I cannot find the answer "
                    "in the provided documents."
                ),
                "retrieved_sources": ["employee_policy.txt"],
            }

        return {
            **state,
            "status": "answered",
            "route": "generate",
            "retry_count": 1,
            "validation_passed": True,
            "answer": (
                "Remote work is permitted up to two "
                "days per week with manager approval."
            ),
            "retrieved_sources": ["employee_policy.txt"],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "graph",
        FakeGraph(),
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Agentic RAG Assistant"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["agent_loaded"] is True


def test_answered_question(client):
    response = client.post(
        "/ask",
        json={"question": "What is the work from home policy?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "answered"
    assert data["validation_passed"] is True
    assert "employee_policy.txt" in data["sources"]


def test_unknown_question(client):
    response = client.post(
        "/ask",
        json={"question": "Who is the CEO of NovaTech?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "abstained"
    assert data["sources"] == []


def test_human_review(client):
    response = client.post(
        "/ask",
        json={"question": "What should I do about a security incident?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "human_review"
    assert data["retry_count"] == 0
    assert data["sources"] == []


def test_empty_question(client):
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_missing_question(client):
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_whitespace_question(client):
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_question_too_long(client):
    response = client.post(
        "/ask",
        json={"question": "A" * 2001},
    )

    assert response.status_code == 422


def test_response_schema(client):
    response = client.post(
        "/ask",
        json={"question": "What is the remote work policy?"},
    )

    data = response.json()

    expected_fields = {
        "question",
        "answer",
        "status",
        "route",
        "retry_count",
        "validation_passed",
        "sources",
    }

    assert response.status_code == 200
    assert set(data.keys()) == expected_fields