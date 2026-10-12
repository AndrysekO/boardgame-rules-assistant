"""Tests for src/api.py. None of them call a model.

Run with: uv run pytest
"""
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    """Fail any test that would reach answer() or agent_answer(), and start
    every test without RULES_API_KEY, so no key is needed by default."""
    def fail(*args, **kwargs):
        raise AssertionError("a test called the model")
    monkeypatch.setattr(api, "answer", fail)
    monkeypatch.setattr(api, "agent_answer", fail)
    monkeypatch.delenv("RULES_API_KEY", raising=False)


def error_messages(response):
    return " ".join(error["msg"] for error in response.json()["detail"])


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_games():
    response = client.get("/games")
    assert response.status_code == 200
    assert response.json() == ["Terraforming Mars", "Everdell", "Bang"]


def test_unknown_game_is_422():
    response = client.post("/ask", json={"question": "How do I win?", "game": "Chess"})
    assert response.status_code == 422
    assert "unknown game 'Chess'" in error_messages(response)


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_422(question):
    response = client.post("/ask", json={"question": question, "game": "Bang"})
    assert response.status_code == 422
    assert "question must not be empty" in error_messages(response)


def test_unknown_mode_is_422():
    response = client.post("/ask", json={"question": "How do I win?", "game": "Bang",
                                         "mode": "fast"})
    assert response.status_code == 422


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("RULES_API_KEY", "test-key")
    assert client.get("/games").status_code == 401
    assert client.get("/games", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/games", headers={"X-API-Key": "test-key"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open


def test_citations_in():
    text = "Quote: ... [Everdell p.6] and [Terraforming Mars FAQ p.18], again [Everdell p.6]"
    assert api.citations_in(text) == [{"game": "Everdell", "page": 6},
                                      {"game": "Terraforming Mars", "page": 18}]
