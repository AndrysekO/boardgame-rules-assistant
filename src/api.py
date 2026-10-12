"""HTTP API for the rules assistant.

Run from the repo root (the Chroma index lives in data/chroma):
    uv run uvicorn api:app --app-dir src --reload
With Claude instead of the local Ollama model:
    RULES_MODEL=claude-haiku-5-5 uv run uvicorn api:app --app-dir src
Interactive docs: http://127.0.0.1:8000/docs

Try it:
    curl http://127.0.0.1:8000/games
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "What does the Barrel card do?", "game": "Bang"}'

If RULES_API_KEY is set in the environment, /games and /ask need a matching
X-API-Key header (add -H "X-API-Key: $RULES_API_KEY" to curl). If it isn't
set, no key is needed - that's for local development.

Tests: uv run pytest
"""

import os
import re
import secrets
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

from agent import agent_answer
from ask import answer

GAMES = ["Terraforming Mars", "Everdell", "Bang"]

# A citation in an answer, like [Everdell p.6] or [Terraforming Mars FAQ p.18].
CITATION = re.compile(r"\[(" + "|".join(re.escape(g) for g in GAMES) + r")[^\]]*? p\.(\d+)\]")

app = FastAPI(title="Board Game Rules Assistant")


class AskRequest(BaseModel):
    question: str
    game: str
    mode: Literal["single", "agent"] = "agent"

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, question):
        if not question.strip():
            raise ValueError("question must not be empty")
        return question.strip()

    @field_validator("game")
    @classmethod
    def known_game(cls, game):
        # Accept any capitalisation, but pass the canonical name on.
        for name in GAMES:
            if game.strip().lower() == name.lower():
                return name
        raise ValueError(f"unknown game {game!r}; valid games are: {', '.join(GAMES)}")


class Citation(BaseModel):
    game: str
    page: int


class ToolCall(BaseModel):
    tool: str
    game: str
    query: str


class Usage(BaseModel):
    model: str
    input_tokens: int | None
    output_tokens: int | None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCall] | None  # agent mode only
    usage: Usage
    seconds: float


def check_api_key(x_api_key: str | None = Header(default=None)):
    """Require X-API-Key to match RULES_API_KEY, but only if that is set."""
    expected = os.environ.get("RULES_API_KEY")
    if not expected:
        return
    # compare_digest takes the same time however much of the key matches,
    # so the response time gives nothing away. The key itself is never logged.
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key header")


def citations_in(text):
    """Return the [Game p.N] citations in an answer, each once, in order."""
    found = []
    for game, page in CITATION.findall(text):
        citation = {"game": game, "page": int(page)}
        if citation not in found:
            found.append(citation)
    return found


# Plain def, not async def: answer() and agent_answer() block while they wait
# for the model, so FastAPI runs each request in its own worker thread.
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/games", dependencies=[Depends(check_api_key)])
def games():
    return GAMES


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_api_key)])
def ask(request: AskRequest):
    start = time.time()
    try:
        if request.mode == "agent":
            result = agent_answer(request.question, request.game)
        else:
            result = answer(request.question, request.game)
    except SystemExit as error:
        # ask.py and agent.py stop with SystemExit when ANTHROPIC_API_KEY is
        # missing; in a server that has to become an error response instead.
        raise HTTPException(status_code=500, detail=str(error))

    tool_calls = None
    if request.mode == "agent":
        tool_calls = [
            {"tool": call["tool"], "game": call["game"], "query": call["query"]}
            for call in result["tool_calls"]
        ]
    return {
        "answer": result["text"],
        "citations": citations_in(result["text"]),
        "tool_calls": tool_calls,
        "usage": result["usage"],
        "seconds": round(time.time() - start, 1),
    }
